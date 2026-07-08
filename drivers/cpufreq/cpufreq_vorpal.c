// SPDX-License-Identifier: GPL-2.0
/*
 * Vorpal CPUFreq Governor - schedutil-derived, tuned for high-refresh gaming
 * and power-efficient daily use.
 *
 * Two operating profiles selected by the gaming_mode sysfs:
 *   - Gaming: per-cluster floors keep the render path warm, a built-in DRM
 *     present feed paces frames, and a bounded boost recovers dropped frames.
 *   - Daily:  power-efficient util-following with touch / UI responsiveness
 *     floors.
 *
 * Topology-aware (Little / Big / Prime by capacity), with a directional EMA util
 * filter, per-cluster rate limiting, an IO-wait boost, a proactive thermal step
 * controller, and optional scheduler coupling (kernel/sched/fair.c) via
 * sched_gaming_active. GKI-5.10 util interface lives in cpufreq_schedutil.c.
 */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/kthread.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/sched/topology.h>
#include <linux/rcupdate.h>
#include <linux/sched/rt.h>
#include <linux/sched/cpufreq.h>
#include <uapi/linux/sched/types.h>
#include <linux/tick.h>
#include <linux/timekeeping.h>
#include <linux/cpu.h>
#include <linux/cpufreq.h>
#include <linux/cpumask.h>
#include <linux/irq_work.h>
#include <linux/percpu.h>
#include <linux/spinlock.h>
#include <linux/mutex.h>
#include <linux/kobject.h>
#include <linux/string.h>
#include <linux/sysfs.h>
#include <linux/types.h>
#include <linux/input.h>
#include <linux/workqueue.h>
#include <linux/atomic.h>
#include <linux/list.h>
#ifdef CONFIG_THERMAL
#include <linux/thermal.h>
#endif

#define CPUFREQ_VORPAL_NAME     "vorpal"
#define CPUFREQ_VORPAL_VERSION  "2.1"
#define CPUFREQ_VORPAL_AUTHOR   "Steambot12"

/*
 * Scheduler coupling symbol. Defined and EXPORT_SYMBOL_GPL'd in
 * kernel/sched/fair.c (file-scope int, NOT a task_struct field -> KMI safe).
 * Written here by gaming_mode_store; read by fair.c gaming biases.
 */
extern int sched_gaming_active;

/* Core-sched util getter / deadline-bandwidth check (owned by core sched). */
extern void rfx_get_util_gki510(int cpu, unsigned long boost, bool gaming,
				unsigned long *util, unsigned long *bwmin);
extern bool rfx_dl_bw_exceeded_gki510(int cpu, unsigned long bwmin);

/*
 * Built-in frame pacing source. drm_vblank.c publishes the timestamp of the
 * last DELIVERED present/flip at its send_vblank_event() chokepoint (one event
 * = one presented frame), so frame timing is automatic whenever gaming_mode=1
 * and the panel flips - no userspace feeder, no root, no dumpsys. DRM only
 * publishes; the governor only consumes (no reverse dependency). Fallback to a
 * local zero atomic if DRM is not built in.
 */
#if IS_ENABLED(CONFIG_DRM)
extern atomic64_t drm_last_present_ns;
extern atomic64_t drm_present_seq;
#else
static atomic64_t drm_last_present_ns = ATOMIC64_INIT(0);
static atomic64_t drm_present_seq = ATOMIC64_INIT(0);
#endif

/*
 * Render-pressure window for the dynamic GPU floor, owned by the devfreq core
 * (always built). We publish a ktime deadline here whenever frames are sagging
 * or rendering just resumed; the devfreq GPU floor engages only while it is in
 * the future, so the GPU stays ready through the bursty asset-streaming phase
 * but idles (cool) once frames are smooth again.
 */
extern atomic64_t vorpal_gpu_boost_until_ns;
extern atomic64_t vorpal_gpu_hard_boost_until_ns;
/*
 * Dedicated cold-start / render-resume GPU floor deadline. The reactive soft/hard
 * windows above are per-sag pulses; the launch window holds a stronger GPU floor
 * across the whole bursty asset-streaming phase of a match spawn (the slow-ramping
 * part of a render start), and is consumed by the devfreq GPU floor exactly like
 * the other two. Owned by the devfreq core (always built).
 */
extern atomic64_t vorpal_gpu_launch_until_ns;

/* ===================================================================== */
/* Tunable defaults (KMI-safe: plain #defines, no struct-layout changes). */
/* ===================================================================== */

/* Cluster identification by arch capacity. */
#define RFX_LITTLE_CAP_THRESHOLD	614
#define RFX_PRIME_CAP_THRESHOLD		1000

/* Per-cluster rate limits (microseconds). up=0 means "scale up instantly". */
#define RFX_LITTLE_RATE_US		1000
#define RFX_LITTLE_UP_US		0
#define RFX_LITTLE_DOWN_US		4000

#define RFX_BIG_RATE_US			500
#define RFX_BIG_UP_US			0
#define RFX_BIG_DOWN_US			8000

#define RFX_PRIME_RATE_US		500
#define RFX_PRIME_UP_US			0
#define RFX_PRIME_DOWN_US		8000

/*
 * Gaming evaluation gate (microseconds). The per-cluster rate_limit_us above
 * throttles how often the governor RECOMPUTES a target at all. While gaming the
 * profile advertises "instant up" (up_rate_delay_ns = 0), but that only affects
 * the commit stage - the evaluation gate would still refuse to recompute for up
 * to rate_limit_us (500us Big/Prime, 1000us Little) after the last change, so a
 * mid-frame demand spike could wait most of a millisecond before the clock even
 * looks. At an 8.33ms/120Hz budget that is 6-12% of a frame of pure controller
 * dead-time on every ramp. Clamp the gaming eval gate to this much shorter value
 * so a spike is acted on within ~a quarter-ms. Only the *evaluation* throttle is
 * lifted; the actual transition shaping (instant-up, slow-decay down-rate,
 * thermal clamp) is unchanged, so this does not reintroduce yoyo. Bounded below
 * by the fact that the util hook only fires on scheduler events.
 */
#define RFX_GAMING_EVAL_US		250

/*
 * Gaming down-rate-limit (microseconds): while gaming, frequency may only step
 * DOWN this slowly. Combined with the floors, this is what kills the yoyo /
 * sawtooth - the clock holds the high band and decays gently.
 *
 * Stability profile: the render clusters (Big/Prime) step down very slowly so a
 * momentary render gap can't drop the clock (the main micro-drop cause). Paired
 * with the high floors below this flattens the FPS graph; LITTLE steps down a bit
 * faster since it only carries support work. The gaming EMA decay is cluster-aware
 * too (rfx_ema).
 */
#define RFX_GAMING_DOWN_US		60000
#define RFX_GAMING_LITTLE_DOWN_US	48000

/* ---- Gaming frequency band, percent of policy fmax ---- */
/*
 * All gaming caps are 100% (= no artificial ceiling). Frequency still follows
 * measured demand, while the floors below and frame-pressure windows keep the
 * render path warm enough to request the next OPP before the frame is already
 * late. A sub-100 cap would slow a saturated core, raising load% at fixed work
 * and starving frames - the opposite of "sustain". The thermal step clamp is
 * the real ceiling.
 *
 * Prime is a single-core cluster: the render thread hops in/out of it, so the
 * floor is adaptive - locked high only while Prime is genuinely busy, dropped
 * to a lower idle floor otherwise (up-rate is 0 so it snaps back instantly).
 */
#define RFX_G_PRIME_FLOOR_PCT		100	/* busy: pinned at fmax for a flat line */
/* idle: keep Prime hot so the render thread never lands on a relaxed core. */
#define RFX_G_PRIME_IDLE_FLOOR_PCT	95
/*
 * Latch busy-hold even on light activity so a bursty render thread lands on a
 * warm Prime. Single core, so the extra warmth is bounded to one CPU.
 */
#define RFX_G_PRIME_BUSY_ENTER_PCT	20
#define RFX_G_PRIME_CAP_PCT		100
/* Stronger recovery lift on a frame miss; still below fmax. */
#define RFX_G_PRIME_FRAME_PCT		96
/*
 * Busy-hold hysteresis: once Prime is busy, keep it locked at the high floor for
 * this long after the last busy sample. A bursty render thread pulses Prime load
 * above/below BUSY_ENTER; without a hold the floor would toggle down into every
 * inter-burst gap and oscillate the clock. The hold bridges those gaps; only a
 * genuinely parked Prime relaxes to the idle floor.
 */
#define RFX_G_PRIME_HOLD_NS		(1000 * NSEC_PER_MSEC)
/*
 * Cold-start warmup: for this long Prime is treated as busy so the first frames
 * after a launch do not land on a cold (idle-floored) core. Armed at
 * gaming_mode=1 and re-armed whenever rendering resumes after a present gap (see
 * rfx_frame_time_sample), which anchors the pre-warm to actual gameplay start
 * rather than the mode toggle. During the window Prime is forced busy, Little is
 * held at the sync floor, and Big is pre-lifted to its frame floor, so all render
 * clusters are warm when the first frames hit.
 */
#define RFX_G_PRIME_WARMUP_NS		(5000 * NSEC_PER_MSEC)
/*
 * Big carries most of the render load. Stability profile: hold it near fmax so a
 * brief render gap can't sag the clock into a micro-drop. The thermal step shave
 * (applied last) still glides it down smoothly before the vendor trip if the SoC
 * heats, so the line stays flat rather than cratering. Fixed builtin (no knob).
 */
#define RFX_G_BIG_FLOOR_PCT		94
#define RFX_G_BIG_CAP_PCT		100
#define RFX_G_BIG_FRAME_PCT		98	/* >floor: lift on a frame miss */
/*
 * Little: support work, no cap. A latency-sensitive worker stuck at fmin finishes
 * much slower than at a mid OPP and can land a frame late, so during gaming Little
 * holds a mid floor instead of dropping to fmin. The ENTER gate is low so only a
 * genuinely idle Little sleeps down. Fixed builtin (no knob).
 */
#define RFX_G_LITTLE_CAP_PCT		100
#define RFX_G_LITTLE_FLOOR_PCT		70
#define RFX_G_LITTLE_FLOOR_ENTER_PCT	12	/* apply floor unless near-idle */
#define RFX_G_LITTLE_FRAME_PCT		85	/* strong support-cluster recovery lift */
/*
 * Cross-cluster sync. When a perf cluster (Big or Prime) is saturated on the
 * frame's critical path, the Little support threads (audio, net, input, asset
 * decode) must keep pace or they bottleneck the frame. So when a perf cluster
 * crosses SAT_PCT, lift Little's floor to SYNC_FLOOR - but only while Little is
 * itself busy, so no clock is wasted on an idle Little and no extra heat lands on
 * the already-maxed perf clusters.
 */
#define RFX_G_PERF_SAT_PCT		92	/* a perf cluster this close to fmax = saturated */
#define RFX_G_LITTLE_SYNC_FLOOR_PCT	85	/* keep support threads in step */
/*
 * Little busy-hold: same hysteresis idea as Prime. A momentary util dip below
 * FLOOR_ENTER would otherwise drop the floor and let Little crash to fmin
 * mid-gameplay; holding the floor briefly after the last busy sample bridges the
 * micro-idle gaps.
 */
#define RFX_G_LITTLE_HOLD_NS		(120 * NSEC_PER_MSEC)
/*
 * Gaming iowait ceiling. Heavy titles spend a lot of time in asset/shader
 * streaming and wake-from-IO worker bursts, but letting every gaming iowait wake
 * climb to full capacity can turn small streaming workers into repeated fmax
 * requests. Keep the feature active, but shape it by cluster so it helps wake to
 * render without adding sustained CPU-load spikes.
 */
#define RFX_G_IOWAIT_BOOST_PCT		76
#define RFX_G_IOWAIT_LITTLE_CAP_PCT	48
#define RFX_G_IOWAIT_BIG_CAP_PCT	74
#define RFX_G_IOWAIT_PRIME_CAP_PCT	76

/*
 * Closed-loop thermal redistribution (OFF by default). When the vendor thermal
 * framework throttles the prime/render core (lowers its policy->max below
 * RFX_PRIME_THROTTLE_DETECT_PCT of the hardware max), the SoC is heat-limited;
 * this can cap the over-saturated LITTLE cluster to free thermal budget so the
 * junction cools and the throttle releases sooner.
 *
 * Default DISABLED: the detection assumes prime's policy->max equals the
 * hardware max in the unthrottled state, which is not true on every device (some
 * keep a mild standing cap), so leaving it on risked capping LITTLE during
 * normal play and stalling support threads. It is only useful against a genuine
 * SUSTAINED thermal throttle. Fixed OFF builtin (RFX_G_SHED_LITTLE_PCT = 0); it
 * was too device-specific to safely leave on by default, so with the tuning
 * knobs removed the LITTLE shed stays disabled.
 */
#define RFX_PRIME_THROTTLE_DETECT_PCT	96	/* prime policy->max below this % of fmax = vendor is throttling it */
#define RFX_G_SHED_LITTLE_PCT		0	/* default off (see above); cap LITTLE to this % of fmax while prime throttled */

/* ---- Daily frequency shaping, percent of policy fmax ---- */
/*
 * "UI active" = a touch is recent OR a render burst was just detected (see
 * the ramp-assist below). When active, LITTLE's cap is relaxed and LITTLE/Big
 * get a responsiveness floor so animations / captions / scrolls do not run at
 * a starved OPP. When idle, LITTLE is capped low for battery.
 */
/*
 * Daily LITTLE idle cap. Kept high enough that gradually-rising UI work the
 * UI-active heuristic misses (a smooth animation, a caption redraw, sustained
 * moderate drawing) is not starved, while at true idle util-following keeps the
 * freq low regardless of the cap. UI-active relaxes the cap further.
 */
#define RFX_D_LITTLE_CAP_PCT		82	/* idle cap: high enough for keyboard, captions, gentle scrolls */
#define RFX_D_LITTLE_BOOST_CAP_PCT	100	/* no cap while UI active: a burst (unlock, app open, fling) must be able to ramp LITTLE fully or it stutters; the cap only bites at idle for battery */
#define RFX_D_LITTLE_UI_FLOOR_PCT	58	/* LITTLE floor while UI active */
#define RFX_D_BIG_UI_FLOOR_PCT		55	/* Big floor while UI active */

/*
 * Daily UI ramp-assist. PELT/WALT util lags the real frame work, and the most
 * stutter-prone UI moments (a video caption appearing, an app open/close
 * animation, a fling-scroll) are often NOT touch events, so a touch-only boost
 * misses them. Instead we watch the smoothed util for a sharp RISE: a jump of
 * >= RFX_D_RAMP_DELTA_PCT points arms a short floor that holds for
 * RFX_D_UI_BOOST_NS, re-arming as long as demand keeps climbing. This lifts the
 * first frames of a burst immediately, independent of input.
 */
#define RFX_D_RAMP_DELTA_PCT		8	/* lower threshold catches gradual UI work (keyboard, video captions) */
#define RFX_D_UI_BOOST_NS		(200 * NSEC_PER_MSEC)	/* hold UI floor through a full app transition / keyboard draw */

/*
 * Touch window: long enough that a tap-initiated app open/close animation
 * (~300 ms) stays boosted through the whole transition instead of sagging
 * halfway.
 */
#define RFX_INPUT_WINDOW_NS		(500 * NSEC_PER_MSEC)

/* ---- Util EMA (directional smoothing). new>old: up_shift, else down_shift ---- */
#define RFX_EMA_UP_SHIFT_DAILY		0	/* instant attack: a render-thread burst (texture upload / compositing) gets clock immediately -> shorter RT spike -> less daily frame-time jitter. Decay stays slow (down-shift 3), so ~no battery cost */
#define RFX_EMA_DN_SHIFT_DAILY		2	/* faster decay: prevents thermal creep on long daily use while still bridging inter-frame gaps */
#define RFX_EMA_UP_SHIFT_GAMING		0	/* instant attack: no lag on a render spike */
#define RFX_EMA_DN_SHIFT_GAMING		3	/* render (Big/Prime) decays slowly -> anti-jitter / steady frame time */
#define RFX_EMA_DN_SHIFT_GAMING_LITTLE	3	/* universal: LITTLE decays like render clusters because some ROMs keep frame-critical support work there */

/*
 * WALT-like window (gaming). PELT util is a geometric average that lags a
 * step-change in demand (enemies spawn / a heavy animation starts) by tens of
 * ms - long enough for the first frames of the burst to render at a low OPP
 * before the average catches up. That lag is the single biggest reason a PELT
 * governor feels a beat behind Qualcomm's WALT on bursts. Emulate WALT's window
 * model cheaply: capture the MAX effective util over a short window and feed
 * that to the EMA, so the clock sees the burst at full magnitude on the first
 * sample and holds it across the window instead of chasing a lagging mean. The
 * peak is bounded by real measured demand and re-based every window, so unlike
 * a static floor it adds no steady-state heat - the exact failure mode of the
 * reverted "broader boost" iteration.
 *
 * ~16ms (about two 120Hz frames) is long enough to bridge inter-frame idle gaps
 * yet short enough that the clock still tracks demand down when the burst ends.
 */
#define RFX_WALT_WINDOW_NS		(16 * NSEC_PER_MSEC)

/*
 * ---- Headroom ----
 * The scheduler helper returns raw aggregate demand; Vorpal applies the
 * schedutil-style 25% DVFS margin here, after the stale-feed pressure gates have
 * sampled demand. That keeps normal frequency selection responsive without
 * letting headroom-inflated util arm fallback boosts too early. Daily keeps the
 * same base margin plus a small tiered curve for UI response.
 */

/* ---- Thermal step controller ---- */
#define RFX_THERMAL_STEP_NS		(6 * NSEC_PER_MSEC)
#define RFX_THERMAL_STEP_DOWN_PCT	2
/*
 * Recover the cap at the same rate it is shaved. An asymmetric slow-up (1) under
 * a skin sensor that dithers +/-0.5C ratchets the cap downward: each warm blip
 * shaves 2% but only 1% is given back per step, so noisy-but-flat temperatures
 * drift the clamp low and it lingers there - a visible FPS droop that does not
 * recover. Up-steps are still gated by the EMA-smoothed temperature, so matching
 * the down rate only removes the ratchet bias; it never lifts freq above what the
 * measured temperature already allows (no extra heat).
 */
#define RFX_THERMAL_STEP_UP_PCT		2
#define RFX_THERMAL_MIN_CAP_PCT		70
#define RFX_THERMAL_POLL_GAMING_MS	50
#define RFX_THERMAL_POLL_IDLE_MS	200
/*
 * Temperature breakpoints (milli-Celsius) -> target cap percent, for a SKIN /
 * board sensor (auto-bound on gaming_mode=1). The goal is a gentle
 * pre-emptive shave that begins before the vendor thermal governor reaches its
 * hard policy->max trip, so heat is bled off smoothly instead of in FPS-crashing
 * slams: hold 100% below GREEN, -2%/C to YELLOW, -3%/C to RED, then the floor
 * (RFX_THERMAL_MIN_CAP_PCT guards it) while the vendor framework handles anything
 * beyond. GREEN sits above the normal gaming skin temp so a healthy frame is
 * never shaved.
 *
 * Use a SKIN/board sensor, not a CPU-junction zone - junction reads far hotter
 * in normal play and would throttle mid-session.
 */
/*
 * Breakpoints pulled ~4C lower than the original 48/52/56 after device
 * telemetry showed CPU temp climbing to 48-50C and the VENDOR thermal governor
 * then slamming policy->max mid-burst - producing deep FPS craters (min-FPS 78)
 * even at only ~59% CPU load (i.e. NOT capacity-bound; heat-bound). Starting
 * Vorpal's own smooth -2%/C shave at 44C bleeds heat off gently BEFORE the die
 * reaches the vendor's hard trip, so the clock glides down a couple percent
 * (invisible) instead of the vendor crashing it (a visible 40-FPS drop). Net
 * effect is a HIGHER min-FPS despite the earlier shave. GREEN still sits above
 * a healthy ~40C gaming skin temp so on-cadence frames are never touched.
 */
#define RFX_TEMP_GREEN_MC		44000	/* hold 100% through normal play */
#define RFX_TEMP_YELLOW_MC		48000	/* -2%/C below, -3%/C above */
#define RFX_TEMP_RED_MC			53000	/* floor reached; vendor beyond */

/* ---- Frame pacing ---- */
#define RFX_FRAME_BUDGET_US_120		8333	/* 1e6/120 */
#define RFX_FRAME_BUDGET_US_100		10000	/* 1e6/100 */
#define RFX_FRAME_BUDGET_US_90		11111	/* 1e6/90  */
#define RFX_FRAME_BUDGET_US_60		16666	/* 1e6/60  */
/*
 * Default gaming frame budget. Use the release-stable 120fps cadence as the
 * universal gaming target: gaming_mode=1 should prefer missed-frame recovery
 * over a device-specific cool profile. Lower targets can still be selected live
 * via frame_budget_us (10000=100fps, 11111=90fps, 16666=60fps), which sets
 * rfx_frame_budget_user_set so gaming-on will not reset it.
 */
#define RFX_FRAME_BUDGET_US_GAMING	RFX_FRAME_BUDGET_US_120
/*
 * Recovery-boost window. Short on purpose: a title that chronically misses its
 * target would, with a long window, keep the boost permanently armed and pin the
 * clusters high (heat -> vendor throttle -> the very drops it tried to prevent).
 * This window spans several 120Hz frames, long enough to bridge a burst-collapse
 * recovery without becoming a sustained latch.
 */
#define RFX_FRAME_BOOST_NS		(112 * NSEC_PER_MSEC)
/*
 * GPU-floor window (consumed by the devfreq dynamic GPU floor). Longer than the
 * CPU recovery boost because GPU devfreq polls slowly and the GPU ramp is the
 * slow part: a single window must bridge dense, bursty render phases (crowded
 * combat, asset streaming, many players entering view). Armed on a frame
 * sag/jank and on render-resume after a present gap.
 */
#define RFX_GPU_BOOST_NS		(600 * NSEC_PER_MSEC)
/* Short high GPU-floor pulse layered on top of the longer soft floor. */
#define RFX_GPU_HARD_BOOST_NS		(180 * NSEC_PER_MSEC)
/*
 * Launch / render-resume GPU floor window. The plain RFX_GPU_BOOST_NS (600ms) is
 * a reactive per-jank pulse; a cold render start needs the GPU held ready through
 * the whole bursty asset-streaming phase, which routinely runs 1-3s. Bridge that
 * with a dedicated launch window - longer than the reactive pulse, but well under
 * the 5s CPU warmup so it does NOT add sustained GPU heat (a full-warmup GPU floor
 * regressed thermals). Armed alongside the Prime warmup (see rfx_arm_launch_window).
 *
 * Widened to span the documented 1-3s asset-streaming burst: at 1500ms the GPU
 * floor lapsed mid-burst (while the 5s CPU warmup was still holding the clusters
 * warm), so the GPU downshifted right as the first heavy frames streamed in -
 * the classic "loads then jitters, fps dips at match spawn". 3000ms bridges the
 * whole burst yet still ends well under the 5s CPU warmup, so it adds no
 * steady-state GPU heat.
 */
#define RFX_GPU_LAUNCH_NS		(3000 * NSEC_PER_MSEC)
/*
 * Strong hard GPU pulse for the very first spawn frames (the heaviest of the
 * launch). Longer than the 180ms reactive hard pulse because the spawn burst is
 * a known, bounded, one-shot event; after it the launch soft floor carries the
 * streaming tail. Time-bounded -> no steady-state heat.
 */
#define RFX_GPU_LAUNCH_HARD_NS		(500 * NSEC_PER_MSEC)
/*
 * Device-agnostic render-resume detector. The DRM present feed is the only path
 * that currently re-arms the cold-start warmup (rfx_frame_time_sample), but many
 * ROMs / MediaTek display stacks never deliver DRM_EVENT_FLIP_COMPLETE, so on those
 * devices render-start protection never fires and the first frames of a match land
 * on cold clusters + a cold GPU. Instead, watch the scheduler util of the perf
 * clusters (always available): when a perf cluster goes busy after being quiet for
 * longer than this, treat it as a render-start / resume edge and arm the launch
 * window. Chosen a little below RFX_FRAME_PRESENT_GAP_NS so a brief scene stall
 * that the DRM path would miss still re-warms the render path.
 */
#define RFX_RENDER_RESUME_GAP_NS	(300 * NSEC_PER_MSEC)
/*
 * Anti-runaway gate. Only arm the recovery boost if a perf cluster still has
 * frequency headroom (below this percent of fmax). A frame miss while the perf
 * clusters are already near fmax is GPU / IO / thermal / placement bound, not
 * frequency bound, so boosting would only add heat. Gating here means the boost
 * fills in clock only where recovery is physically possible (cold start, scene
 * transitions) and never pins an already-busy cluster.
 */
#define RFX_FRAME_BOOST_GATE_PCT	94
#define RFX_FRAME_BOOST_BUSY_PCT	98
/* Ignore present intervals longer than this (app switch / resume / paused). */
#define RFX_FRAME_PRESENT_GAP_NS	(500 * NSEC_PER_MSEC)
#define RFX_FRAME_FEED_STALE_NS		(500 * NSEC_PER_MSEC)
#define RFX_DRM_SAMPLE_RETRIES		3
#define RFX_JANK_WINDOW_NS		(2000 * NSEC_PER_MSEC)

/*
 * Proactive frame-margin sustain. The reactive boost above only fires once a
 * frame has already overrun 1.5x budget (a real jank). To hold a steady target
 * we also catch frames that are merely sliding below it. A smoothed frame time
 * (rfx_ft_ema_us) arms the same bounded, gated boost as soon as it creeps past
 * this percent of budget. When the game is on-cadence the smoothed frame time
 * sits near 100% of budget, below this threshold, so the feature stays off and
 * the clusters race-to-idle - power is only spent on the edge. Raise it to make
 * the controller lazier, lower it to defend the target harder. The headroom gate,
 * the boost window, and the below-fmax boost floors keep it bounded. With the
 * default 120fps budget, 104% arms at ~115fps - i.e. CPU recovery engages just as
 * the smoothed frame time approaches the 115fps sustain floor, before it drops
 * under, giving the clock time to recover rather than reacting after the visible
 * dip. The softer GPU window (102%, ~118fps) still catches smaller slips first, so
 * the cheaper GPU-readiness floor leads and the hotter CPU boost follows.
 */
/* Arm proactively when EMA frame time crosses this percent of budget. */
#define RFX_FRAME_SUSTAIN_PCT		104
/* Raw single-frame drop: recover before it becomes a 1.5x jank. */
#define RFX_FRAME_NEAR_MISS_PCT		130
/*
 * WALT-like proactive ramp threshold. The governor follows PELT util, which lags
 * a sudden demand spike (a scene change, an animation start, dropping into a dense
 * combat area) by tens of ms - long enough for the first frames of the spike to
 * render late before the clock catches up. That lag is exactly why a PELT governor
 * feels a beat behind a WALT one. So watch the per-cluster demand for a sharp RISE:
 * when the post-headroom util jumps by >= this many points in one sample, pre-arm
 * the frame boost + GPU window immediately - ahead of the late frame - so the
 * render path ramps ON the spike instead of one frame after the drop. Edge-
 * triggered: steady play never crosses it, so no extra steady-state heat. Lower to
 * catch gentler animation bursts, raise to fire only on hard scene changes.
 */
#define RFX_G_RAMP_DELTA_PCT		6
#define RFX_FT_EMA_SHIFT		2	/* frame-time EMA: alpha = 1/4 (anti single-frame noise) */
/*
 * Adaptive per-game cadence (universal mixed-FPS support). The sustain target
 * defaults to 120fps, but a title that caps at 90/60fps would otherwise be scored
 * as a chronic jank every frame -> the recovery boost stays permanently armed,
 * burning heat that trips the vendor thermal governor and causes the very drops it
 * was meant to prevent. So detect the game's real steady cadence and relax the
 * effective budget to it - but ONLY when the cadence is genuinely stable (a true
 * cap), never when frames are merely struggling below a higher target (which must
 * still be recovered). A frame counts as "stable" when the raw present interval is
 * within RFX_CADENCE_STABLE_PCT of its own EMA; after RFX_CADENCE_STABLE_FRAMES
 * consecutive stable frames sitting at/under a slower standard cadence, that
 * standard becomes the effective budget. Any jittery frame, or the EMA climbing
 * back toward the configured target, resets the detection. Disabled while the user
 * has pinned frame_budget_us.
 */
/*
 * Hardened so a 120-target title dipping into a busy patch cannot be mistaken for
 * a slower cap and stop being defended. Symptom: PUBG at 120 sags to ~104fps for
 * a couple of seconds, the old 48-frame (~0.4s) / 96%-match / 12%-stable gate
 * latched the 100fps budget, sustain then scored against 100 and the clock
 * settled there instead of recovering to 110-117. Now a cap must be genuinely
 * sustained (~1.5s), matched tightly, and jitter-free before it is accepted;
 * a real 90/60/100-locked title still runs steady far longer than this and
 * relaxes normally (which keeps the boost from pinning and cooking the SoC).
 */
#define RFX_CADENCE_STABLE_PCT		10	/* |raw-ema| within this % of ema = stable frame */
#define RFX_CADENCE_STABLE_FRAMES	180	/* consecutive stable frames to accept a slower cap */
#define RFX_CADENCE_MATCH_PCT		98	/* ema must reach this % of a standard cadence to snap to it */
/*
 * GPU-only sustain edge. Crowded scenes often slide below the panel cadence
 * before the CPU clusters need another recovery floor. Arm only the devfreq GPU
 * readiness window at this lighter edge; the CPU boost still waits for
 * RFX_FRAME_SUSTAIN_PCT / real jank so it does not add unnecessary die heat.
 */
#define RFX_GPU_SUSTAIN_PCT		101
/*
 * Severe mid-game burst (e.g. a PUBG match landing: the map region streams in and
 * many players spawn into view, collapsing the smoothed frame rate to ~70-80fps).
 * This is not a cold start - the render clusters are already busy, so neither the
 * DRM present-gap re-arm nor the util quiet->busy resume detector fires, and the
 * burst only gets the 600ms/soft reactive GPU pulse. When the smoothed frame time
 * crosses this percent of budget, give the GPU the stronger launch-class floor
 * instead. GPU-ONLY on purpose: it must NOT re-arm the 5s Prime CPU warmup, or a
 * long combat sequence would hold the perf clusters warm continuously and cook the
 * die (the reverted-floor failure mode). The GPU floor is a minimum, so it is a
 * no-op whenever real GPU demand already exceeds it; it only bridges the inter-
 * burst dips. Rate-limited and time-bounded -> no steady-state heat.
 */
#define RFX_GPU_BURST_PCT		140	/* EMA frame time > this % of budget = heavy burst */
#define RFX_GPU_BURST_REFRESH_NS	(600 * NSEC_PER_MSEC)	/* min spacing between re-arms */
/*
 * CPU-pressure fallback for ROMs whose display path does not reliably publish
 * frame-present events. A moderate perf-cluster load can keep the GPU softly
 * warm, but CPU recovery needs higher pressure and is pulsed slowly so stale
 * feeds cannot become a permanent hard boost.
 */
#define RFX_STALE_GPU_PRESSURE_PCT	76
#define RFX_STALE_GPU_REFRESH_NS	(300 * NSEC_PER_MSEC)
#define RFX_CPU_PRESSURE_PCT		94
#define RFX_CPU_PRESSURE_REFRESH_NS	(320 * NSEC_PER_MSEC)
#define RFX_VENDOR_THROTTLE_PCT		94

#define IOWAIT_BOOST_MIN		(SCHED_CAPACITY_SCALE / 8)

enum rfx_cluster_type {
	RFX_CLUSTER_LITTLE = 0,
	RFX_CLUSTER_BIG,
	RFX_CLUSTER_PRIME,
};

/* ===================================================================== */
/* Global state                                                          */
/* ===================================================================== */

/* Master gaming switch, written by gaming_mode sysfs (Prime cluster only). */
static atomic_t rfx_gaming = ATOMIC_INIT(0);

static inline bool rfx_gaming_enabled(void)
{
	return atomic_read(&rfx_gaming) != 0;
}

/*
 * Gaming profile values. Fixed builtin: initialised from the #define defaults
 * above and never written at runtime (the per-cluster tuning knobs were removed -
 * the profile is baked in). Kept as atomic_t so the lockless/raw-spin fast path
 * reads them with clean cross-CPU visibility and no fast-path churn.
 */
static atomic_t rfx_g_down_us       = ATOMIC_INIT(RFX_GAMING_DOWN_US);
static atomic_t rfx_g_little_down_us = ATOMIC_INIT(RFX_GAMING_LITTLE_DOWN_US);
static atomic_t rfx_g_big_floor_pct  = ATOMIC_INIT(RFX_G_BIG_FLOOR_PCT);
static atomic_t rfx_g_prime_floor_pct = ATOMIC_INIT(RFX_G_PRIME_FLOOR_PCT);
static atomic_t rfx_g_prime_idle_floor_pct =
	ATOMIC_INIT(RFX_G_PRIME_IDLE_FLOOR_PCT);
static atomic_t rfx_g_little_floor_pct = ATOMIC_INIT(RFX_G_LITTLE_FLOOR_PCT);
static atomic_t rfx_g_shed_little_pct = ATOMIC_INIT(RFX_G_SHED_LITTLE_PCT);
static atomic_t rfx_g_gpu_sustain_pct = ATOMIC_INIT(RFX_GPU_SUSTAIN_PCT);
static atomic_t rfx_g_stale_pressure_pct =
	ATOMIC_INIT(RFX_CPU_PRESSURE_PCT);
static atomic_t rfx_g_iowait_boost_pct =
	ATOMIC_INIT(RFX_G_IOWAIT_BOOST_PCT);
/* WALT-like proactive-ramp rise threshold (percent points; 0 = disable). */
static atomic_t rfx_g_ramp_delta_pct = ATOMIC_INIT(RFX_G_RAMP_DELTA_PCT);

/* Last input event timestamp (daily touch boost). */
static atomic64_t rfx_input_ts_ns = ATOMIC64_INIT(0);

/* Thermal: target cap published by the poller, consumed by the fast path. */
static atomic_t rfx_thermal_cap_pct = ATOMIC_INIT(100);
/* Userspace-fed temperature fallback (milli-Celsius); 0 = unavailable. */
static atomic_t rfx_temp_mc = ATOMIC_INIT(0);

/* Frame pacing telemetry (userspace feeder writes frame_time_us). */
static atomic_t rfx_frame_time_us = ATOMIC_INIT(0);
static atomic_t rfx_frame_budget_us = ATOMIC_INIT(RFX_FRAME_BUDGET_US_GAMING);
/* True once userspace pins frame_budget_us via sysfs -> gaming won't auto-set it. */
static bool rfx_frame_budget_user_set;
static atomic_t rfx_frames_seen = ATOMIC_INIT(0);
static atomic_t rfx_janks_seen = ATOMIC_INIT(0);
static atomic_t rfx_jank_pct = ATOMIC_INIT(0);
/* Smoothed frame time (us) for the proactive sustain controller. 0 = unset. */
static atomic_t rfx_ft_ema_us = ATOMIC_INIT(0);
/*
 * Adaptive cadence detection state. rfx_cadence_bud_us is the detected effective
 * budget (us) once a stable slower-than-configured cap is confirmed, else 0 (use
 * the configured budget). rfx_cadence_stable is the consecutive-stable-frame
 * counter. Both are only touched from rfx_frame_account, whose body runs for one
 * CPU at a time (each present is claimed once via cmpxchg), so there is no fast-
 * path contention; kept atomic for clean cross-CPU visibility and reset from sysfs.
 */
static atomic_t rfx_cadence_bud_us = ATOMIC_INIT(0);
static atomic_t rfx_cadence_stable = ATOMIC_INIT(0);

/*
 * Frame-miss boost deadline (ns since boot), GLOBAL so every cluster reacts to
 * a dropped frame - not just Prime. A missed frame means SOME cluster was too
 * slow; since the render thread's placement is not known in-kernel, all of
 * Prime/Big/Little lift their floor together until this deadline. This is a
 * ktime deadline because it is shared across policies and can be armed from
 * either the DRM present feed or the scheduler update hook.
 */
static atomic64_t rfx_frame_boost_end_ns = ATOMIC64_INIT(0);

/* Deadline until which Prime is force-held high after gaming starts (cold-start). */
static atomic64_t rfx_gaming_warmup_end_ns = ATOMIC64_INIT(0);

/* Rate-limit for the severe-burst GPU launch-floor re-arm (see rfx_frame_account). */
static atomic64_t rfx_gpu_burst_next_ns = ATOMIC64_INIT(0);

/*
 * Frame accounting runs from the fast path on every gaming cluster, far faster
 * than the frame rate. rfx_last_seq_consumed claims each present exactly once
 * (cmpxchg on the DRM present sequence); rfx_prev_present_ns holds the timestamp
 * of the last claimed present so the next sample can measure the interval.
 */
static atomic64_t rfx_last_seq_consumed = ATOMIC64_INIT(0);
static atomic64_t rfx_prev_present_ns = ATOMIC64_INIT(0);
static atomic64_t rfx_last_frame_feed_ns = ATOMIC64_INIT(0);
static atomic64_t rfx_stale_gpu_next_ns = ATOMIC64_INIT(0);
static atomic64_t rfx_cpu_pressure_next_ns = ATOMIC64_INIT(0);
/*
 * Last time (ktime ns) a perf cluster (Big/Prime) was seen busy while gaming.
 * The util-driven render-resume detector reads this to spot a busy edge after a
 * quiet gap and re-arm the launch window - device-agnostic, no DRM feed needed.
 */
static atomic64_t rfx_last_perf_busy_ns = ATOMIC64_INIT(0);

/*
 * Cross-migration demand carry (gaming). The render thread hops between Big and
 * Prime; the destination cluster's per-CPU filt_util starts cold, so the first
 * frames after a migration render at a low OPP - a WALT governor avoids this by
 * carrying the task's window demand across the migration. Emulate it at the
 * cluster level: publish the most recent perf-cluster window peak (as a percent
 * of that cluster's capacity) here, and seed a going-busy perf cluster's filter
 * from it so the render thread lands on an already-warm core. Percent units so
 * it is portable across the Big/Prime capacity difference. Bounded by the same
 * window decay, so it cannot pin heat.
 */
static atomic_t rfx_perf_demand_pct = ATOMIC_INIT(0);

/*
 * Per-perf-cluster requested frequency as a percent of fmax, published each
 * gaming update. The frame-boost gate reads BOTH: it arms only when at least
 * one perf cluster still has headroom (min < gate). A miss while BOTH are near
 * fmax is GPU/IO/thermal/placement bound (boost would be heat-only); a miss
 * while one cluster is cold (e.g. the render thread just migrated onto an
 * idle-floored Prime) is exactly the recoverable case the boost exists for.
 * 0 until that cluster runs (treated as headroom = arm, the safe default).
 */
static atomic_t rfx_sys_perf_pct = ATOMIC_INIT(0);	/* Big (render) cluster */
static atomic_t rfx_prime_perf_pct = ATOMIC_INIT(0);	/* Prime cluster */

/*
 * Prime cluster policy->max as a percent of its hardware max, published each
 * gaming update. Below RFX_PRIME_THROTTLE_DETECT_PCT means the vendor thermal
 * framework is throttling the render core; the LITTLE shed (above) reads this.
 */
static atomic_t rfx_sys_capped_pct = ATOMIC_INIT(100);
static atomic_t rfx_prime_capped_pct = ATOMIC_INIT(100);

/* All live policies, so gaming-off can reset every cluster (not just Prime). */
static LIST_HEAD(rfx_policy_list);
static DEFINE_SPINLOCK(rfx_policy_list_lock);

/* ===================================================================== */
/* Data structures                                                       */
/* ===================================================================== */

struct rfx_tunables {
	struct gov_attr_set attr_set;
	unsigned int rate_limit_us;
	unsigned int up_rate_limit_us;
	unsigned int down_rate_limit_us;
	enum rfx_cluster_type cluster_type;
	unsigned int gaming_mode;
};

struct rfx_policy {
	struct cpufreq_policy *policy;
	struct rfx_tunables *tunables;
	struct list_head tunables_hook;
	struct list_head gov_node;	/* on rfx_policy_list */

	raw_spinlock_t update_lock;

	u64 last_upfreq_time;
	u64 last_downfreq_time;
	s64 freq_update_delay_ns;
	s64 up_rate_delay_ns;
	s64 down_rate_delay_ns;

	unsigned int next_freq;
	unsigned int cached_raw_freq;

	struct irq_work irq_work;
	struct kthread_work work;
	struct mutex work_lock;
	struct kthread_worker worker;
	struct task_struct *thread;
	bool work_in_progress;

	bool limits_changed;
	bool need_freq_update;

	bool is_prime;			/* this policy is the Prime cluster */
	bool is_little;

	unsigned int prev_upct;		/* last util%, for daily + gaming ramp detect */
	u64 ui_boost_end_ns;		/* daily: UI render-burst floor hold */
	u64 prime_busy_hold_ns;		/* gaming: hold Prime high across burst gaps */
	u64 little_busy_hold_ns;	/* gaming: hold LITTLE floor across micro-idle */

	int thermal_applied_pct;	/* walked toward rfx_thermal_cap_pct */
	u64 thermal_step_ns;
};

struct rfx_cpu {
	struct update_util_data update_util;
	struct rfx_policy *rfx_policy;
	unsigned int cpu;

	bool iowait_boost_pending;
	unsigned int iowait_boost;
	u64 last_update;

	unsigned long util;
	unsigned long bwmin;
	unsigned long filt_util;	/* directional EMA of effective util */

	/*
	 * WALT-like window max-capture (gaming only). win_peak holds the highest
	 * effective util seen since win_start_ns; every RFX_WALT_WINDOW_NS the
	 * window re-bases to the current sample. Feeding the EMA the window peak
	 * (instead of the instantaneous PELT read) holds the clock steady across
	 * inter-frame gaps - the anti-jitter WALT gets from its window model -
	 * WITHOUT a static floor's heat, because the peak can never exceed
	 * measured demand and is dropped every window.
	 */
	u64 win_start_ns;
	unsigned long win_peak;
};

static DEFINE_PER_CPU(struct rfx_cpu, rfx_cpu);

static inline struct rfx_tunables *to_rfx_tunables(struct gov_attr_set *attr_set)
{
	return container_of(attr_set, struct rfx_tunables, attr_set);
}

static inline struct gov_attr_set *rfx_to_gov_attr_set(struct kobject *kobj)
{
	return container_of(kobj, struct gov_attr_set, kobj);
}

static inline bool rfx_cap_is_little(unsigned long cap)
{
	return cap <= (unsigned long)RFX_LITTLE_CAP_THRESHOLD;
}

static inline bool rfx_cap_is_prime(unsigned long cap)
{
	return cap >= (unsigned long)RFX_PRIME_CAP_THRESHOLD;
}

/* fmax * pct / 100 */
static inline unsigned int rfx_pct(unsigned int fmax, unsigned int pct)
{
	return (unsigned int)((u64)fmax * pct / 100);
}

static inline bool rfx_input_active(void)
{
	u64 ts = (u64)atomic64_read(&rfx_input_ts_ns);

	/*
	 * rfx_input_ts_ns is stamped from the input handler with ktime_get_ns()
	 * (CLOCK_MONOTONIC), so the window must be measured against ktime too -
	 * not the scheduler's rq_clock update time, which is a different clock
	 * domain on arm64 and would make the touch window read as expired.
	 */
	return ts && (ktime_get_ns() - ts) < RFX_INPUT_WINDOW_NS;
}

/* ===================================================================== */
/* Util smoothing                                                        */
/* ===================================================================== */

/*
 * Directional EMA. Rising demand is tracked quickly (small up_shift) so a new
 * frame is not starved; falling demand decays slowly (larger down_shift) so the
 * clock does not chase every micro-dip. When the decayed step rounds down to
 * zero the filter snaps the last few units to val so it settles exactly instead
 * of dithering by one unit forever.
 */
static unsigned long rfx_ema(unsigned long old, unsigned long val, bool gaming,
			     bool little)
{
	unsigned long up = gaming ? RFX_EMA_UP_SHIFT_GAMING : RFX_EMA_UP_SHIFT_DAILY;
	unsigned long dn = gaming ? RFX_EMA_DN_SHIFT_GAMING : RFX_EMA_DN_SHIFT_DAILY;
	unsigned long diff;

	/*
	 * Hybrid decay: while gaming, LITTLE (support work) decays fast so it
	 * races to idle and banks thermal headroom for the render clusters,
	 * which keep the slow anti-jitter decay.
	 */
	if (gaming && little)
		dn = RFX_EMA_DN_SHIFT_GAMING_LITTLE;

	if (!old)
		return val;
	if (val > old) {
		diff = (val - old) >> up;
		return diff ? old + diff : val;
	}
	if (val < old) {
		diff = (old - val) >> dn;
		return diff ? old - diff : val;
	}
	return val;
}

/*
 * WALT-like window max-capture. Returns the highest effective util seen over the
 * current RFX_WALT_WINDOW_NS window, then re-bases the window to the latest
 * sample. Feeding this to the EMA (gaming only) gives the clock WALT's key
 * property - it sees a demand burst at full magnitude on the first sample and
 * holds it across inter-frame gaps - without a floor's steady-state heat,
 * because the peak is bounded by measured demand and dropped every window.
 *
 * `time` is rq_clock (the util-hook timebase); the window is compared against it
 * consistently, so no cross-clock mismatch. Called once per CPU per update from
 * the fast path, before rfx_ema.
 */
static unsigned long rfx_walt_window(struct rfx_cpu *rfx_c, unsigned long eff,
				     u64 time)
{
	if (!rfx_c->win_start_ns ||
	    (s64)(time - rfx_c->win_start_ns) >= (s64)RFX_WALT_WINDOW_NS) {
		/*
		 * Window elapsed: start a fresh one, but carry a decayed fraction
		 * (7/8) of the prior peak instead of hard-rebasing to the raw
		 * sample. A demand dip aligned with the window boundary - e.g. an
		 * I/O stall that dequeues the render thread and deflates PELT util -
		 * would otherwise collapse the held demand in a single step and
		 * inject a ~60Hz downclock jitter. The carry bridges the stall while
		 * still decaying every window (~1/8 per 16ms, ~halved by 5 windows,
		 * gone in ~10 / ~166ms), so it stays bounded by measured demand and
		 * adds no steady-state heat. Sized for the deeper write-stall a
		 * read-priority I/O scheduler produces (a strong read bias defers
		 * writes long enough that the render thread's sync writes block past
		 * the old ~80ms 3/4 bridge -> demand collapse -> the stall reads as a
		 * jank and needlessly arms the recovery boost); a scheduler with
		 * shallower stalls never deflates long enough to exercise the extra
		 * carry, so its pacing is unchanged.
		 */
		unsigned long carry = rfx_c->win_peak - (rfx_c->win_peak >> 3);

		rfx_c->win_start_ns = time;
		rfx_c->win_peak = max(eff, carry);
		return rfx_c->win_peak;
	}
	if (eff > rfx_c->win_peak)
		rfx_c->win_peak = eff;
	return rfx_c->win_peak;
}

/*
 * Publish this perf cluster's windowed demand as a percent of its capacity, for
 * the cross-migration carry. LITTLE is support work and never the render-thread
 * migration target, so it does not contribute. Monotonic-max within a coarse
 * decay is unnecessary: the seed only matters for a cold cluster, and stale-high
 * values are harmless because the seeded filter immediately re-tracks real
 * demand via the EMA next sample.
 */
static inline void rfx_publish_perf_demand(struct rfx_policy *p,
					   unsigned long eff, unsigned long max_cap)
{
	unsigned int pct;

	if (p->is_little || !max_cap)
		return;
	pct = (unsigned int)min_t(unsigned long, eff * 100 / max_cap, 100);
	if (pct > (unsigned int)atomic_read(&rfx_perf_demand_pct))
		atomic_set(&rfx_perf_demand_pct, pct);
	else
		/* Decay toward the latest so it tracks down after a burst ends. */
		atomic_set(&rfx_perf_demand_pct,
			   ((unsigned int)atomic_read(&rfx_perf_demand_pct) * 3 +
			    pct) / 4);
}

/*
 * Seed value (raw util units) for a cold perf-cluster filter from the published
 * cross-cluster demand high-water. Scaled to THIS cluster's capacity so a Prime
 * seed derived from Big demand (or vice versa) lands at the right magnitude.
 * Bounded to max_cap. Returns 0 if no demand has been published yet (safe: the
 * EMA then behaves exactly as the stock cold-start path).
 */
static inline unsigned long rfx_migr_seed(unsigned long max_cap)
{
	unsigned int pct = (unsigned int)atomic_read(&rfx_perf_demand_pct);

	if (!pct)
		return 0;
	return min(max_cap, (unsigned long)(max_cap * pct / 100));
}

/*
 * Headroom: request slightly more capacity than measured so we land on an OPP
 * with room to spare (avoids running pinned at 100% util, which is both a
 * latency and a load-percent problem). The pressure fallback reads raw demand
 * before this margin; normal frequency selection still gets the schedutil-style
 * 25% boost. Daily then adds a small tiered curve for UI response.
 */
static unsigned long rfx_apply_headroom(unsigned long util, unsigned long max_cap,
					bool gaming, bool little)
{
	unsigned int upct;

	if (!max_cap || util >= max_cap)
		return max_cap;

	upct = (unsigned int)(util * 100 / max_cap);
	if (upct >= 95)
		return max_cap;

	if (gaming)
		return min(util + (util >> 2), max_cap);

	util = min(util + (util >> 2), max_cap);
	if (util >= max_cap)
		return max_cap;
	upct = (unsigned int)(util * 100 / max_cap);

	if (little) {
		if (upct >= 70)
			return min(util + (util >> 4), max_cap);
		if (upct >= 45)
			return min(util + (util >> 5), max_cap);
		return util;
	}

	if (upct >= 75)
		return min(util + (util >> 4), max_cap);
	if (upct >= 50)
		return min(util + (util >> 5), max_cap);
	return min(util + (util >> 6), max_cap);
}

/* ===================================================================== */
/* Thermal step controller (final clamp)                                 */
/* ===================================================================== */

/*
 * Map temperature (milli-Celsius) to a target cap percent along a continuous
 * piecewise-linear curve: flat 100% below GREEN, -2%/C to GREEN..YELLOW, then
 * -3%/C to YELLOW..RED, then a flat floor. The pieces meet (100 at GREEN, 92 at
 * YELLOW, 80 at RED) so there is no step at the breakpoints. The vendor thermal
 * governor handles anything beyond.
 */
static int rfx_temp_to_cap(int t_mc)
{
	if (t_mc < RFX_TEMP_GREEN_MC)
		return 100;
	if (t_mc < RFX_TEMP_YELLOW_MC)			/* 100 -> 92 */
		return 100 - (t_mc - RFX_TEMP_GREEN_MC) * 2 / 1000;
	if (t_mc < RFX_TEMP_RED_MC)			/* 92 -> 80 */
		return 92 - (t_mc - RFX_TEMP_YELLOW_MC) * 3 / 1000;
	return 80;
}

/*
 * Walk the applied cap toward the published target in small, rate-limited
 * steps and clamp the requested frequency to it. Stepping down faster than up
 * gives a smooth throttle entry and a gentle recovery (no oscillation at the
 * trip point). This is the LAST clamp, so it always wins over the floors -
 * that is what bounds power/temperature under the FPS-first policy.
 */
static unsigned int rfx_thermal_clamp(struct rfx_policy *p, unsigned int freq,
				      unsigned int fmax)
{
	int target = atomic_read(&rfx_thermal_cap_pct);
	int applied = p->thermal_applied_pct ? p->thermal_applied_pct : 100;
	u64 now = ktime_get_ns();

	if ((s64)(now - p->thermal_step_ns) >= (s64)RFX_THERMAL_STEP_NS) {
		/*
		 * Step toward target, but never STEP PAST it (avoids a 1-2%
		 * dither at the throttle knee when the step size doesn't divide
		 * the remaining distance).
		 */
		if (applied > target) {
			applied -= RFX_THERMAL_STEP_DOWN_PCT;
			if (applied < target)
				applied = target;
		} else if (applied < target) {
			applied += RFX_THERMAL_STEP_UP_PCT;
			if (applied > target)
				applied = target;
		}
		applied = clamp(applied, RFX_THERMAL_MIN_CAP_PCT, 100);
		p->thermal_applied_pct = applied;
		p->thermal_step_ns = now;
	}

	if (applied < 100) {
		unsigned int cap = rfx_pct(fmax, applied);

		if (freq > cap)
			freq = cap;
	}
	return freq;
}

/* ===================================================================== */
/* Frame pacing                                                          */
/* ===================================================================== */

static bool rfx_drm_present_snapshot(u64 *seqp, u64 *presentp)
{
	int i;

	for (i = 0; i < RFX_DRM_SAMPLE_RETRIES; i++) {
		u64 seq = (u64)atomic64_read(&drm_present_seq);
		u64 present, seq2;

		if (!seq)
			return false;

		present = (u64)atomic64_read(&drm_last_present_ns);
		seq2 = (u64)atomic64_read(&drm_present_seq);
		if (likely(seq == seq2)) {
			*seqp = seq;
			*presentp = present;
			return true;
		}

		/*
		 * The feed is alive but racing this consumer. Do not let an
		 * active OOS/COS-style flip stream look stale and fall into the
		 * coarser CPU-pressure fallback; a later update can consume the
		 * widened seq delta and recover the mean frame interval.
		 */
		atomic64_set(&rfx_last_frame_feed_ns,
			     present ? present : ktime_get_ns());
		cpu_relax();
	}

	return false;
}

static inline void rfx_arm_gpu_boost(u64 now, bool hard)
{
	atomic64_set(&vorpal_gpu_boost_until_ns, now + RFX_GPU_BOOST_NS);
	if (hard)
		atomic64_set(&vorpal_gpu_hard_boost_until_ns,
			     now + RFX_GPU_HARD_BOOST_NS);
}

/*
 * Arm the cold-start / render-resume launch window: hold the Prime warmup AND a
 * GPU-floor window that spans the bursty launch phase, plus a short hard GPU pulse
 * for the very first burst. Single choke-point so every render-start trigger
 * (gaming-on, DRM present-resume, and the util-driven detector) warms CPU and GPU
 * together instead of leaving the GPU cold once the 600ms reactive pulse expires.
 * `now` MUST be a ktime_get_ns() value: every deadline it writes is consumed
 * against ktime (rfx_in_warmup, the devfreq GPU floor), not rq_clock.
 */
static inline void rfx_arm_launch_window(u64 now)
{
	atomic64_set(&rfx_gaming_warmup_end_ns, now + RFX_G_PRIME_WARMUP_NS);
	/*
	 * Dedicated cold-start GPU floor: hold the GPU at the (higher) launch floor
	 * for the whole bursty asset-streaming phase, with a short stronger hard
	 * pulse for the heaviest first spawn frames. The GPU is the slow-ramping part
	 * of a render start, so give it its own window rather than borrowing the 50%
	 * reactive soft floor - which lapsed mid-burst and let the GPU downshift right
	 * as the first heavy frames streamed in (the "loads then jitters, fps dips at
	 * spawn" symptom). The plain soft window is armed too as a graceful fallback
	 * if the launch floor is tuned to 0. All time-bounded (< the 5s CPU warmup),
	 * so no steady-state GPU heat.
	 */
	atomic64_set(&vorpal_gpu_launch_until_ns, now + RFX_GPU_LAUNCH_NS);
	atomic64_set(&vorpal_gpu_boost_until_ns, now + RFX_GPU_LAUNCH_NS);
	atomic64_set(&vorpal_gpu_hard_boost_until_ns, now + RFX_GPU_LAUNCH_HARD_NS);
}

/*
 * Device-agnostic render-start / resume edge. Called from the gaming fast path
 * for a perf cluster (Big/Prime) with its post-headroom util percent. When the
 * cluster goes busy after a quiet gap (no perf-cluster busy sample for longer than
 * RFX_RENDER_RESUME_GAP_NS) it means rendering just (re)started - e.g. entering a
 * match after a menu, or a scene load - so re-arm the launch window even when no
 * DRM present feed exists to signal it. ktime domain throughout (deadlines are
 * consumed against ktime, not rq_clock); the ktime read is skipped entirely on
 * near-idle perf-cluster samples via the early BUSY_ENTER return.
 */
static inline void rfx_render_resume_detect(unsigned int upct)
{
	u64 prev, now, warmup_end;

	if (upct < RFX_G_PRIME_BUSY_ENTER_PCT)
		return;

	now = ktime_get_ns();
	prev = (u64)atomic64_read(&rfx_last_perf_busy_ns);
	atomic64_set(&rfx_last_perf_busy_ns, now);

	/*
	 * Already warming up: do not re-arm. This bounds the detector to one launch
	 * window per real cold-start - otherwise a bursty light game whose perf
	 * cluster idles > the gap between frames would be perpetually re-warmed,
	 * wasting heat. During steady play the perf clusters go busy far more often
	 * than the gap, so last_perf_busy keeps advancing and no spurious edge fires.
	 * (Inlined deadline read to avoid a forward reference to rfx_in_warmup.)
	 */
	warmup_end = (u64)atomic64_read(&rfx_gaming_warmup_end_ns);
	if (warmup_end && now < warmup_end)
		return;

	if (!prev || (s64)(now - prev) > (s64)RFX_RENDER_RESUME_GAP_NS)
		rfx_arm_launch_window(now);
}

static inline bool rfx_perf_vendor_throttled(void)
{
	return atomic_read(&rfx_sys_capped_pct) < RFX_VENDOR_THROTTLE_PCT &&
	       atomic_read(&rfx_prime_capped_pct) < RFX_VENDOR_THROTTLE_PCT;
}

/*
 * Derive this update's frame time (microseconds), consuming each present exactly
 * once across all clusters. A userspace write to frame_time_us takes priority
 * (benchmark / override path); otherwise the DRM present feed is used: the
 * interval between consecutive delivered flips = the frame's display time.
 *
 * The feed publishes only the LATEST present timestamp, so if several presents
 * elapsed between two samples (e.g. a fast-path gap), the sequence counter lets
 * us divide the interval by the number of presents and recover the true mean
 * frame time instead of mis-charging it as one slow frame.
 */
static unsigned int rfx_frame_time_sample(void)
{
	unsigned int ft = atomic_xchg(&rfx_frame_time_us, 0);
	u64 seq, last_seq, present, prev, delta, n;

	if (ft) {
		atomic64_set(&rfx_last_frame_feed_ns, ktime_get_ns());
		return ft;			/* userspace override wins */
	}

	if (!rfx_drm_present_snapshot(&seq, &present))
		return 0;			/* no stable DRM sample yet */

	last_seq = (u64)atomic64_read(&rfx_last_seq_consumed);
	if (seq == last_seq)
		return 0;			/* no new present since last sample */

	/* Claim these presents exactly once - losers of the race bail out. */
	if ((u64)atomic64_cmpxchg(&rfx_last_seq_consumed, last_seq, seq)
	    != last_seq)
		return 0;

	prev = (u64)atomic64_read(&rfx_prev_present_ns);
	atomic64_set(&rfx_prev_present_ns, present);
	if (present)
		atomic64_set(&rfx_last_frame_feed_ns, present);

	if (!last_seq || !prev || present <= prev)
		return 0;			/* first sample: no interval yet */

	delta = present - prev;
	if (delta > RFX_FRAME_PRESENT_GAP_NS) {
		/*
		 * Rendering resumed after a pause (match load / scene transition).
		 * Re-arm the launch window (Prime warmup + a GPU floor that spans the
		 * post-load asset burst, not just the 600ms reactive pulse) so the
		 * first frames after the gap land on a warm CPU and a warm GPU, and
		 * reset the frame-time EMA so the stale pre-gap value does not bias the
		 * sustain controller.
		 */
		rfx_arm_launch_window(ktime_get_ns());
		atomic_set(&rfx_ft_ema_us, 0);
		/* Scene/level changed: drop any detected cap and re-detect cadence. */
		atomic_set(&rfx_cadence_bud_us, 0);
		atomic_set(&rfx_cadence_stable, 0);
		return 0;			/* gap itself is not a frame time */
	}

	n = seq - last_seq;			/* presents elapsed -> mean interval */
	delta = div_u64(delta, n);
	return (unsigned int)(delta / NSEC_PER_USEC);
}

/*
 * Adaptive effective budget: relax the sustain target to the game's real steady
 * cadence when a slower FPS cap is confirmed stable, so a 90/60fps title is not
 * scored as a chronic 120 jank (which would pin the boost and cook the SoC), while
 * a game merely struggling below a higher target keeps the tighter configured
 * budget and is still recovered. Returns the budget (us) to score this frame with.
 * Only entered from rfx_frame_account (one CPU per present), so the small detection
 * state is race-free in practice.
 */
static unsigned int rfx_effective_budget(unsigned int ft, int ema, unsigned int bud)
{
	static const unsigned int std_bud[] = {
		RFX_FRAME_BUDGET_US_60, RFX_FRAME_BUDGET_US_90,
		RFX_FRAME_BUDGET_US_100,
	};
	unsigned int detected, cand = 0;
	int diff, stable, i;
	bool frame_stable;

	/* User pinned frame_budget_us -> honour it verbatim, no auto-detection. */
	if (rfx_frame_budget_user_set) {
		if (atomic_read(&rfx_cadence_bud_us))
			atomic_set(&rfx_cadence_bud_us, 0);
		return bud;
	}

	/* Slowest standard cadence the smoothed frame time has settled at/under. */
	for (i = 0; i < ARRAY_SIZE(std_bud); i++) {
		if (std_bud[i] <= bud)
			continue;		/* never relax tighter than configured */
		if ((s64)ema * 100 >= (s64)std_bud[i] * RFX_CADENCE_MATCH_PCT &&
		    std_bud[i] > cand)
			cand = std_bud[i];
	}

	/* Stable = this raw frame is close to its own EMA (a real cap, not jitter). */
	diff = (int)ft - ema;
	if (diff < 0)
		diff = -diff;
	frame_stable = ema && (s64)diff * 100 <= (s64)ema * RFX_CADENCE_STABLE_PCT;

	if (cand && frame_stable) {
		stable = atomic_read(&rfx_cadence_stable) + 1;
		/* Stop counting once latched; the compare is only >=. */
		if (stable <= RFX_CADENCE_STABLE_FRAMES)
			atomic_set(&rfx_cadence_stable, stable);
		if (stable >= RFX_CADENCE_STABLE_FRAMES)
			atomic_set(&rfx_cadence_bud_us, cand);
	} else {
		/* Jittery, or cadence climbed back to the configured target: re-detect. */
		atomic_set(&rfx_cadence_stable, 0);
		if (!cand)
			atomic_set(&rfx_cadence_bud_us, 0);
	}

	detected = atomic_read(&rfx_cadence_bud_us);
	return detected > bud ? detected : bud;
}

/*
 * Called each gaming update from any cluster. If a presented frame overran 1.5x
 * the budget, arm a SHORT floor boost so the next frames recover - but only if
 * the render cluster still has headroom (the gate). The boost lifts the floor
 * (see target_freq), never forces fmax, so load stays mid and heat stays bounded.
 */
static void rfx_frame_account(void)
{
	unsigned int ft = rfx_frame_time_sample();
	unsigned int bud = atomic_read(&rfx_frame_budget_us);
	unsigned int gpu_sustain_pct = atomic_read(&rfx_g_gpu_sustain_pct);
	u64 now = 0;
	int ema;
	bool jank, near_miss, sag, gpu_pressure, want_boost;

	if (!ft || !bud)
		return;

	atomic_inc(&rfx_frames_seen);

	/*
	 * Smooth the frame time so a single noisy present does not drive control;
	 * the proactive sustain decision runs off this EMA, not the raw sample.
	 */
	ema = atomic_read(&rfx_ft_ema_us);
	ema = ema ? ema + (((int)ft - ema) >> RFX_FT_EMA_SHIFT) : (int)ft;
	atomic_set(&rfx_ft_ema_us, ema);

	/*
	 * Relax the target to the game's real steady cadence (90/60) when confirmed,
	 * so a capped title is not scored against 120 every frame. Struggling-below-a
	 * -higher-target games keep the tighter configured budget (see helper).
	 */
	bud = rfx_effective_budget(ft, ema, bud);

	/* Reactive: a real jank (raw frame already overran 1.5x budget). */
	jank = ft > bud + (bud >> 1);
	if (jank)
		atomic_inc(&rfx_janks_seen);
	near_miss = (s64)ft * 100 > (s64)bud * RFX_FRAME_NEAR_MISS_PCT;

	/*
	 * Proactive sustain: the SMOOTHED frame time has crept into the sag band
	 * (FPS sliding below ~115 but not yet a jank). Catch it BEFORE it becomes
	 * a visible drop so min-FPS holds near the 120 target. Off entirely when
	 * frames are on-cadence (ema ~ budget < threshold) -> easy scenes stay in
	 * race-to-idle and average CPU load is unchanged (not greedy).
	 */
	sag = (s64)ema * 100 > (s64)bud * RFX_FRAME_SUSTAIN_PCT;
	gpu_pressure = jank || near_miss ||
		(gpu_sustain_pct &&
		 (s64)ema * 100 > (s64)bud * gpu_sustain_pct);

	want_boost = jank || near_miss || sag;
	if (gpu_pressure) {
		now = ktime_get_ns();
		/*
		 * Signal the devfreq GPU floor: frames are under pressure, so keep
		 * the GPU ready for the next burst. Ungated by the CPU headroom gate
		 * below - the GPU is a separate resource. ktime domain (the devfreq
		 * consumer reads ktime_get_ns()).
		 */
		rfx_arm_gpu_boost(now, jank || near_miss);
	}
	if (want_boost) {
		unsigned int sys = atomic_read(&rfx_sys_perf_pct);
		unsigned int prime = atomic_read(&rfx_prime_perf_pct);
		unsigned int lo = min(sys, prime);
		unsigned int hi = max(sys, prime);

		if (!now)
			now = ktime_get_ns();
		/*
		 * Anti-runaway gate over BOTH perf clusters: arm only if one cluster
		 * still has recovery headroom. The busiest cluster may sit just below
		 * fmax during crowd/asset bursts, so do not block the floor until it
		 * is effectively pinned. A counted jank may still pre-warm the colder
		 * perf cluster even if the other one is pinned; if both are already
		 * pinned, lo fails the gate and the CPU boost stays off.
		 *
		 * A near_miss (a single raw frame already >=1.3x budget) is a firmer
		 * signal than a mere smoothed sag, so - like a jank - it may pre-warm
		 * the colder cluster even when the other is momentarily pinned; the
		 * lo-headroom gate still prevents a runaway when BOTH are pinned. This
		 * stops the proactive recovery from silently switching off the instant
		 * one perf cluster brushes fmax, which let min-FPS keep sliding.
		 */
		if (!rfx_perf_vendor_throttled() &&
		    lo < RFX_FRAME_BOOST_GATE_PCT &&
		    (hi < RFX_FRAME_BOOST_BUSY_PCT || jank || near_miss))
			atomic64_set(&rfx_frame_boost_end_ns,
				     now + RFX_FRAME_BOOST_NS);
	}

	/*
	 * Severe sustained burst (e.g. a match landing): the smoothed frame rate has
	 * collapsed well below target. The CPU floors already have the perf clusters
	 * near fmax (caps are 100%), so the residual bottleneck is the GPU ramping the
	 * freshly-streamed scene. Give it the stronger, longer launch-class floor - but
	 * GPU-ONLY (do NOT touch rfx_gaming_warmup_end_ns), so a long combat sequence
	 * can never hold the CPU warmup on and cook the die (the reverted-floor failure
	 * mode). Rate-limited so it refreshes across a multi-second burst without
	 * hammering the atomics; time-bounded and a no-op whenever real GPU demand
	 * already exceeds the floor, so it adds no steady-state heat.
	 */
	if ((s64)ema * 100 > (s64)bud * RFX_GPU_BURST_PCT) {
		u64 nb = (u64)atomic64_read(&rfx_gpu_burst_next_ns);

		if (!now)
			now = ktime_get_ns();
		if (!nb || (s64)(now - nb) >= 0) {
			atomic64_set(&vorpal_gpu_launch_until_ns, now + RFX_GPU_LAUNCH_NS);
			atomic64_set(&vorpal_gpu_boost_until_ns, now + RFX_GPU_LAUNCH_NS);
			atomic64_set(&vorpal_gpu_hard_boost_until_ns,
				     now + RFX_GPU_HARD_BOOST_NS);
			atomic64_set(&rfx_gpu_burst_next_ns,
				     now + RFX_GPU_BURST_REFRESH_NS);
		}
	}
}

static inline bool rfx_frame_boost_active(void)
{
	u64 end = (u64)atomic64_read(&rfx_frame_boost_end_ns);

	return end && ktime_get_ns() < end;
}

static inline bool rfx_frame_feed_stale(void)
{
	u64 last = (u64)atomic64_read(&rfx_last_frame_feed_ns);

	return !last ||
	       (s64)(ktime_get_ns() - last) > (s64)RFX_FRAME_FEED_STALE_NS;
}

/*
 * Some ROM display stacks do not deliver a reliable DRM flip-complete stream to
 * the generic vblank path. When that feed is stale, split the fallback into a
 * soft GPU readiness pulse at moderate perf-cluster demand and a stricter CPU
 * recovery pulse only at high raw demand. This keeps render from restarting on
 * cold OPPs without turning stale frame-feed paths into continuous hard boost.
 */
static inline void rfx_cpu_pressure_boost(unsigned int upct, bool little)
{
	unsigned int pct = atomic_read(&rfx_g_stale_pressure_pct);
	u64 now, next;

	if (little || !rfx_frame_feed_stale())
		return;

	now = ktime_get_ns();

	if (upct >= RFX_STALE_GPU_PRESSURE_PCT) {
		next = (u64)atomic64_read(&rfx_stale_gpu_next_ns);
		if (!next || (s64)(now - next) >= 0) {
			rfx_arm_gpu_boost(now, false);
			atomic64_set(&rfx_stale_gpu_next_ns,
				     now + RFX_STALE_GPU_REFRESH_NS);
		}
	}

	if (!pct || upct < pct || rfx_perf_vendor_throttled())
		return;

	next = (u64)atomic64_read(&rfx_cpu_pressure_next_ns);
	if (next && (s64)(now - next) < 0)
		return;

	/*
	 * Sustained high perf-cluster demand on a feed-less ROM: this is the only
	 * signal we get that the render path is straining, so layer a short hard
	 * GPU pulse on top of the soft floor - the GPU is the likely bottleneck a
	 * high-CPU stale path cannot otherwise reach. Still bounded: the hard pulse
	 * is RFX_GPU_HARD_BOOST_NS and this branch is rate-limited by
	 * RFX_CPU_PRESSURE_REFRESH_NS, so it never becomes a continuous hard boost.
	 */
	rfx_arm_gpu_boost(now, true);
	atomic64_set(&rfx_frame_boost_end_ns, now + RFX_FRAME_BOOST_NS);
	atomic64_set(&rfx_cpu_pressure_next_ns,
		     now + RFX_CPU_PRESSURE_REFRESH_NS);
}

/*
 * Cold-start launch window: Prime forced busy + Little/Big held at launch
 * readiness. The deadline is armed from ktime_get_ns() (gaming-on) and from the
 * DRM present timestamp (present-resume re-arm) - both CLOCK_MONOTONIC - so it
 * MUST be checked against ktime_get_ns() too, not the scheduler's rq_clock
 * `time` (a different clock domain on arm64, which would make the window read
 * as already-expired or far in the future).
 */
static inline bool rfx_in_warmup(void)
{
	u64 end = (u64)atomic64_read(&rfx_gaming_warmup_end_ns);

	return end && ktime_get_ns() < end;
}

/* ===================================================================== */
/* Frequency decision                                                    */
/* ===================================================================== */

/*
 * Pure-ish frequency selection from a (smoothed) util value. Order:
 *   1. headroom -> base freq from util/capacity
 *   2. profile shaping (gaming band lock OR daily caps/floors)
 *   3. thermal step clamp (final ceiling)
 *   4. resolve to a real OPP (cached to skip redundant table walks)
 */
static unsigned int rfx_target_freq(struct rfx_policy *p, unsigned long util,
				    unsigned long max_cap, u64 time, bool gaming)
{
	struct cpufreq_policy *pol = p->policy;
	unsigned int fmax = pol->cpuinfo.max_freq;
	unsigned int fmin = pol->cpuinfo.min_freq;
	bool little = rfx_cap_is_little(max_cap);
	bool prime = rfx_cap_is_prime(max_cap);
	unsigned int freq, raw_upct, upct;

	if (!fmax)
		return pol->cur;

	raw_upct = max_cap ? (unsigned int)(util * 100 / max_cap) : 0;
	util = rfx_apply_headroom(util, max_cap, gaming, little);
	upct = max_cap ? (unsigned int)(util * 100 / max_cap) : 0;

	if (unlikely(!max_cap))
		return pol->cur;
	freq = (unsigned int)((u64)fmax * util / max_cap);
	freq = clamp(freq, fmin, fmax);

	if (gaming) {
		bool fboost = rfx_frame_boost_active();

		rfx_cpu_pressure_boost(raw_upct, little);
		/*
		 * Util-driven render-start detector (device-agnostic). A perf
		 * cluster going busy after a quiet gap = rendering just (re)started;
		 * re-arm the launch window so the render path warms even on ROMs that
		 * never deliver a DRM present feed. Keyed off upct (post-headroom) so
		 * it tracks real demand, not the floors this function is about to add.
		 */
		if (!little)
			rfx_render_resume_detect(upct);
		fboost = rfx_frame_boost_active();

		/*
		 * WALT-like proactive ramp. A sharp one-sample rise in perf-cluster
		 * demand is a spike PELT has only just begun to track; pre-arm the
		 * frame boost (lifts every cluster's floor toward fmax) and the GPU
		 * window NOW so the render path is already high when the spike's first
		 * frames land, instead of ramping a beat late and dropping one. Gated
		 * on a non-zero prev so the cold-start (handled by the launch window)
		 * does not double-fire, on !little so support work does not trigger it,
		 * and bounded by RFX_FRAME_BOOST_NS so steady play adds no heat. Also
		 * makes THIS sample use the boosted floors by setting fboost.
		 */
		{
			unsigned int ramp = atomic_read(&rfx_g_ramp_delta_pct);

			if (ramp && !little && p->prev_upct &&
			    upct > p->prev_upct + ramp) {
				u64 nk = ktime_get_ns();

				atomic64_set(&rfx_frame_boost_end_ns,
					     nk + RFX_FRAME_BOOST_NS);
				rfx_arm_gpu_boost(nk, true);
				fboost = true;
			}
		}
		p->prev_upct = upct;

		if (prime) {
			/*
			 * Adaptive floor + busy-hold hysteresis. Lock Prime high
			 * while busy and KEEP it locked for RFX_G_PRIME_HOLD_NS
			 * after the last busy sample, so a bursty render thread does
			 * not let the floor toggle down into the inter-burst gaps.
			 *
			 * A frame-pressure boost also pre-warms Prime briefly. The
			 * next burst may land there after the sample that detected
			 * the sag; waiting for Prime to become busy first lets it
			 * start from a cold OPP and turns the recovery into another
			 * late frame. The boost is bounded by RFX_FRAME_BOOST_NS and
			 * the frame floor, so it is readiness, not an fmax pin.
			 */
			bool busy;
			unsigned int fl, cap;

			if (upct >= RFX_G_PRIME_BUSY_ENTER_PCT)
				p->prime_busy_hold_ns = time + RFX_G_PRIME_HOLD_NS;
			busy = (upct >= RFX_G_PRIME_BUSY_ENTER_PCT) ||
			       (p->prime_busy_hold_ns &&
				time < p->prime_busy_hold_ns) ||
			       rfx_in_warmup() || fboost;

			fl = busy ? rfx_pct(fmax, atomic_read(&rfx_g_prime_floor_pct)) :
				    rfx_pct(fmax, atomic_read(&rfx_g_prime_idle_floor_pct));
			cap = rfx_pct(fmax, RFX_G_PRIME_CAP_PCT);

			if (fboost && busy)
				fl = max(fl, rfx_pct(fmax, RFX_G_PRIME_FRAME_PCT));
			if (freq < fl)
				freq = fl;
			if (freq > cap)
				freq = cap;
		} else if (!little) {		/* Big: carries most load */
			unsigned int fl = rfx_pct(fmax, atomic_read(&rfx_g_big_floor_pct));
			unsigned int cap = rfx_pct(fmax, RFX_G_BIG_CAP_PCT);

			if (fboost)
				fl = max(fl, rfx_pct(fmax, RFX_G_BIG_FRAME_PCT));
			/*
			 * Cold-start dip fix (the one flaw of the cool v2.1 base):
			 * the render thread lives mostly on Big, so at match spawn a
			 * cold Big (sitting at its 85% floor, ramping via util) lands
			 * the first heavy frames late -> the start drop to ~70fps.
			 * During the warmup window ONLY, pre-lift Big to its frame
			 * floor so those first frames render on a warm cluster. This is
			 * strictly time-bounded to the cold-start window (re-armed on
			 * present-resume), so it adds NO steady-state heat - unlike a
			 * raised static floor, which regressed thermals + jank.
			 */
			if (rfx_in_warmup())
				fl = max(fl, rfx_pct(fmax, RFX_G_BIG_FRAME_PCT));
			if (freq < fl)
				freq = fl;
			if (freq > cap)
				freq = cap;
		} else {			/* Little: dynamic, soft floor */
			unsigned int cap = rfx_pct(fmax, RFX_G_LITTLE_CAP_PCT);
			unsigned int fl = 0;
			bool busy;

			/* Busy-hold: bridge micro-idle so we don't crash to fmin. */
			if (upct > RFX_G_LITTLE_FLOOR_ENTER_PCT)
				p->little_busy_hold_ns = time + RFX_G_LITTLE_HOLD_NS;
			busy = (upct > RFX_G_LITTLE_FLOOR_ENTER_PCT) ||
			       (p->little_busy_hold_ns &&
				time < p->little_busy_hold_ns);

			if (busy)
				fl = rfx_pct(fmax, atomic_read(&rfx_g_little_floor_pct));
			/* Same rule as Prime: don't lift an idle Little (waste heat). */
			if (fboost && busy)
				fl = max(fl, rfx_pct(fmax, RFX_G_LITTLE_FRAME_PCT));
			/*
			 * Cross-cluster sync: a saturated Big/Prime means the frame's
			 * critical path is on a perf core; lift the busy LITTLE support
			 * threads so they keep pace (heat-cheap: coolest cluster, only
			 * when busy AND a perf cluster is actually saturated).
			 */
			if (busy &&
			    (atomic_read(&rfx_sys_perf_pct) >= RFX_G_PERF_SAT_PCT ||
			     atomic_read(&rfx_prime_perf_pct) >= RFX_G_PERF_SAT_PCT))
				fl = max(fl, rfx_pct(fmax, RFX_G_LITTLE_SYNC_FLOOR_PCT));
			/*
			 * Cold-start launch readiness: during the warmup window hold
			 * Little at the sync floor even if not yet busy, so the spawn
			 * burst (physics / audio / asset-decode support threads) is
			 * served from the first frame instead of ramping up late.
			 * Time-bounded to the launch window -> no steady-state heat.
			 */
			if (rfx_in_warmup())
				fl = max(fl, rfx_pct(fmax, RFX_G_LITTLE_SYNC_FLOOR_PCT));
			/*
			 * Closed-loop thermal redistribution: if the vendor has
			 * throttled the prime/render core (SoC is heat-limited), shed
			 * this LITTLE cluster to a lower cap to free thermal budget so
			 * prime can recover. Acts only while prime is actually
			 * throttled; bounded/tunable (0 or 100 = off). The cap is
			 * applied last, so it wins over the floors above.
			 */
			{
				int shed = atomic_read(&rfx_g_shed_little_pct);

				if (shed > 0 && shed < 100 &&
				    atomic_read(&rfx_prime_capped_pct) <
				    RFX_PRIME_THROTTLE_DETECT_PCT)
					cap = min(cap, rfx_pct(fmax, shed));
			}
			if (freq < fl)
				freq = fl;
			if (freq > cap)
				freq = cap;
		}
	} else {
		bool ui_active;

		/*
		 * Detect a render burst: a sharp rise in smoothed util re-arms
		 * the UI floor. Catches caption draws / open-close animations /
		 * fling-scrolls that touch detection alone would miss.
		 */
		if (upct > p->prev_upct &&
		    upct - p->prev_upct >= RFX_D_RAMP_DELTA_PCT)
			p->ui_boost_end_ns = time + RFX_D_UI_BOOST_NS;
		p->prev_upct = upct;

		ui_active = rfx_input_active() ||
			    (p->ui_boost_end_ns && time < p->ui_boost_end_ns);

		if (little) {
			unsigned int cap = ui_active ?
				rfx_pct(fmax, RFX_D_LITTLE_BOOST_CAP_PCT) :
				rfx_pct(fmax, RFX_D_LITTLE_CAP_PCT);

			if (freq > cap)
				freq = cap;
			if (ui_active) {
				unsigned int fl = rfx_pct(fmax, RFX_D_LITTLE_UI_FLOOR_PCT);

				if (freq < fl)
					freq = fl;
			}
		} else if (!prime && ui_active) {	/* Big UI floor */
			unsigned int fl = rfx_pct(fmax, RFX_D_BIG_UI_FLOOR_PCT);

			if (freq < fl)
				freq = fl;
		}
	}

	freq = rfx_thermal_clamp(p, freq, fmax);
	freq = clamp(freq, fmin, fmax);

	/*
	 * Publish each perf cluster's headroom for the frame-boost gate. The
	 * render thread can live on Big OR Prime, so the gate needs both: it
	 * arms when the least-loaded perf cluster still has room (see
	 * rfx_frame_account). Little is support work, not a gate input.
	 */
	if (gaming) {
		unsigned int pct = (unsigned int)((u64)freq * 100 / fmax);

		if (prime) {
			atomic_set(&rfx_prime_perf_pct, pct);
			/*
			 * Publish how far the vendor has capped prime's policy->max
			 * below the hardware max - the LITTLE thermal shed keys off
			 * this to detect (and react to) a prime thermal throttle.
			 */
			atomic_set(&rfx_prime_capped_pct,
				   (unsigned int)((u64)pol->max * 100 / fmax));
		} else if (!little) {
			atomic_set(&rfx_sys_perf_pct, pct);
			atomic_set(&rfx_sys_capped_pct,
				   (unsigned int)((u64)pol->max * 100 / fmax));
		}
	}

	if (freq == p->cached_raw_freq && !p->need_freq_update)
		return p->next_freq;
	p->cached_raw_freq = freq;
	return cpufreq_driver_resolve_freq(pol, freq);
}

/* ===================================================================== */
/* IO-wait boost                                                         */
/* ===================================================================== */

static bool rfx_iowait_reset(struct rfx_cpu *rfx_c, u64 time, bool set)
{
	s64 delta_ns = time - rfx_c->last_update;

	if (delta_ns <= TICK_NSEC)
		return false;

	rfx_c->iowait_boost = set ? IOWAIT_BOOST_MIN : 0;
	rfx_c->iowait_boost_pending = set;
	return true;
}

static void rfx_iowait_boost(struct rfx_cpu *rfx_c, u64 time, unsigned int flags)
{
	bool set = flags & SCHED_CPUFREQ_IOWAIT;
	unsigned long max_cap;
	unsigned int cap;

	/* Reset the boost if the CPU appears to have been idle long enough. */
	if (rfx_c->iowait_boost && rfx_iowait_reset(rfx_c, time, set))
		return;

	/* Only tasks waking up after IO get boosted. */
	if (!set)
		return;

	/* Double the boost at most once per IO-wakeup request. */
	if (rfx_c->iowait_boost_pending)
		return;
	rfx_c->iowait_boost_pending = true;

	max_cap = arch_scale_cpu_capacity(rfx_c->cpu);
	if (rfx_gaming_enabled()) {
		unsigned int pct = atomic_read(&rfx_g_iowait_boost_pct);
		unsigned int cluster_cap_pct;

		/*
		 * Gaming keeps iowait readiness active but bounded per cluster.
		 * LITTLE support wakes should finish promptly without dragging the
		 * coolest cluster to fmax; Big/Prime can climb into the render band
		 * but still leave headroom for real measured demand and the final
		 * thermal clamp.
		 */
		if (rfx_cap_is_little(max_cap))
			cluster_cap_pct = RFX_G_IOWAIT_LITTLE_CAP_PCT;
		else if (rfx_cap_is_prime(max_cap))
			cluster_cap_pct = RFX_G_IOWAIT_PRIME_CAP_PCT;
		else
			cluster_cap_pct = RFX_G_IOWAIT_BIG_CAP_PCT;

		pct = min(pct, cluster_cap_pct);
		cap = SCHED_CAPACITY_SCALE * pct / 100;
		cap = clamp(cap, IOWAIT_BOOST_MIN, SCHED_CAPACITY_SCALE);
	} else {
		/*
		 * Daily remains conservative: LITTLE support work is held low and
		 * perf clusters only ramp to a useful mid/high OPP.
		 */
		cap = rfx_cap_is_little(max_cap) ? (SCHED_CAPACITY_SCALE / 6) :
				(SCHED_CAPACITY_SCALE * 3 / 4);
	}

	/* Double the existing boost, else start at the minimum. */
	if (rfx_c->iowait_boost)
		rfx_c->iowait_boost = min_t(unsigned int,
					    rfx_c->iowait_boost << 1, cap);
	else
		rfx_c->iowait_boost = IOWAIT_BOOST_MIN;
}

static unsigned long rfx_iowait_apply(struct rfx_cpu *rfx_c, u64 time,
				      unsigned long max_cap)
{
	if (!rfx_c->iowait_boost)
		return 0;
	if (rfx_iowait_reset(rfx_c, time, false))
		return 0;
	if (!rfx_c->iowait_boost_pending) {
		rfx_c->iowait_boost >>= 1;
		if (rfx_c->iowait_boost < IOWAIT_BOOST_MIN) {
			rfx_c->iowait_boost = 0;
			return 0;
		}
	}
	rfx_c->iowait_boost_pending = false;
	return rfx_c->iowait_boost * max_cap >> SCHED_CAPACITY_SHIFT;
}

static void rfx_get_util(struct rfx_cpu *rfx_c, unsigned long boost, bool gaming)
{
	rfx_get_util_gki510(rfx_c->cpu, boost, gaming, &rfx_c->util, &rfx_c->bwmin);
}

static inline void rfx_ignore_dl_rate_limit(struct rfx_cpu *rfx_c)
{
	if (rfx_dl_bw_exceeded_gki510(rfx_c->cpu, rfx_c->bwmin))
		rfx_c->rfx_policy->need_freq_update = true;
}

/* ===================================================================== */
/* Rate limiting                                                         */
/* ===================================================================== */

/* Set the active down-rate-limit for this update (long while gaming). */
static inline void rfx_set_down_delay(struct rfx_policy *p, bool gaming)
{
	if (gaming) {
		/*
		 * Hybrid: LITTLE sheds fast (race-to-idle); render clusters
		 * (Big/Prime) decay steadily. Both fixed builtin.
		 */
		int us = p->is_little ? atomic_read(&rfx_g_little_down_us) :
					atomic_read(&rfx_g_down_us);

		p->down_rate_delay_ns = (s64)us * NSEC_PER_USEC;
	} else {
		p->down_rate_delay_ns =
			(s64)p->tunables->down_rate_limit_us * NSEC_PER_USEC;
	}
}

/* up-rate-limit: instant up while gaming, tunable otherwise. */
static inline void rfx_pol_up_delay(struct rfx_policy *p, bool gaming)
{
	if (gaming)
		p->up_rate_delay_ns = 0;
	else
		p->up_rate_delay_ns =
			(s64)p->tunables->up_rate_limit_us * NSEC_PER_USEC;
}

/* Evaluation gate: cheap throttle on how often we recompute at all. */
static bool rfx_should_update_freq(struct rfx_policy *p, u64 time)
{
	s64 delta, gate;

	if (!p || !p->policy)
		return false;
	if (!cpufreq_this_cpu_can_update(p->policy))
		return false;

	if (unlikely(READ_ONCE(p->limits_changed))) {
		WRITE_ONCE(p->limits_changed, false);
		p->need_freq_update = true;
		smp_mb();
		return true;
	}
	if (p->need_freq_update)
		return true;

	/*
	 * Gaming lifts the evaluation throttle to RFX_GAMING_EVAL_US so a
	 * mid-frame spike is re-evaluated within ~a quarter-ms instead of
	 * waiting out the full daily rate_limit_us. Commit-stage rate limits
	 * (instant-up / slow-decay) still shape the transition itself.
	 */
	gate = p->freq_update_delay_ns;
	if (rfx_gaming_enabled() &&
	    gate > (s64)RFX_GAMING_EVAL_US * NSEC_PER_USEC)
		gate = (s64)RFX_GAMING_EVAL_US * NSEC_PER_USEC;

	delta = (s64)(time - max(p->last_upfreq_time, p->last_downfreq_time));
	return delta >= gate;
}

/* Commit next_freq subject to directional up/down rate limits. */
static bool rfx_commit_freq(struct rfx_policy *p, u64 time, unsigned int next_freq)
{
	s64 delta;

	if (p->need_freq_update) {
		p->need_freq_update = false;
		if (p->next_freq == next_freq)
			return false;
	} else if (p->next_freq == next_freq) {
		return false;
	}

	if (next_freq < p->next_freq) {
		delta = (s64)(time - p->last_downfreq_time);
		if (p->down_rate_delay_ns > 0 && delta < p->down_rate_delay_ns) {
			/*
			 * Down-step deferred by the rate limiter. We must NOT
			 * leave the OPP cache pointing at this (lower) raw freq
			 * while next_freq still holds the old (higher) one: the
			 * cache short-circuit in rfx_target_freq would then keep
			 * returning the stale-high next_freq every update and the
			 * frequency would stay pinned high far longer than the
			 * intended rate-limit window (stuck-high -> wasted heat ->
			 * vendor throttle). Invalidate the cache so the next
			 * evaluation re-derives the true target and commits the
			 * down-step as soon as the window elapses.
			 */
			p->cached_raw_freq = 0;
			return false;
		}
		p->last_downfreq_time = time;
	} else {
		delta = (s64)(time - p->last_upfreq_time);
		if (p->up_rate_delay_ns > 0 && delta < p->up_rate_delay_ns) {
			p->cached_raw_freq = 0;
			return false;
		}
		p->last_upfreq_time = time;
	}

	p->next_freq = next_freq;
	return true;
}

/* ===================================================================== */
/* Update hooks                                                          */
/* ===================================================================== */

static void rfx_deferred_update(struct rfx_policy *p)
{
	if (!p->work_in_progress) {
		p->work_in_progress = true;
		irq_work_queue(&p->irq_work);
	}
}

static void rfx_update_single_freq(struct update_util_data *hook, u64 time,
				   unsigned int flags)
{
	struct rfx_cpu *rfx_c = container_of(hook, struct rfx_cpu, update_util);
	struct rfx_policy *p = rfx_c->rfx_policy;
	bool gaming = rfx_gaming_enabled();
	unsigned long max_cap, boost, eff;
	unsigned int next_f;

	max_cap = arch_scale_cpu_capacity(rfx_c->cpu);

	rfx_iowait_boost(rfx_c, time, flags);
	rfx_c->last_update = time;
	rfx_ignore_dl_rate_limit(rfx_c);

	if (!rfx_should_update_freq(p, time))
		return;

	boost = rfx_iowait_apply(rfx_c, time, max_cap);
	rfx_get_util(rfx_c, boost, gaming);
	eff = max(rfx_c->util, boost);

	if (gaming) {
		/*
		 * WALT-window max-capture + cross-migration carry. Seed a cold
		 * filter (render thread just migrated onto this perf core) from the
		 * recent perf-cluster demand high-water so the first frames after
		 * the hop land on a warm OPP, then feed the EMA the window peak so
		 * the burst is tracked at full magnitude without lag. Both bounded
		 * by measured demand + window decay -> no static-floor heat.
		 */
		eff = rfx_walt_window(rfx_c, eff, time);
		if (!rfx_c->filt_util && !p->is_little)
			rfx_c->filt_util = rfx_migr_seed(max_cap);
		rfx_publish_perf_demand(p, eff, max_cap);
	}

	rfx_c->filt_util = rfx_ema(rfx_c->filt_util, eff, gaming, p->is_little);

	if (gaming)
		rfx_frame_account();

	rfx_set_down_delay(p, gaming);
	rfx_pol_up_delay(p, gaming);

	next_f = rfx_target_freq(p, rfx_c->filt_util, max_cap, time, gaming);

	if (!rfx_commit_freq(p, time, next_f))
		return;

	if (p->policy->fast_switch_enabled) {
		cpufreq_driver_fast_switch(p->policy, p->next_freq);
	} else {
		raw_spin_lock(&p->update_lock);
		rfx_deferred_update(p);
		raw_spin_unlock(&p->update_lock);
	}
}

static unsigned int rfx_next_freq_shared(struct rfx_cpu *rfx_c, u64 time,
					 bool gaming)
{
	struct rfx_policy *p = rfx_c->rfx_policy;
	struct cpufreq_policy *policy = p->policy;
	unsigned long max_cap = arch_scale_cpu_capacity(rfx_c->cpu);
	unsigned long max_util = 0;
	unsigned int j;

	for_each_cpu(j, policy->cpus) {
		struct rfx_cpu *jc = per_cpu_ptr(&rfx_cpu, j);
		unsigned long jb, je;

		jb = rfx_iowait_apply(jc, time, max_cap);
		rfx_get_util(jc, jb, gaming);
		je = max(jc->util, jb);

		if (gaming) {
			je = rfx_walt_window(jc, je, time);
			if (!jc->filt_util && !p->is_little)
				jc->filt_util = rfx_migr_seed(max_cap);
			rfx_publish_perf_demand(p, je, max_cap);
		}

		jc->filt_util = rfx_ema(jc->filt_util, je, gaming, p->is_little);
		if (jc->filt_util > max_util)
			max_util = jc->filt_util;
	}

	if (gaming)
		rfx_frame_account();

	rfx_set_down_delay(p, gaming);
	rfx_pol_up_delay(p, gaming);

	return rfx_target_freq(p, max_util, max_cap, time, gaming);
}

static void rfx_update_shared(struct update_util_data *hook, u64 time,
			      unsigned int flags)
{
	struct rfx_cpu *rfx_c = container_of(hook, struct rfx_cpu, update_util);
	struct rfx_policy *p = rfx_c->rfx_policy;
	bool gaming = rfx_gaming_enabled();
	unsigned int next_f;

	raw_spin_lock(&p->update_lock);

	rfx_iowait_boost(rfx_c, time, flags);
	rfx_c->last_update = time;
	rfx_ignore_dl_rate_limit(rfx_c);

	if (rfx_should_update_freq(p, time)) {
		next_f = rfx_next_freq_shared(rfx_c, time, gaming);
		if (rfx_commit_freq(p, time, next_f)) {
			if (p->policy->fast_switch_enabled)
				cpufreq_driver_fast_switch(p->policy, p->next_freq);
			else
				rfx_deferred_update(p);
		}
	}

	raw_spin_unlock(&p->update_lock);
}

static void rfx_work(struct kthread_work *work)
{
	struct rfx_policy *p = container_of(work, struct rfx_policy, work);
	unsigned int freq;
	unsigned long flags;

	raw_spin_lock_irqsave(&p->update_lock, flags);
	freq = p->next_freq;
	p->work_in_progress = false;
	raw_spin_unlock_irqrestore(&p->update_lock, flags);

	mutex_lock(&p->work_lock);
	cpufreq_driver_target(p->policy, freq, CPUFREQ_RELATION_L);
	mutex_unlock(&p->work_lock);
}

static void rfx_irq_work(struct irq_work *irq_work)
{
	struct rfx_policy *p = container_of(irq_work, struct rfx_policy, irq_work);

	kthread_queue_work(&p->worker, &p->work);
}

/* ===================================================================== */
/* Thermal poller (slow path, may sleep -> never in the util hook)       */
/* ===================================================================== */

#ifdef CONFIG_THERMAL
static struct thermal_zone_device *rfx_tz;
static char rfx_tz_name[THERMAL_NAME_LENGTH];
static int rfx_tz_tries;		/* bound the auto-bind probing */
#define RFX_TZ_MAX_TRIES	60

/*
 * Curated SKIN / board sensor names for gaming auto-bind. Never list a
 * CPU-junction zone here - those read far hotter in normal play and would
 * throttle the skin curve mid-session. The bind is fully automatic: it engages
 * on gaming_mode=1 (rfx_thermal_fn only probes while gaming) and is released on
 * gaming_mode=0, so daily runs with no skin cap. No sysfs knob is involved.
 */
static const char * const rfx_skin_candidates[] = {
	/* Qualcomm / generic skin + board sensors. */
	"skin-therm", "quiet-therm", "skin-msm-therm",
	"xo-therm", "sys-therm", "board-therm",
	/*
	 * MediaTek board / skin NTC thermistors - physical board sensors, not
	 * the cpu/gpu/soc junction zones. ap_ntc (NTC beside the AP) is MTK's
	 * standard skin sensor; backlight_therm (display NTC) is a close fallback.
	 */
	"ap_ntc", "backlight_therm", "mtkcsbts",
};

static void rfx_tz_autobind(void)
{
	int i;

	if (READ_ONCE(rfx_tz) || rfx_tz_tries >= RFX_TZ_MAX_TRIES)
		return;
	rfx_tz_tries++;

	for (i = 0; i < ARRAY_SIZE(rfx_skin_candidates); i++) {
		struct thermal_zone_device *tz =
			thermal_zone_get_zone_by_name(rfx_skin_candidates[i]);

		if (!IS_ERR(tz)) {
			strscpy(rfx_tz_name, rfx_skin_candidates[i],
				sizeof(rfx_tz_name));
			WRITE_ONCE(rfx_tz, tz);
			pr_info("vorpal: auto-bound skin thermal zone '%s'\n",
				rfx_tz_name);
			return;
		}
	}
}

/* Drop the auto-bound zone on gaming-off so daily runs with no skin cap. */
static void rfx_tz_release_auto(void)
{
	WRITE_ONCE(rfx_tz, NULL);
	rfx_tz_name[0] = '\0';
	rfx_tz_tries = 0;
}
#endif
static struct delayed_work rfx_thermal_work;
static u64 rfx_jank_window_start;
static int rfx_temp_smoothed = -1;	/* EMA of the sensor, -1 = unset */

static void rfx_thermal_fn(struct work_struct *w)
{
	int t_mc = 0;
	bool have = false;
	unsigned int delay_ms;
	int frames, janks;
	u64 now = ktime_get_ns();

#ifdef CONFIG_THERMAL
	struct thermal_zone_device *tz;

	/* Lazily bind a skin zone while gaming (rides out late driver reg). */
	if (rfx_gaming_enabled())
		rfx_tz_autobind();

	tz = READ_ONCE(rfx_tz);		/* snapshot: gaming-off may clear it */
	if (tz && !thermal_zone_get_temp(tz, &t_mc))
		have = true;
#endif
	if (!have) {
		t_mc = atomic_read(&rfx_temp_mc);
		if (t_mc > 0)
			have = true;
	}

	/*
	 * Smooth the raw sensor with an EMA (alpha 0.25) before mapping to a cap.
	 * A bare sensor dithers ~+/-0.5C; when it sits near a breakpoint that
	 * dither would flip the published cap (e.g. 100<->98) every poll, and the
	 * fast-path step controller would chase it up and down -> a sawtooth on
	 * the clamped freq = visible jitter right at the throttle knee. Filtering
	 * the temperature publishes a steady cap, so the throttle stays smooth.
	 */
	if (have) {
		if (rfx_temp_smoothed < 0)
			rfx_temp_smoothed = t_mc;
		else
			rfx_temp_smoothed += (t_mc - rfx_temp_smoothed) >> 2;
		atomic_set(&rfx_thermal_cap_pct, rfx_temp_to_cap(rfx_temp_smoothed));
	} else {
		rfx_temp_smoothed = -1;
		atomic_set(&rfx_thermal_cap_pct, 100);
	}

	/* Jank window: publish jank percent roughly every RFX_JANK_WINDOW_NS. */
	if (!rfx_jank_window_start)
		rfx_jank_window_start = now;
	if (now - rfx_jank_window_start >= RFX_JANK_WINDOW_NS) {
		frames = atomic_xchg(&rfx_frames_seen, 0);
		janks = atomic_xchg(&rfx_janks_seen, 0);
		atomic_set(&rfx_jank_pct, frames ? janks * 100 / frames : 0);
		rfx_jank_window_start = now;
	}

	delay_ms = rfx_gaming_enabled() ? RFX_THERMAL_POLL_GAMING_MS :
					  RFX_THERMAL_POLL_IDLE_MS;
	schedule_delayed_work(&rfx_thermal_work, msecs_to_jiffies(delay_ms));
}

/* ===================================================================== */
/* Input handler (daily touch boost; off during gaming)                  */
/* ===================================================================== */

static void rfx_input_event(struct input_handle *handle, unsigned int type,
			    unsigned int code, int value)
{
	if (rfx_gaming_enabled())
		return;
	if (type == EV_ABS || type == EV_KEY)
		atomic64_set(&rfx_input_ts_ns, ktime_get_ns());
}

static int rfx_input_connect(struct input_handler *handler,
			     struct input_dev *dev,
			     const struct input_device_id *id)
{
	struct input_handle *handle;
	int err;

	handle = kzalloc(sizeof(*handle), GFP_KERNEL);
	if (!handle)
		return -ENOMEM;

	handle->dev = dev;
	handle->handler = handler;
	handle->name = "vorpal";

	err = input_register_handle(handle);
	if (err)
		goto err_free;
	err = input_open_device(handle);
	if (err)
		goto err_unregister;
	return 0;

err_unregister:
	input_unregister_handle(handle);
err_free:
	kfree(handle);
	return err;
}

static void rfx_input_disconnect(struct input_handle *handle)
{
	input_close_device(handle);
	input_unregister_handle(handle);
	kfree(handle);
}

static const struct input_device_id rfx_input_ids[] = {
	{
		.flags = INPUT_DEVICE_ID_MATCH_EVBIT |
			 INPUT_DEVICE_ID_MATCH_ABSBIT,
		.evbit = { BIT_MASK(EV_ABS) },
		.absbit = { [BIT_WORD(ABS_MT_POSITION_X)] =
			    BIT_MASK(ABS_MT_POSITION_X) },
	},
	{
		.flags = INPUT_DEVICE_ID_MATCH_EVBIT |
			 INPUT_DEVICE_ID_MATCH_KEYBIT,
		.evbit = { BIT_MASK(EV_KEY) },
		.keybit = { [BIT_WORD(BTN_TOUCH)] = BIT_MASK(BTN_TOUCH) },
	},
	{ },
};

static struct input_handler rfx_input_handler = {
	.event		= rfx_input_event,
	.connect	= rfx_input_connect,
	.disconnect	= rfx_input_disconnect,
	.name		= "vorpal",
	.id_table	= rfx_input_ids,
};

/* ===================================================================== */
/* sysfs                                                                 */
/* ===================================================================== */

static struct rfx_tunables *rfx_global_tunables;
static DEFINE_MUTEX(rfx_global_tunables_lock);

static ssize_t rate_limit_us_show(struct gov_attr_set *attr_set, char *buf)
{
	return sprintf(buf, "%u\n", to_rfx_tunables(attr_set)->rate_limit_us);
}
static ssize_t rate_limit_us_store(struct gov_attr_set *attr_set,
				   const char *buf, size_t count)
{
	struct rfx_tunables *t = to_rfx_tunables(attr_set);
	struct rfx_policy *p;
	unsigned int val;

	if (kstrtouint(buf, 10, &val))
		return -EINVAL;
	t->rate_limit_us = val;
	list_for_each_entry(p, &attr_set->policy_list, tunables_hook)
		p->freq_update_delay_ns = (s64)val * NSEC_PER_USEC;
	return count;
}
static struct governor_attr rate_limit_us = __ATTR_RW(rate_limit_us);

static ssize_t up_rate_limit_us_show(struct gov_attr_set *attr_set, char *buf)
{
	return sprintf(buf, "%u\n", to_rfx_tunables(attr_set)->up_rate_limit_us);
}
static ssize_t up_rate_limit_us_store(struct gov_attr_set *attr_set,
				      const char *buf, size_t count)
{
	struct rfx_tunables *t = to_rfx_tunables(attr_set);
	unsigned int val;

	if (kstrtouint(buf, 10, &val))
		return -EINVAL;
	t->up_rate_limit_us = val;
	return count;
}
static struct governor_attr up_rate_limit_us = __ATTR_RW(up_rate_limit_us);

static ssize_t down_rate_limit_us_show(struct gov_attr_set *attr_set, char *buf)
{
	return sprintf(buf, "%u\n", to_rfx_tunables(attr_set)->down_rate_limit_us);
}
static ssize_t down_rate_limit_us_store(struct gov_attr_set *attr_set,
					const char *buf, size_t count)
{
	struct rfx_tunables *t = to_rfx_tunables(attr_set);
	unsigned int val;

	if (kstrtouint(buf, 10, &val))
		return -EINVAL;
	t->down_rate_limit_us = val;
	return count;
}
static struct governor_attr down_rate_limit_us = __ATTR_RW(down_rate_limit_us);

/* Reset transient gaming residue on every live policy (all clusters). */
static void rfx_reset_all_policies(void)
{
	struct rfx_policy *p;
	unsigned long flags;

	atomic64_set(&rfx_frame_boost_end_ns, 0);

	spin_lock_irqsave(&rfx_policy_list_lock, flags);
	list_for_each_entry(p, &rfx_policy_list, gov_node) {
		/*
		 * Clear transient per-policy state so gaming residue does not leak
		 * into daily (and vice versa). thermal_applied_pct is read in BOTH
		 * modes, so a value left throttled-down by a hot gaming session
		 * would otherwise pin daily frequency low until it slowly walked
		 * back up.
		 */
		p->thermal_applied_pct = 100;
		p->prime_busy_hold_ns = 0;
		p->little_busy_hold_ns = 0;
		p->ui_boost_end_ns = 0;
		p->prev_upct = 0;
		p->need_freq_update = true;
	}
	spin_unlock_irqrestore(&rfx_policy_list_lock, flags);
}

static ssize_t gaming_mode_show(struct gov_attr_set *attr_set, char *buf)
{
	return sprintf(buf, "%u\n", to_rfx_tunables(attr_set)->gaming_mode);
}
static ssize_t gaming_mode_store(struct gov_attr_set *attr_set,
				 const char *buf, size_t count)
{
	struct rfx_tunables *t = to_rfx_tunables(attr_set);
	unsigned int val;

	if (kstrtouint(buf, 10, &val))
		return -EINVAL;
	if (val > 1)
		return -EINVAL;

	t->gaming_mode = val;
	atomic_set(&rfx_gaming, val);
	/* Drive the scheduler-side gaming biases in lockstep (KMI-safe int). */
	WRITE_ONCE(sched_gaming_active, (int)val);

	if (!val) {
		atomic_set(&rfx_frame_time_us, 0);
		/* Flush per-CPU EMA so gaming residue doesn't bias daily freq. */
		{
			int cpu;
			for_each_possible_cpu(cpu) {
				per_cpu(rfx_cpu, cpu).filt_util = 0;
				per_cpu(rfx_cpu, cpu).win_start_ns = 0;
				per_cpu(rfx_cpu, cpu).win_peak = 0;
			}
		}
		atomic_set(&rfx_frames_seen, 0);
		atomic_set(&rfx_janks_seen, 0);
		atomic_set(&rfx_jank_pct, 0);
		atomic_set(&rfx_ft_ema_us, 0);
		atomic_set(&rfx_cadence_bud_us, 0);
		atomic_set(&rfx_cadence_stable, 0);
		atomic64_set(&rfx_last_seq_consumed, 0);
		atomic64_set(&rfx_prev_present_ns, 0);
		atomic64_set(&rfx_last_frame_feed_ns, 0);
		atomic64_set(&rfx_stale_gpu_next_ns, 0);
		atomic64_set(&rfx_cpu_pressure_next_ns, 0);
		atomic_set(&rfx_sys_perf_pct, 0);
		atomic_set(&rfx_perf_demand_pct, 0);
		atomic_set(&rfx_prime_perf_pct, 0);
		atomic_set(&rfx_sys_capped_pct, 100);
		atomic_set(&rfx_prime_capped_pct, 100);
		rfx_reset_all_policies();
		atomic64_set(&rfx_gaming_warmup_end_ns, 0);
		atomic64_set(&rfx_gpu_burst_next_ns, 0);
		atomic64_set(&rfx_last_perf_busy_ns, 0);
		atomic64_set(&vorpal_gpu_boost_until_ns, 0);
		atomic64_set(&vorpal_gpu_hard_boost_until_ns, 0);
		atomic64_set(&vorpal_gpu_launch_until_ns, 0);
#ifdef CONFIG_THERMAL
		/* Unbind the auto-bound skin zone -> back to (none) for daily. */
		rfx_tz_release_auto();
#endif
	} else {
		u64 now = ktime_get_ns();

		/*
		 * Restore the builtin gaming frame budget unless userspace pinned
		 * one via frame_budget_us. This is the governor's sustain target,
		 * not an FPS cap on the game.
		 */
		if (!rfx_frame_budget_user_set)
			atomic_set(&rfx_frame_budget_us, RFX_FRAME_BUDGET_US_GAMING);
		/* Fresh frame-pacing baseline so the first interval is sane. */
		atomic_set(&rfx_ft_ema_us, 0);
		atomic_set(&rfx_cadence_bud_us, 0);
		atomic_set(&rfx_cadence_stable, 0);
		atomic64_set(&rfx_last_seq_consumed, 0);
		atomic64_set(&rfx_prev_present_ns, 0);
		atomic64_set(&rfx_last_frame_feed_ns, now);
		atomic64_set(&rfx_stale_gpu_next_ns, 0);
		atomic64_set(&rfx_cpu_pressure_next_ns, 0);
		atomic_set(&rfx_sys_perf_pct, 0);
		atomic_set(&rfx_perf_demand_pct, 0);
		atomic_set(&rfx_prime_perf_pct, 0);
		atomic_set(&rfx_sys_capped_pct, 100);
		atomic_set(&rfx_prime_capped_pct, 100);
		atomic64_set(&vorpal_gpu_boost_until_ns, 0);
		atomic64_set(&vorpal_gpu_hard_boost_until_ns, 0);
		atomic64_set(&vorpal_gpu_launch_until_ns, 0);
		/* Flush per-CPU EMA so daily residue doesn't hold freq low. */
		{
			int cpu;
			for_each_possible_cpu(cpu) {
				per_cpu(rfx_cpu, cpu).filt_util = 0;
				per_cpu(rfx_cpu, cpu).win_start_ns = 0;
				per_cpu(rfx_cpu, cpu).win_peak = 0;
			}
		}
		/*
		 * Arm the launch window now. With manual activation this fires while
		 * still on the home screen, so it is only the first line of defence -
		 * the util-driven detector (rfx_render_resume_detect) re-arms it when
		 * gameplay actually starts rendering, regardless of when gaming was
		 * toggled or whether a DRM present feed exists.
		 */
		atomic64_set(&rfx_last_perf_busy_ns, 0);
		rfx_arm_launch_window(now);
#ifdef CONFIG_THERMAL
		/* Re-probe the skin zone for this session (driver may be late). */
		rfx_tz_tries = 0;
#endif
		/* Sample temperature sooner once gaming begins (also auto-binds). */
		mod_delayed_work(system_wq, &rfx_thermal_work,
				 msecs_to_jiffies(RFX_THERMAL_POLL_GAMING_MS));
	}
	return count;
}
static struct governor_attr gaming_mode = __ATTR_RW(gaming_mode);

static ssize_t frame_budget_us_show(struct gov_attr_set *attr_set, char *buf)
{
	return sprintf(buf, "%d\n", atomic_read(&rfx_frame_budget_us));
}
static ssize_t frame_budget_us_store(struct gov_attr_set *attr_set,
				     const char *buf, size_t count)
{
	unsigned int val;

	if (kstrtouint(buf, 10, &val) || val < 1000)
		return -EINVAL;
	atomic_set(&rfx_frame_budget_us, val);
	rfx_frame_budget_user_set = true;	/* honour the manual pin over the gaming default */
	return count;
}
static struct governor_attr frame_budget_us = __ATTR_RW(frame_budget_us);

static ssize_t frame_time_us_show(struct gov_attr_set *attr_set, char *buf)
{
	return sprintf(buf, "%d\n", atomic_read(&rfx_frame_time_us));
}
static ssize_t frame_time_us_store(struct gov_attr_set *attr_set,
				   const char *buf, size_t count)
{
	unsigned int val;

	if (kstrtouint(buf, 10, &val))
		return -EINVAL;
	atomic_set(&rfx_frame_time_us, val);
	return count;
}
static struct governor_attr frame_time_us = __ATTR_RW(frame_time_us);

static ssize_t jank_pct_show(struct gov_attr_set *attr_set, char *buf)
{
	return sprintf(buf, "%d\n", atomic_read(&rfx_jank_pct));
}
static struct governor_attr jank_pct = __ATTR_RO(jank_pct);

/*
 * RO diagnostic: the thermal cap the governor is currently publishing (percent
 * of fmax). 100 = not throttling. If a cluster's freq sags while this reads
 * <100, the governor thermal step is the cause (raise the breakpoints / check
 * the bound zone scale); if it reads 100, the sag is util-following / placement.
 */
static ssize_t cap_pct_show(struct gov_attr_set *attr_set, char *buf)
{
	return sprintf(buf, "%d\n", atomic_read(&rfx_thermal_cap_pct));
}
static struct governor_attr cap_pct = __ATTR_RO(cap_pct);

static struct attribute *rfx_little_attrs[] = {
	&rate_limit_us.attr,
	&up_rate_limit_us.attr,
	&down_rate_limit_us.attr,
	NULL
};
ATTRIBUTE_GROUPS(rfx_little);

static struct attribute *rfx_big_attrs[] = {
	&rate_limit_us.attr,
	&up_rate_limit_us.attr,
	&down_rate_limit_us.attr,
	NULL
};
ATTRIBUTE_GROUPS(rfx_big);

/*
 * Prime cluster: the gaming profile is fully builtin - all floors, rates, and
 * thresholds are fixed RFX_* constants (skin thermal auto-binds on gaming), so
 * there are no per-cluster tuning knobs to set. Only the master gaming_mode
 * switch, the optional userspace frame feed, and read-only telemetry remain.
 */
static struct attribute *rfx_prime_attrs[] = {
	&rate_limit_us.attr,
	&up_rate_limit_us.attr,
	&down_rate_limit_us.attr,
	&gaming_mode.attr,
	&frame_budget_us.attr,
	&frame_time_us.attr,
	&jank_pct.attr,
	&cap_pct.attr,
	NULL
};
ATTRIBUTE_GROUPS(rfx_prime);

static void rfx_tunables_free(struct kobject *kobj)
{
	kfree(to_rfx_tunables(rfx_to_gov_attr_set(kobj)));
}

static struct kobj_type rfx_little_ktype = {
	.default_groups = rfx_little_groups,
	.sysfs_ops = &governor_sysfs_ops,
	.release = rfx_tunables_free,
};
static struct kobj_type rfx_big_ktype = {
	.default_groups = rfx_big_groups,
	.sysfs_ops = &governor_sysfs_ops,
	.release = rfx_tunables_free,
};
static struct kobj_type rfx_prime_ktype = {
	.default_groups = rfx_prime_groups,
	.sysfs_ops = &governor_sysfs_ops,
	.release = rfx_tunables_free,
};

static struct cpufreq_governor vorpal_gov;

/* ===================================================================== */
/* Allocation / kthread                                                  */
/* ===================================================================== */

static struct rfx_policy *rfx_policy_alloc(struct cpufreq_policy *policy)
{
	struct rfx_policy *p;

	p = kzalloc(sizeof(*p), GFP_KERNEL);
	if (!p)
		return NULL;
	p->policy = policy;
	p->thermal_applied_pct = 100;
	raw_spin_lock_init(&p->update_lock);
	INIT_LIST_HEAD(&p->gov_node);
	return p;
}

static void rfx_policy_free(struct rfx_policy *p)
{
	kfree(p);
}

static int rfx_kthread_create(struct rfx_policy *p)
{
	struct task_struct *thread;
	struct cpufreq_policy *policy = p->policy;
	struct sched_param sp = { .sched_priority = MAX_RT_PRIO / 2 };
	int ret;

	if (policy->fast_switch_enabled)
		return 0;

	kthread_init_work(&p->work, rfx_work);
	kthread_init_worker(&p->worker);
	thread = kthread_create(kthread_worker_fn, &p->worker, "rfx_gov/%d",
				cpumask_first(policy->related_cpus));
	if (IS_ERR(thread)) {
		pr_err("vorpal: kthread create failed %ld\n", PTR_ERR(thread));
		return PTR_ERR(thread);
	}

	ret = sched_setscheduler_nocheck(thread, SCHED_FIFO, &sp);
	if (ret) {
		kthread_stop(thread);
		pr_warn("vorpal: failed to set SCHED_FIFO\n");
		return ret;
	}

	p->thread = thread;
	if (policy->dvfs_possible_from_any_cpu)
		set_cpus_allowed_ptr(thread, policy->related_cpus);
	else
		kthread_bind_mask(thread, policy->related_cpus);

	init_irq_work(&p->irq_work, rfx_irq_work);
	mutex_init(&p->work_lock);
	wake_up_process(thread);
	return 0;
}

static void rfx_kthread_stop(struct rfx_policy *p)
{
	if (p->policy->fast_switch_enabled)
		return;
	kthread_flush_worker(&p->worker);
	kthread_stop(p->thread);
	mutex_destroy(&p->work_lock);
}

static struct rfx_tunables *rfx_tunables_alloc(struct rfx_policy *p)
{
	struct rfx_tunables *t;

	t = kzalloc(sizeof(*t), GFP_KERNEL);
	if (t) {
		gov_attr_set_init(&t->attr_set, &p->tunables_hook);
		if (!have_governor_per_policy())
			rfx_global_tunables = t;
	}
	return t;
}

static void rfx_clear_global_tunables(void)
{
	if (!have_governor_per_policy())
		rfx_global_tunables = NULL;
}

/* ===================================================================== */
/* Governor callbacks                                                    */
/* ===================================================================== */

static int rfx_init(struct cpufreq_policy *policy)
{
	struct rfx_policy *p;
	struct rfx_tunables *t;
	unsigned long max_cap;
	struct kobj_type *ktype;
	int ret = 0;

	if (policy->governor_data)
		return -EBUSY;

	cpufreq_enable_fast_switch(policy);

	p = rfx_policy_alloc(policy);
	if (!p) {
		ret = -ENOMEM;
		goto disable_fast_switch;
	}

	ret = rfx_kthread_create(p);
	if (ret)
		goto free_p;

	max_cap = arch_scale_cpu_capacity(cpumask_first(policy->cpus));
	p->is_prime = rfx_cap_is_prime(max_cap);
	p->is_little = rfx_cap_is_little(max_cap);

	mutex_lock(&rfx_global_tunables_lock);

	if (rfx_global_tunables) {
		if (WARN_ON(have_governor_per_policy())) {
			ret = -EINVAL;
			goto stop_kthread;
		}
		policy->governor_data = p;
		p->tunables = rfx_global_tunables;
		gov_attr_set_get(&rfx_global_tunables->attr_set, &p->tunables_hook);
		goto out;
	}

	t = rfx_tunables_alloc(p);
	if (!t) {
		ret = -ENOMEM;
		goto stop_kthread;
	}

	if (p->is_little) {
		t->cluster_type = RFX_CLUSTER_LITTLE;
		t->rate_limit_us = RFX_LITTLE_RATE_US;
		t->up_rate_limit_us = RFX_LITTLE_UP_US;
		t->down_rate_limit_us = RFX_LITTLE_DOWN_US;
		ktype = &rfx_little_ktype;
	} else if (p->is_prime) {
		t->cluster_type = RFX_CLUSTER_PRIME;
		t->rate_limit_us = RFX_PRIME_RATE_US;
		t->up_rate_limit_us = RFX_PRIME_UP_US;
		t->down_rate_limit_us = RFX_PRIME_DOWN_US;
		ktype = &rfx_prime_ktype;
	} else {
		t->cluster_type = RFX_CLUSTER_BIG;
		t->rate_limit_us = RFX_BIG_RATE_US;
		t->up_rate_limit_us = RFX_BIG_UP_US;
		t->down_rate_limit_us = RFX_BIG_DOWN_US;
		ktype = &rfx_big_ktype;
	}

	policy->governor_data = p;
	p->tunables = t;

	ret = kobject_init_and_add(&t->attr_set.kobj, ktype,
				   get_governor_parent_kobj(policy),
				   "%s", vorpal_gov.name);
	if (ret)
		goto fail;

out:
	p->freq_update_delay_ns = (s64)p->tunables->rate_limit_us * NSEC_PER_USEC;
	p->up_rate_delay_ns = (s64)p->tunables->up_rate_limit_us * NSEC_PER_USEC;
	p->down_rate_delay_ns = (s64)p->tunables->down_rate_limit_us * NSEC_PER_USEC;
	mutex_unlock(&rfx_global_tunables_lock);
	return 0;

fail:
	kobject_put(&t->attr_set.kobj);
	policy->governor_data = NULL;
	rfx_clear_global_tunables();
stop_kthread:
	rfx_kthread_stop(p);
	mutex_unlock(&rfx_global_tunables_lock);
free_p:
	rfx_policy_free(p);
disable_fast_switch:
	cpufreq_disable_fast_switch(policy);
	pr_err("vorpal: init failed error %d\n", ret);
	return ret;
}

static void rfx_exit(struct cpufreq_policy *policy)
{
	struct rfx_policy *p = policy->governor_data;
	struct rfx_tunables *t = p->tunables;
	unsigned int count;

	mutex_lock(&rfx_global_tunables_lock);
	count = gov_attr_set_put(&t->attr_set, &p->tunables_hook);
	policy->governor_data = NULL;
	if (!count) {
		rfx_clear_global_tunables();
		/*
		 * Last policy releasing vorpal: the governor is going away (switch
		 * or rmmod). Clear the gaming signal so the scheduler biases and the
		 * gaming-synced I/O schedulers do not stay latched with no governor
		 * driving them, and drop any GPU floor we were holding. Not reached
		 * on a transient suspend stop (that keeps the policy), so gaming
		 * survives suspend/resume as before.
		 */
		atomic_set(&rfx_gaming, 0);
		WRITE_ONCE(sched_gaming_active, 0);
		atomic64_set(&rfx_gaming_warmup_end_ns, 0);
		atomic64_set(&vorpal_gpu_boost_until_ns, 0);
		atomic64_set(&vorpal_gpu_hard_boost_until_ns, 0);
		atomic64_set(&vorpal_gpu_launch_until_ns, 0);
	}
	mutex_unlock(&rfx_global_tunables_lock);

	rfx_kthread_stop(p);
	rfx_policy_free(p);
	cpufreq_disable_fast_switch(policy);
}

static int rfx_start(struct cpufreq_policy *policy)
{
	struct rfx_policy *p = policy->governor_data;
	void (*uu)(struct update_util_data *data, u64 time, unsigned int flags);
	unsigned long flags;
	unsigned int cpu;

	p->freq_update_delay_ns = (s64)p->tunables->rate_limit_us * NSEC_PER_USEC;
	p->up_rate_delay_ns = (s64)p->tunables->up_rate_limit_us * NSEC_PER_USEC;
	p->down_rate_delay_ns = (s64)p->tunables->down_rate_limit_us * NSEC_PER_USEC;

	p->last_upfreq_time = 0;
	p->last_downfreq_time = 0;
	p->next_freq = policy->cur > 0 ? policy->cur : policy->cpuinfo.min_freq;
	p->cached_raw_freq = 0;
	p->work_in_progress = false;
	p->limits_changed = false;
	p->need_freq_update = false;
	p->prev_upct = 0;
	p->ui_boost_end_ns = 0;
	p->prime_busy_hold_ns = 0;
	p->little_busy_hold_ns = 0;
	p->thermal_applied_pct = 100;
	p->thermal_step_ns = ktime_get_ns();

	spin_lock_irqsave(&rfx_policy_list_lock, flags);
	list_add(&p->gov_node, &rfx_policy_list);
	spin_unlock_irqrestore(&rfx_policy_list_lock, flags);

	for_each_cpu(cpu, policy->cpus) {
		struct rfx_cpu *rfx_c = per_cpu_ptr(&rfx_cpu, cpu);

		memset(rfx_c, 0, sizeof(*rfx_c));
		rfx_c->cpu = cpu;
		rfx_c->rfx_policy = p;
	}

	uu = policy_is_shared(policy) ? rfx_update_shared : rfx_update_single_freq;
	for_each_cpu(cpu, policy->cpus)
		cpufreq_add_update_util_hook(cpu, &per_cpu_ptr(&rfx_cpu, cpu)->update_util, uu);
	return 0;
}

static void rfx_stop(struct cpufreq_policy *policy)
{
	struct rfx_policy *p = policy->governor_data;
	unsigned long flags;
	unsigned int cpu;

	for_each_cpu(cpu, policy->cpus)
		cpufreq_remove_update_util_hook(cpu);

	synchronize_rcu();

	spin_lock_irqsave(&rfx_policy_list_lock, flags);
	list_del(&p->gov_node);
	spin_unlock_irqrestore(&rfx_policy_list_lock, flags);

	if (!policy->fast_switch_enabled) {
		irq_work_sync(&p->irq_work);
		kthread_cancel_work_sync(&p->work);
	}
}

static void rfx_limits(struct cpufreq_policy *policy)
{
	struct rfx_policy *p = policy->governor_data;

	if (!policy->fast_switch_enabled) {
		mutex_lock(&p->work_lock);
		cpufreq_policy_apply_limits(policy);
		mutex_unlock(&p->work_lock);
	}
	smp_wmb();
	WRITE_ONCE(p->limits_changed, true);
}

static struct cpufreq_governor vorpal_gov = {
	.name = CPUFREQ_VORPAL_NAME,
	.owner = THIS_MODULE,
	.flags = CPUFREQ_GOV_DYNAMIC_SWITCHING,
	.init = rfx_init,
	.exit = rfx_exit,
	.start = rfx_start,
	.stop = rfx_stop,
	.limits = rfx_limits,
};

#ifdef CONFIG_CPU_FREQ_DEFAULT_GOV_VORPAL
struct cpufreq_governor *cpufreq_default_governor(void)
{
	return &vorpal_gov;
}
#endif

static int __init vorpal_gov_init(void)
{
	int ret;

	pr_info("Vorpal Governor v%s by %s\n", CPUFREQ_VORPAL_VERSION,
		CPUFREQ_VORPAL_AUTHOR);

	INIT_DEFERRABLE_WORK(&rfx_thermal_work, rfx_thermal_fn);
	schedule_delayed_work(&rfx_thermal_work,
			      msecs_to_jiffies(RFX_THERMAL_POLL_IDLE_MS));

	if (input_register_handler(&rfx_input_handler))
		pr_warn("vorpal: input handler register failed (touch boost off)\n");

	ret = cpufreq_register_governor(&vorpal_gov);
	if (ret) {
		input_unregister_handler(&rfx_input_handler);
		cancel_delayed_work_sync(&rfx_thermal_work);
	}
	return ret;
}

static void __exit vorpal_gov_exit(void)
{
	cpufreq_unregister_governor(&vorpal_gov);
	input_unregister_handler(&rfx_input_handler);
	cancel_delayed_work_sync(&rfx_thermal_work);
}

module_init(vorpal_gov_init);
module_exit(vorpal_gov_exit);

MODULE_AUTHOR("Steambot12");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Vorpal CPUFreq Governor");
