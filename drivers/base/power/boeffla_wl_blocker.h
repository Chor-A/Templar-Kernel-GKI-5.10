/*
 * Author: andip71, 01.09.2017
 *
 * Version 1.1.0
 *
 * This software is licensed under the terms of the GNU General Public
 * License version 2, as published by the Free Software Foundation, and
 * may be copied, distributed, and modified under those terms.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 */

#define BOEFFLA_WL_BLOCKER_VERSION	"1.2.0"

/*
 * Default block list. Portable + safe across MTK/QCOM: every entry is either
 * vendor-namespaced (inert where the SoC differs) or a WLAN background-offload
 * wakelock that is safe to drop everywhere (WiFi still connects on wake, only
 * background scan/roam wakeups are suppressed). Deliberately NO device-specific
 * input/i2c/spi nodes and NO alarmtimer/timerfd -- those break wake-on-input or
 * RTC alarms and their numbering is not portable. Per-device offenders belong
 * in the runtime sysfs 'wakelock_blocker' node, not here.
 */
#define LIST_WL_DEFAULT				"RMNET_DFC;DIAG_WS;qcom_rx_wakelock;wlan;wlan_wow_wl;wlan_extscan_wl;wlan_pno_wl;wlan_roam_wl;wlan_ipa;netmgr_wl;NETLINK;a600000.ssusb;998000.qcom,qup_uart;hal_bluetooth_lock;IPA_WS;IPA_CLIENT_APPS_WAN_COAL_CONS;IPA_CLIENT_APPS_WAN_LOW_LAT_CONS;IPA_CLIENT_APPS_LAN_CONS;rmnet_ipa%d;rmnet_ctl;RMNET_SHS"

#define LENGTH_LIST_WL				1024
#define LENGTH_LIST_WL_DEFAULT		(strlen(LIST_WL_DEFAULT) + 1)
#define LENGTH_LIST_WL_SEARCH		LENGTH_LIST_WL + LENGTH_LIST_WL_DEFAULT + 5
