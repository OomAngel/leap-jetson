#!/bin/bash
# Runs Leap V2 2.3.1 leapd (x64) under box64 as the current user for N seconds, RSS watchdog 4 GB.
set -u
DUR=${1:-40}; OUT=~/leap-exp/V2_${2:-run}; rm -rf "$OUT"; mkdir -p "$OUT"
V=~/leap-v2; B=${BOXDIR:-$HOME/src/box64-0.4.4/build}; XL=${XLDIR:-$HOME/src/box64-0.4.4/x64lib}
cd "$V/usr/sbin" || exit 1
env -i HOME=$HOME PATH=/usr/bin:/bin BOX64_NOBANNER=1 \
  BOX64_LD_LIBRARY_PATH=$XL:$V/usr/lib/Leap ${EXTRA_ENV:-} \
  setsid $B/box64 $V/usr/sbin/leapd > "$OUT/leapd.out" 2>&1 &
PID=$!; t=0; peak=0
while [ $t -lt $DUR ] && kill -0 $PID 2>/dev/null; do
  rss=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null || echo 0); [ "${rss:-0}" -gt $peak ] && peak=$rss
  [ "${rss:-0}" -gt 4000000 ] && { echo "KILLED rss $rss"; break; }
  sleep 2; t=$((t+2))
done
kill -0 $PID 2>/dev/null && alive=yes || alive=no
[ -n "${HOLD:-}" ] && { echo "holding pid $PID"; echo $PID > ~/leap-exp/v2.pid; exit 0; }
kill -INT -$PID 2>/dev/null; sleep 3; kill -KILL -$PID 2>/dev/null
echo "alive_at_end=$alive peak_rss_kB=$peak"
