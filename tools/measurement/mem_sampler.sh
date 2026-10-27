#!/bin/bash
PID=$1; OUT=$2; IV=${3:-1}
echo -e "time\tepoch_s\trss_kb\thwm_kb\tgpu_used_mib" > "$OUT"; T0=$(date +%s.%N)
while kill -0 "$PID" 2>/dev/null; do
  r=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null); h=$(awk '/VmHWM/{print $2}' /proc/$PID/status 2>/dev/null)
  g=$(nvidia-smi 2>/dev/null | grep -oE "[0-9]+MiB / [0-9]+MiB" | head -1 | cut -d'M' -f1)
  echo -e "$(date '+%H:%M:%S')\t$(awk -v a=$(date +%s.%N) -v b=$T0 "BEGIN{printf \"%.1f\", a-b}")\t${r:-0}\t${h:-0}\t${g:-0}" >> "$OUT"
  sleep "$IV"
done
