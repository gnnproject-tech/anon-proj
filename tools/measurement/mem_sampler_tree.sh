#!/bin/bash
PID=$1; OUT=$2; IV=${3:-2}; T0=$(date +%s.%N)
echo -e "time\tepoch_s\tpss_kb\trss_kb\tnproc\tgpu_used_mib" > "$OUT"
tree(){ echo $1; for c in $(pgrep -P $1); do tree $c; done; }
while kill -0 $PID 2>/dev/null; do
  P=0; R=0; N=0
  for p in $(tree $PID); do
    v=$(awk '/^Pss:/{p=$2} /^Rss:/{r=$2} END{print p, r}' /proc/$p/smaps_rollup 2>/dev/null); [ -n "$v" ] || continue
    P=$((P + ${v% *})); R=$((R + ${v#* })); N=$((N+1))
  done
  G=$(nvidia-smi 2>/dev/null | grep -oE "[0-9]+MiB / [0-9]+MiB" | head -1 | grep -oE "^[0-9]+")
  echo -e "$(date +%H:%M:%S)\t$(awk -v a=$(date +%s.%N) -v b=$T0 "BEGIN{printf \"%.1f\", a-b}")\t$P\t$R\t$N\t${G:-0}" >> "$OUT"; sleep $IV
done
