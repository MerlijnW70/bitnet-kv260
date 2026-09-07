#!/bin/sh
cd "$(dirname "$0")"
log=${LOG:-bitnet-power.log}
marks=${MARKS:-bitnet-power.marks}
ids=${IDS:-ids-power.txt}
run=${RUN:-bitnet-power-run.log}
idle=${IDLE:-10}
: > "$log"
: > "$marks"
rm -f "$marks.first"
python3 ina260-sample.py "$log" "${PERIOD:-0.05}" &
sampler=$!
sleep "$idle"
echo "t_start $(date +%s.%N)" >> "$marks"
"${EXE:-./bitnet_kria}" --max-new "${NEW:-256}" --context "${CTX:-512}" --threads "${THREADS:-4}" --timing ${FLAGS:-} < "$ids" 2> "$run.err" |
while IFS= read -r line; do
  printf '%s\n' "$line"
  case "$line" in
    "  setup "*) echo "t_setup $(date +%s.%N)" >> "$marks" ;;
    done*)       echo "t_done $(date +%s.%N)" >> "$marks" ;;
    [0-9]*)      [ -f "$marks.first" ] || { echo "t_first $(date +%s.%N)" >> "$marks"; : > "$marks.first"; } ;;
  esac
done > "$run"
echo "t_end $(date +%s.%N)" >> "$marks"
sleep "$idle"
kill $sampler
rm -f "$marks.first"
grep -E "^  |^prompt |^done |^per forward" "$run"
awk -v M="$marks" 'BEGIN { while ((getline l < M) > 0) { split(l, f, " "); m[f[1]] = f[2] + 0 }
  printf "setup %.3f s, prompt %.3f s, generation %.3f s, whole run %.3f s\n",
         m["t_setup"] - m["t_start"], m["t_first"] - m["t_setup"],
         m["t_done"] - m["t_first"], m["t_end"] - m["t_start"] }'
awk -v M="$marks" '
  BEGIN { while ((getline l < M) > 0) { split(l, f, " "); m[f[1]] = f[2] + 0 } }
  { p = $2 / 1e6; t = $1 + 0;
    ph = (t < m["t_start"]) ? "1 idle-before" :
         (t < m["t_setup"]) ? "2 setup" :
         (t < m["t_first"]) ? "3 prompt" :
         (t < m["t_done"])  ? "4 generation" :
         (t < m["t_end"])   ? "5 tail" : "6 idle-after";
    n[ph]++; s[ph] += p; if (!(ph in mx) || p > mx[ph]) mx[ph] = p; if (!(ph in mn) || p < mn[ph]) mn[ph] = p;
    sc[ph] += $3; sv[ph] += $4 }
  END { for (ph in n) printf "%-14s %4d samples  power mean %.3f W  min %.3f W  max %.3f W   mean %.0f mA at %.3f V\n",
                             ph, n[ph], s[ph] / n[ph], mn[ph], mx[ph], sc[ph] / n[ph], sv[ph] / n[ph] / 1000 }' "$log" | sort
