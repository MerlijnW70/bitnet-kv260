#!/bin/sh
B=${BITNET:-/home/ubuntu/bitnet}
log=${LOG:-$B/bitnetcpp-power.log}
out=${OUT:-$B/bitnetcpp-power-run}
idle=${IDLE:-10}
sampler_py=${SAMPLER:-/home/ubuntu/bitnet-kria/ina260-sample.py}
: > "$log"
python3 "$sampler_py" "$log" "${PERIOD:-0.05}" &
sampler=$!
sleep "$idle"
t0=$(date +%s.%N)
cd "$B/BitNet"
if [ -n "${SYS-}" ]; then
  set -- -sys "$SYS"
else
  set --
fi
/usr/bin/time -v build/bin/llama-completion -m "$B/ggml-model-i2_s.gguf" -t "${THREADS:-4}" \
  -n "${NEW:-256}" -c "${CTX:-2048}" --temp 0 -ngl 0 \
  --override-kv tokenizer.ggml.pre=str:llama-bpe --jinja --chat-template-file "$B/bitnet_chat.jinja" \
  -st "$@" -p "${PROMPT:-Name three things a shift register is used for.}" \
  > "$out.out" 2> "$out.err"
rc=$?
t1=$(date +%s.%N)
sleep "$idle"
kill $sampler
echo "llama-completion exit $rc, wall $(echo "$t1 - $t0" | bc) s"
grep -E "eval time|total time|load time|Maximum resident" "$out.err"
awk -v t0="$t0" -v t1="$t1" '
  { p = $2 / 1e6; ph = ($1 < t0) ? "before" : ($1 <= t1) ? "during" : "after";
    n[ph]++; s[ph] += p; if (!(ph in mx) || p > mx[ph]) mx[ph] = p; if (!(ph in mn) || p < mn[ph]) mn[ph] = p;
    sc[ph] += $3; sv[ph] += $4 }
  END { for (ph in n) printf "%-6s %4d samples  power mean %.3f W  min %.3f W  max %.3f W   mean %.0f mA at %.3f V\n",
                             ph, n[ph], s[ph] / n[ph], mn[ph], mx[ph], sc[ph] / n[ph], sv[ph] / n[ph] / 1000 }' "$log" | sort
