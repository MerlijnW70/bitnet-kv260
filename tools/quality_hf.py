"""Run the quality battery through the HF checkpoint in bf16 on the RTX 4080, greedy.

usage (on the PC):
  python quality_hf.py quality_prompts.json hf-answers.json

Loading is ask.py's: microsoft/bitnet-b1.58-2B-4T with TORCHDYNAMO_DISABLE=1, the uint8 ternary
weights read as int8 where the shape already has one row a neuron and unpacked otherwise, then bf16
on the card. The prompt text is built with the same template the board uses (bitnet_chat.jinja,
which is the checkpoint's own), through the tokenizer's apply_chat_template, and generation is
do_sample=False with max_new_tokens = the battery's max_new for that prompt.
"""
import argparse
import json
import os
import sys
import time

os.environ["TORCHDYNAMO_DISABLE"] = "1"
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "microsoft/bitnet-b1.58-2B-4T"


def load():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16)
    from transformers.integrations.bitnet import unpack_weights
    packed = 0
    for mod in model.modules():
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.dtype == torch.uint8 and hasattr(mod, "weight_scale"):
            out_features = getattr(mod, "out_features", None)
            if out_features is not None and w.shape[0] == out_features:
                mod.weight = torch.nn.Parameter(w.view(torch.int8).to(torch.bfloat16),
                                                requires_grad=False)
            else:
                mod.weight = torch.nn.Parameter(unpack_weights(w, dtype=torch.bfloat16),
                                                requires_grad=False)
            packed += 1
    model = model.to("cuda").eval()
    print(f"loaded {NAME} in {time.time() - t0:.0f} s; unpacked {packed} ternary layers", flush=True)
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--only", default=None, help="comma-separated prompt ids")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    spec = json.load(open(a.prompts, encoding="utf-8"))
    system = spec.get("system")
    prompts = spec["prompts"]
    if a.only:
        want = set(a.only.split(","))
        prompts = [p for p in prompts if p["id"] in want]

    tok, model = load()
    eos = [i for i in (128001, 128009) if i is not None]
    runs = []
    for p in prompts:
        for rep in range(int(p.get("repeats", 1))):
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": p["prompt"]})
            text_in = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            ids = tok(text_in, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=int(p["max_new"]), do_sample=False,
                                     eos_token_id=eos)
            dt = time.time() - t0
            new = out[0][ids.shape[1]:].tolist()
            answer = tok.decode([t for t in new if t not in eos], skip_special_tokens=True)
            rec = {"id": p["id"], "category": p["category"], "rep": rep,
                   "max_new": int(p["max_new"]), "prompt": p["prompt"], "system": system,
                   "prompt_text": text_in, "prompt_ids": ids[0].tolist(), "gen_ids": new,
                   "answer": answer, "wall_s": round(dt, 3),
                   "prompt_tokens": int(ids.shape[1]), "gen_tokens": len(new),
                   "gen_tps": round(len(new) / dt, 3),
                   "hit_cap": len(new) >= int(p["max_new"])}
            runs.append(rec)
            print(f"=== {p['id']} rep {rep} [{p['category']}] max_new {p['max_new']}")
            print(f"> {p['prompt']}")
            print(answer)
            print(f"{len(new)} tokens in {dt:.1f} s ({len(new) / dt:.1f} tok/s), "
                  f"{ids.shape[1]} prompt ids, hit_cap {rec['hit_cap']}")
            sys.stdout.flush()

    json.dump({"way": "hf_bf16_rtx4080", "system": system, "model": NAME,
               "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
               "runs": runs}, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    det = [x for x in runs if x["id"] == "q24_determinism"]
    if len(det) >= 2:
        print(f"\ndeterminism q24: identical ids {det[0]['gen_ids'] == det[1]['gen_ids']}")
    print(f"{len(runs)} runs written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
