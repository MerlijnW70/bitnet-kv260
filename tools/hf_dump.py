"""Run microsoft/bitnet-b1.58-2B-4T through transformers on the card and dump every tensor the
numpy reference (ref_model.py) is checked against: the hidden state after every layer, layer 0's
attention and FFN intermediates, the final logits, and a greedy continuation.

usage: python hf_dump.py [--prompt "..."] [--new 32] [--layer0-tokens 8] [--out DIR]

The model is loaded exactly as ask.py and hf_ffn.py load it (TORCHDYNAMO_DISABLE=1, bf16, the
packed uint8 BitLinear weights unpacked by hand) with attn_implementation="eager" so the
attention is the documented softmax path. Hidden states are taken with forward hooks on
embed_tokens, on each decoder layer and on the final norm, so the indexing is unambiguous:
hidden[0] is the embedding output, hidden[l + 1] the residual stream after layer l, and
final_norm the output of model.norm.

Writes into DIR (default this directory), with T the prompt length and P the layer-0 positions
(the first --layer0-tokens tokens):
  hf_hidden.npy      float32 [31, T, 2560]  the residual stream, hidden[0] = embeddings
  hf_final_norm.npy  float32 [T, 2560]      model.norm output
  hf_logits.npy      float32 [T, 128256]    lm_head output (bf16 matmul, kept as float32)
  hf_l0_attn_in.npy  float32 [P, 2560]      layer 0 input_layernorm output
  hf_l0_q.npy        float32 [P, 20, 128]   q after RoPE      hf_l0_k.npy  [P, 5, 128]
  hf_l0_v.npy        float32 [P, 5, 128]    v (no RoPE)
  hf_l0_attn_out.npy float32 [P, 2560]      the attention output, before attn_sub_norm
  hf_l0_attn_sub.npy float32 [P, 2560]      attn_sub_norm output, the o_proj input
  hf_l0_o.npy        float32 [P, 2560]      o_proj output
  hf_l0_mlp_in.npy   float32 [P, 2560]      post_attention_layernorm output
  hf_l0_g.npy        float32 [P, 6912]      gate_proj output   hf_l0_u.npy  [P, 6912]
  hf_l0_h.npy        float32 [P, 6912]      ffn_sub_norm output, the down_proj input
  hf_l0_q8.npy       int8    [P, 6912]      transformers' own int8 quantisation of that input
  hf_l0_q8_scale.npy float32 [P]            its scale (127 / max|h|)
  hf_l0_d.npy        float32 [P, 2560]      down_proj output
  hf_gen_ids.npy     int64   [new]          greedy continuation, do_sample=False
  hf_dump.json                              prompt, chat text, token ids, weight scales of every
                                            layer-0 projection, the model geometry, versions, the
                                            greedy text, and the self-checks

q and k after RoPE are recomputed here from the captured q_proj/k_proj outputs and the captured
(cos, sin) with the same bf16 arithmetic transformers uses (rotate_half), and checked against a
second capture through the attention module's own call.
"""
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
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "microsoft/bitnet-b1.58-2B-4T"
PROMPT = "What is the capital of France, and what river runs through it? Explain in three sentences."


def load_model(attn="eager"):
    from transformers.integrations.bitnet import unpack_weights

    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16, attn_implementation=attn)
    packed = 0
    for mod in model.modules():
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.dtype == torch.uint8 and hasattr(mod, "weight_scale"):
            out_features = getattr(mod, "out_features", None)
            if out_features is not None and w.shape[0] == out_features:
                mod.weight = torch.nn.Parameter(w.view(torch.int8).to(torch.bfloat16), requires_grad=False)
            else:
                mod.weight = torch.nn.Parameter(unpack_weights(w, dtype=torch.bfloat16), requires_grad=False)
            packed += 1
    return tok, model.to("cuda"), packed


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x, cos, sin):
    """transformers' apply_rotary_pos_emb on one of q/k, in the tensors' own dtype."""
    return (x * cos) + (rotate_half(x) * sin)


def act_quant(x):
    """transformers.integrations.bitnet.ActQuant's quantisation half: (int8, scale)."""
    a = x.float()
    scale = 127 / a.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    q = (a * scale).round().clamp(-128, 127)
    return q.to(torch.int8), scale


def rope_theta_of(cfg):
    """The RoPE base, wherever this transformers version keeps it."""
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict) and "rope_theta" in params:
        return float(params["rope_theta"])
    return float(cfg.rope_theta)


def main(argv):
    prompt, new_tokens, l0_tokens, out_dir = PROMPT, 32, 8, HERE
    i = 0
    while i < len(argv):
        if argv[i] == "--prompt":
            prompt = argv[i + 1]
        elif argv[i] == "--new":
            new_tokens = int(argv[i + 1])
        elif argv[i] == "--layer0-tokens":
            l0_tokens = int(argv[i + 1])
        elif argv[i] == "--out":
            out_dir = argv[i + 1]
        else:
            raise SystemExit(__doc__)
        i += 2
    os.makedirs(out_dir, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.time()
    tok, model, packed = load_model()
    cfg = model.config
    print(f"loaded {NAME} in {time.time() - t0:.0f} s; unpacked {packed} ternary layers; "
          f"torch {torch.__version__} transformers {transformers.__version__} on {torch.cuda.get_device_name(0)}; "
          f"attn {cfg._attn_implementation}", flush=True)
    n_layers = cfg.num_hidden_layers
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    print(f"geometry: hidden {cfg.hidden_size} inter {cfg.intermediate_size} layers {n_layers} "
          f"heads {n_q}/{n_kv} x {head_dim} rope_theta {rope_theta_of(cfg)} eps {cfg.rms_norm_eps} "
          f"vocab {cfg.vocab_size} tied {cfg.tie_word_embeddings}")

    messages = [{"role": "user", "content": prompt}]
    text_in = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tok(text_in, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    n = ids.shape[1]
    ids_list = ids[0].tolist()
    print(f"prompt {n} tokens: {ids_list}")
    print(f"chat text: {text_in!r}")

    layer = model.model.layers[0]
    attn, mlp = layer.self_attn, layer.mlp
    cap = {}
    hidden = [None] * (n_layers + 1)

    def keep(name):
        def hook(module, args, output):
            cap[name] = output.detach().clone()
        return hook

    def keep_in(name):
        def hook(module, args):
            cap[name] = args[0].detach().clone()
        return hook

    def keep_hidden(idx):
        def hook(module, args, output):
            hidden[idx] = (output[0] if isinstance(output, tuple) else output).detach().clone()
        return hook

    def keep_pos(module, args, kwargs):
        cap["cos"], cap["sin"] = (t.detach().clone() for t in kwargs["position_embeddings"])

    hooks = [model.model.embed_tokens.register_forward_hook(keep_hidden(0)),
             model.model.norm.register_forward_hook(keep("final_norm")),
             attn.register_forward_pre_hook(keep_pos, with_kwargs=True),
             attn.q_proj.register_forward_hook(keep("q_pre")),
             attn.k_proj.register_forward_hook(keep("k_pre")),
             attn.v_proj.register_forward_hook(keep("v")),
             attn.attn_sub_norm.register_forward_pre_hook(keep_in("attn_out")),
             attn.attn_sub_norm.register_forward_hook(keep("attn_sub")),
             attn.o_proj.register_forward_hook(keep("o")),
             layer.input_layernorm.register_forward_hook(keep("attn_in")),
             layer.post_attention_layernorm.register_forward_hook(keep("mlp_in")),
             mlp.gate_proj.register_forward_hook(keep("g")),
             mlp.up_proj.register_forward_hook(keep("u")),
             mlp.ffn_sub_norm.register_forward_hook(keep("h")),
             mlp.down_proj.register_forward_hook(keep("d"))]
    for l in range(n_layers):
        hooks.append(model.model.layers[l].register_forward_hook(keep_hidden(l + 1)))

    with torch.no_grad():
        out = model(ids, use_cache=False)
    for h in hooks:
        h.remove()

    logits = out.logits[0].float()
    next_id = int(logits[-1].argmax())
    print(f"next token by argmax: {next_id} {tok.decode([next_id])!r}")

    p = min(l0_tokens, n)
    sel = slice(0, p)
    cos, sin = cap["cos"], cap["sin"]
    q_pre = cap["q_pre"][0].view(n, n_q, head_dim).transpose(0, 1)
    k_pre = cap["k_pre"][0].view(n, n_kv, head_dim).transpose(0, 1)
    q_rope = apply_rope(q_pre, cos[0], sin[0]).transpose(0, 1)
    k_rope = apply_rope(k_pre, cos[0], sin[0]).transpose(0, 1)
    v_heads = cap["v"][0].view(n, n_kv, head_dim)

    scaling = head_dim ** -0.5
    qh = q_rope.transpose(0, 1).float()
    kh = k_rope.transpose(0, 1).float()
    vh = v_heads.transpose(0, 1).float()
    groups = n_q // n_kv
    mask = torch.full((n, n), float("-inf"), device=qh.device).triu(1)
    rebuilt = torch.empty(n, n_q * head_dim, device=qh.device)
    for hq in range(n_q):
        s = (qh[hq] @ kh[hq // groups].transpose(0, 1)) * scaling + mask
        w = torch.softmax(s, dim=-1)
        rebuilt[:, hq * head_dim:(hq + 1) * head_dim] = w @ vh[hq // groups]
    attn_out = cap["attn_out"][0].float()
    attn_rebuild_err = (rebuilt - attn_out).abs().max().item()
    attn_rebuild_cos = torch.nn.functional.cosine_similarity(rebuilt, attn_out, dim=-1).min().item()
    print(f"attention rebuilt from the captured q, k, v: max |diff| {attn_rebuild_err:.6g}, "
          f"min cosine over positions {attn_rebuild_cos:.8f}")

    q8, q8_scale = act_quant(cap["h"][0])
    deq = (q8.float() / q8_scale).to(torch.bfloat16)
    with torch.no_grad():
        d_recomputed = torch.nn.functional.linear(deq, mlp.down_proj.weight) * mlp.down_proj.weight_scale
    d_err = (d_recomputed.float() - cap["d"][0].float()).abs().max().item()
    print(f"down_proj recomputed from its int8 input: max |diff| {d_err:.6g}")

    arrays = {"hf_hidden": torch.stack([h[0] for h in hidden]).float(),
              "hf_final_norm": cap["final_norm"][0].float(),
              "hf_logits": logits,
              "hf_l0_attn_in": cap["attn_in"][0][sel].float(),
              "hf_l0_q": q_rope[sel].float(),
              "hf_l0_k": k_rope[sel].float(),
              "hf_l0_v": v_heads[sel].float(),
              "hf_l0_attn_out": attn_out[sel],
              "hf_l0_attn_sub": cap["attn_sub"][0][sel].float(),
              "hf_l0_o": cap["o"][0][sel].float(),
              "hf_l0_mlp_in": cap["mlp_in"][0][sel].float(),
              "hf_l0_g": cap["g"][0][sel].float(),
              "hf_l0_u": cap["u"][0][sel].float(),
              "hf_l0_h": cap["h"][0][sel].float(),
              "hf_l0_q8": q8[sel],
              "hf_l0_q8_scale": q8_scale[sel, 0].float(),
              "hf_l0_d": cap["d"][0][sel].float()}
    for name, t in arrays.items():
        a = t.cpu().numpy()
        np.save(os.path.join(out_dir, f"{name}.npy"), a)
        print(f"{name}.npy {a.shape} {a.dtype}")

    t1 = time.time()
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=new_tokens, do_sample=False)
    gen_ids = gen[0][n:].cpu().numpy()
    np.save(os.path.join(out_dir, "hf_gen_ids.npy"), gen_ids)
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    print(f"greedy {len(gen_ids)} tokens in {time.time() - t1:.1f} s: {gen_text.strip()!r}")
    print(f"greedy ids: {gen_ids.tolist()}")

    scales = {}
    for l in range(n_layers):
        lay = model.model.layers[l]
        scales[str(l)] = {"q_proj": lay.self_attn.q_proj.weight_scale.item(),
                          "k_proj": lay.self_attn.k_proj.weight_scale.item(),
                          "v_proj": lay.self_attn.v_proj.weight_scale.item(),
                          "o_proj": lay.self_attn.o_proj.weight_scale.item(),
                          "gate_proj": lay.mlp.gate_proj.weight_scale.item(),
                          "up_proj": lay.mlp.up_proj.weight_scale.item(),
                          "down_proj": lay.mlp.down_proj.weight_scale.item()}
    info = {"model": NAME, "prompt": prompt, "chat_text": text_in, "token_ids": ids_list, "prompt_tokens": n,
            "layer0_positions": list(range(p)), "next_token": {"id": next_id, "text": tok.decode([next_id])},
            "generated_ids": gen_ids.tolist(), "generated_text": gen_text,
            "geometry": {"hidden_size": cfg.hidden_size, "intermediate_size": cfg.intermediate_size,
                         "num_hidden_layers": n_layers, "num_attention_heads": n_q, "num_key_value_heads": n_kv,
                         "head_dim": head_dim, "rope_theta": rope_theta_of(cfg),
                         "rms_norm_eps": cfg.rms_norm_eps, "vocab_size": cfg.vocab_size,
                         "tie_word_embeddings": cfg.tie_word_embeddings, "hidden_act": cfg.hidden_act},
            "weight_scale": scales,
            "checks": {"attention_rebuilt_max_abs_diff": attn_rebuild_err, "attention_rebuilt_min_cosine": attn_rebuild_cos,
                       "down_from_int8_max_abs_diff": d_err},
            "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "device": torch.cuda.get_device_name(0), "attn_implementation": cfg._attn_implementation}
    json.dump(info, open(os.path.join(out_dir, "hf_dump.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"hf_dump.json written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
