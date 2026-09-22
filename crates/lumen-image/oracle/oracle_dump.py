"""Run Qwen-Image-2.1 in the reference pipeline and freeze reference tensors.

Every artifact written here is the numerical contract the Rust implementation is
checked against. Inputs are captured at component boundaries, not just outputs,
so a later stage can be exercised with oracle-supplied inputs and its own error
cannot hide behind upstream error.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_common import component_dtypes, generate_noise, load_pipe


def np_save(path, t):
    np.save(path, t.detach().to(torch.float32).cpu().numpy())


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_rev(repo):
    try:
        rev = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"],
                                      text=True).strip()
        dirty = subprocess.check_output(["git", "-C", repo, "status", "--porcelain"],
                                        text=True).strip()
        return {"commit": rev, "dirty": bool(dirty)}
    except Exception as e:  # pragma: no cover - environment dependent
        return {"error": str(e)}


def environment_record(pipe, model_dir):
    import diffusers
    import huggingface_hub
    import transformers

    ckpt = {}
    for root, _, files in os.walk(model_dir):
        for f in sorted(files):
            if f.endswith((".safetensors", ".json")):
                p = os.path.join(root, f)
                ckpt[os.path.relpath(p, model_dir)] = sha256(p)

    return {
        "model_dir": model_dir,
        "checkpoint_sha256": ckpt,
        "diffusers": {"version": diffusers.__version__,
                      "source": diffusers.__file__,
                      "git": git_rev(os.path.dirname(os.path.dirname(
                          os.path.dirname(diffusers.__file__))))},
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "component_dtypes": component_dtypes(pipe),
        "attention_processor": type(
            pipe.transformer.transformer_blocks[0].attn.processor).__name__,
        "vae_use_tiling": bool(pipe.vae.use_tiling),
        "vae_use_slicing": bool(pipe.vae.use_slicing),
        "scheduler": type(pipe.scheduler).__name__,
        "drop_idx": int(pipe._drop_idx),
        "img_token_id": int(pipe._img_token_id),
        "vae_scale_factor": int(pipe.vae_scale_factor),
        "latent_channels": int(pipe.latent_channels),
        "use_kv_cache": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the Qwen-Image-2.1 checkpoint directory")
    ap.add_argument("--out", required=True, help="where the reference dumps are written")
    ap.add_argument("--prompt", default="A red apple on a wooden table, studio lighting")
    ap.add_argument("--negative", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    pipe = load_pipe(args.model)
    pipe.enable_model_cpu_offload()
    if args.tag == "main":
        with open(os.path.join(args.out, "environment.json"), "w") as f:
            json.dump(environment_record(pipe, args.model), f, indent=2, sort_keys=True)

    # --- transformer inputs/outputs, per call ---
    tx_in, tx_out = [], []

    def tx_pre(module, a, kwargs):
        rec = {}
        for k in ("hidden_states", "encoder_hidden_states", "timestep", "img_mask",
                  "encoder_hidden_states_mask"):
            v = kwargs.get(k)
            rec[k] = (v.detach().to(torch.float32).cpu().numpy()
                      if isinstance(v, torch.Tensor) else None)
        rec["kv_cache_mode"] = kwargs.get("kv_cache_mode")
        rec["img_shapes"] = kwargs.get("img_shapes")
        tx_in.append(rec)

    def tx_post(module, a, kwargs, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        tx_out.append(o.detach().to(torch.float32).cpu().numpy())

    h1 = pipe.transformer.register_forward_pre_hook(tx_pre, with_kwargs=True)
    h2 = pipe.transformer.register_forward_hook(tx_post, with_kwargs=True)

    # Deterministic initial noise, generated here and handed to the pipeline so
    # the exact tensor is recorded rather than re-derived (Rust and PyTorch RNGs
    # are not expected to agree). Generated in the latent dtype, as the pipeline
    # would, and saved as the values actually consumed.
    init_latents = generate_noise(pipe, args.height, args.width, args.seed)

    # --- vae decode input/output ---
    # The pipeline calls `vae.decode`, not the module's `forward`, so wrap the
    # method rather than registering a module hook.
    vae_in, vae_out = [], []
    _decode = pipe.vae.decode

    def decode_capture(z, *a, **kw):
        vae_in.append(z.detach().to(torch.float32).cpu().numpy())
        out = _decode(z, *a, **kw)
        s = out[0] if isinstance(out, (tuple, list)) else out
        s = s.sample if hasattr(s, "sample") else s
        vae_out.append(s.detach().to(torch.float32).cpu().numpy())
        return out

    pipe.vae.decode = decode_capture

    # --- per-step latents via the documented callback ---
    step_latents = []

    def on_step(pipe_, i, t, kw):
        step_latents.append(kw["latents"].detach().to(torch.float32).cpu().numpy())
        return kw

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        true_cfg_scale=args.guidance,
        latents=init_latents,
        callback_on_step_end=on_step,
        output_type="np",
    ).images[0]

    for h in (h1, h2):
        h.remove()
    pipe.vae.decode = _decode

    from PIL import Image
    Image.fromarray((image * 255).round().astype("uint8")).save(
        os.path.join(args.out, f"out_{args.tag}.png"))
    np_save(os.path.join(args.out, f"raw_image_{args.tag}.npy"),
            torch.from_numpy(image))
    np_save(os.path.join(args.out, f"init_noise_{args.tag}.npy"), init_latents)
    # Also under the resolution alone, for passing to `generate --init-latents`.
    np_save(os.path.join(args.out, f"init_noise_{args.height}x{args.width}.npy"), init_latents)

    # Tokenized prompt, for the Rust tokenizer check.
    templated = pipe.prompt_template_t2i.format(args.prompt or " ")
    proc = pipe.processor(text=[templated], padding=True, padding_side="left",
                          return_tensors="pt")
    np_save(os.path.join(args.out, f"prompt_input_ids_{args.tag}.npy"), proc.input_ids)
    np_save(os.path.join(args.out, f"prompt_attention_mask_{args.tag}.npy"),
            proc.attention_mask)

    # Replay set: every DiT input and output, in order.
    np_save(os.path.join(args.out, f"encoder_hidden_states_{args.tag}.npy"),
            torch.from_numpy(tx_in[0]["encoder_hidden_states"]))
    np_save(os.path.join(args.out, f"img_mask_{args.tag}.npy"),
            torch.from_numpy(tx_in[0]["img_mask"]))
    if tx_in[0]["encoder_hidden_states_mask"] is not None:
        np_save(os.path.join(args.out, f"encoder_hidden_states_mask_{args.tag}.npy"),
                torch.from_numpy(tx_in[0]["encoder_hidden_states_mask"]))
    np_save(os.path.join(args.out, f"dit_timestep_{args.tag}.npy"),
            torch.from_numpy(np.array([r["timestep"] for r in tx_in])))
    np_save(os.path.join(args.out, f"dit_inputs_{args.tag}.npy"),
            torch.from_numpy(np.stack([r["hidden_states"] for r in tx_in])))

    # Step 0 (cache "extract") returns the whole joint sequence; later steps
    # ("cached") return only the target-image tokens.
    decode = [i for i, o in enumerate(tx_out) if o.shape != tx_out[0].shape]
    np_save(os.path.join(args.out, f"dit_out_prefill_{args.tag}.npy"),
            torch.from_numpy(tx_out[0]))
    if decode:
        shapes = {tx_out[i].shape for i in decode}
        if len(shapes) != 1:
            raise ValueError(f"decode-step outputs have mixed shapes: {shapes}")
        np_save(os.path.join(args.out, f"dit_out_decode_{args.tag}.npy"),
                torch.from_numpy(np.stack([tx_out[i] for i in decode])))

    np_save(os.path.join(args.out, f"latents_{args.tag}.npy"),
            torch.from_numpy(np.stack(step_latents)))
    np_save(os.path.join(args.out, f"scheduler_sigmas_{args.tag}.npy"),
            pipe.scheduler.sigmas.detach().cpu())
    np_save(os.path.join(args.out, f"scheduler_timesteps_{args.tag}.npy"),
            pipe.scheduler.timesteps.detach().cpu())
    np_save(os.path.join(args.out, f"vae_decode_input_{args.tag}.npy"),
            torch.from_numpy(vae_in[0]))
    np_save(os.path.join(args.out, f"vae_decode_output_{args.tag}.npy"),
            torch.from_numpy(vae_out[0]))

    meta = {
        "prompt": args.prompt,
        "negative_prompt": args.negative,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "true_cfg_scale": args.guidance,
        "num_transformer_calls": len(tx_in),
        "dit_input_shapes": [list(r["hidden_states"].shape) for r in tx_in],
        "dit_output_shapes": [list(o.shape) for o in tx_out],
        "kv_cache_modes": [r["kv_cache_mode"] for r in tx_in],
        "img_shapes": tx_in[0]["img_shapes"],
        "image_shape": list(np.array(image).shape),
    }
    with open(os.path.join(args.out, f"meta_{args.tag}.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(json.dumps({k: meta[k] for k in
                      ("num_transformer_calls", "image_shape", "img_shapes")},
                     default=str))


if __name__ == "__main__":
    main()
