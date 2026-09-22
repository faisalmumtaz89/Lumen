# Reference dumps

The oracle-backed `*-check` binaries in this crate (`dit-check`, `dit-check-gpu`, `e2e-check`,
`text-check`, `text-check-gpu`, `tokenizer-check`, `vae-check`, `vae-check-gpu`,
`vae-input-check`) compare Lumen's components against tensors captured from the reference
implementation of Qwen-Image-2.1 (the `diffusers` pipeline at the commit pinned in
`requirements.lock`). Each reads `<oracle-dir>/<name>_<tag>.npy`; the usual arguments are
`<lbi-dir> <oracle-dir> [tag]`
(`tokenizer-check` takes the checkpoint directory instead of the `.lbi` directory,
`vae-input-check` takes only `<oracle-dir> [tag]`, `text-check-gpu` accepts a trailing
`[drop_idx]`), and `tag` defaults to `smoke`.

Produce a dump with the reference pipeline on a CUDA machine:

```sh
python3 -m venv venv && venv/bin/pip install -r requirements.lock
venv/bin/python oracle_dump.py --model /path/to/Qwen-Image-2.1 --out /path/to/oracle \
  --prompt "A red apple on a wooden table, studio lighting" --seed 42 \
  --height 1024 --width 1024 --steps 40 --tag smoke
```

`oracle_dump.py` writes, per tag: `prompt_input_ids`, `prompt_attention_mask`,
`encoder_hidden_states`, `encoder_hidden_states_mask`, `img_mask`, `dit_timestep`,
`dit_inputs`, `dit_out_prefill`, `dit_out_decode`, `scheduler_sigmas`, `scheduler_timesteps`,
`latents`, `vae_decode_input`, `vae_decode_output`, `raw_image`, `out_<tag>.png` and `init_noise` (also as
`init_noise_<H>x<W>.npy`, which `generate --init-latents` accepts), plus `meta_<tag>.json`
(prompt, seed, size, steps and tensor shapes) and, for `--tag main`, `environment.json`
(checkpoint hashes and library versions). `oracle_common.py` holds the shared loader.
