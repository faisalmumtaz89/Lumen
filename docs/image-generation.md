# Image generation

`lumen-server` can serve Qwen-Image-2.1 text-to-image next to a text model on one
CUDA device. The endpoint is `POST /v1/images/generations` and is compiled in with the
`image` Cargo feature.

## Requirements

- NVIDIA CUDA, compute capability 8.0+ (the CPU path exists for reference checks only
  and takes minutes per image).
- Device memory: the text encoder's language tower (14.1 GiB of BF16 weights), the
  transformer (13.3 GiB) and the VAE decoder (about 1 GiB). Loaded one at a time, a
  generation's device memory peaks at 15.5 GiB for a 1024×1024 image and 21.0 GiB for a
  2048×2048 one (the VAE decode). The server refuses to start on a device with less than
  21.0 GiB in total, naming both amounts. How the components are held depends on where
  the text model runs; see [Sharing the device with a text model](#sharing-the-device-with-a-text-model).

## Convert the checkpoint

The pipeline reads three `.lbi` containers converted from the Qwen-Image-2.1 checkpoint
directory (the one holding `transformer/`, `vae/`, `text_encoder/` and `processor/`):

```sh
cargo run --release -p lumen-image --bin lbi-convert -- /path/to/Qwen-Image-2.1 /path/to/lbi
```

This writes `transformer.lbi`, `vae.lbi` and `text_encoder.lbi`. The checkpoint's
`processor/vocab.json`, `processor/merges.txt` and `processor/added_tokens.json` are read
directly at run time.

## Run

```sh
cargo build --release -p lumen-server --features bin,cuda,image
LUMEN_IMAGE_LBI=/path/to/lbi LUMEN_IMAGE_CKPT=/path/to/Qwen-Image-2.1 \
  lumen-server qwen3.8-27b:q4_0 --backend cuda
```

Both variables must be set together; the server checks every file it will need at
startup (tensor shapes, bf16 storage for the weights the CUDA path multiplies, the VAE
configuration, the tokenizer's template markers) and refuses to start otherwise. `LUMEN_IMAGE_MODEL_ID` renames the served image model
(default `Qwen-Image-2.1`); `LUMEN_IMAGE_DEVICE` selects `cuda` (default; `gpu` is an
alias) or `cpu`.
Every variable is described in [environment-variables.md](environment-variables.md).

## Request

```sh
curl http://localhost:8000/v1/images/generations \
  -H 'content-type: application/json' \
  -d '{"prompt":"A red apple on a wooden table, studio lighting","size":"1024x1024","num_inference_steps":40,"seed":7}'
```

| Field | Default | Limits |
|---|---|---|
| `model` | the served image model id | must match `LUMEN_IMAGE_MODEL_ID` when given |
| `prompt` | required | up to 8 KiB |
| `size` | `1024x1024` | `WxH`, each side 32–2048; sides round down to a multiple of 32 |
| `num_inference_steps` | `40` | 1–200 |
| `seed` | `42` | any `u64` |
| `true_cfg_scale` | `1.0` | must be `1.0`: one conditional pass per step, no negative prompt |
| `n` | `1` | must be `1`: one image per request |
| `response_format` | `b64_json` | `b64_json` or `url` (a `data:` URL carrying the same bytes) |
| `output_format` | `png` | `png` only |

The response is `{"created": <unix-seconds>, "data": [{"b64_json": "<PNG>"}]}` (or
`{"url": "data:image/png;base64,…"}`). The same request with the same seed produces the
same bytes.

## Sharing the device with a text model

Generations run one at a time. When the text model is served on the same CUDA device, a
generation evicts it, loads the three image components one at a time, and restores the
text model before the image is returned; text requests made meanwhile get a retryable
`503` with the message `the model is evicted for an image generation; retry shortly`.

When the text model runs on the CPU or on another device, the image endpoint keeps the
transformer and the VAE loaded on CUDA device 0 between generations (about 14.4 GiB,
held while the server runs), and only the text encoder loads for each image; the
server fails to start if that device cannot hold them. Prompt encoding and image
decoding run beside the resident transformer when they fit. When one runs out of device
memory there — a 2048×2048 decode on a 32 GiB card, a prompt of thousands of tokens, or
any prompt on a card much smaller than 32 GiB — the transformer is released and the step
retried: after an encoding it is loaded again for the denoising steps of the same
generation, after a decode by the next generation. Later work at least that large
releases it up front. Images are identical either way. Any generation that runs out pays
for its failed attempt; work at or above a remembered size costs what the evicting mode
costs. A single
out-of-memory event caused by another process on the device lowers that size threshold
for the rest of the server's life.

On CUDA the server opens the three `.lbi` files once at startup and keeps them mapped, so
their pages count toward its resident memory (page cache the system can reclaim).
`lbi-convert` writes each file beside its destination and renames it into place, so
re-converting while the server runs leaves it serving the files it started with; restart
it to load the new ones. Overwriting a file in place (for example with `cp`) while the
server runs can crash it.

A request whose client disconnects is dropped: if it was still queued nothing
is evicted, and if its generation was running it stops at the next denoising step and the
text model is restored then (the disconnect is seen when the connection carries no further
pipelined request behind the image request). Stopping the server (Ctrl-C) stops a generation that is still
denoising the same way and answers it with `503`; one already decoding its image completes.

Errors are the same JSON shape as the text endpoints: validation failures are `400` with
`param` naming the field, a generation failure is `500`.
