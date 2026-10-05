# Image generation

`lumen-server` serves Qwen-Image-2.1 text-to-image as an image-only server: started with
the image model's name, or with the image variables, and no text model, it serves
`POST /v1/images/generations` and a `GET /v1/models` that lists the image model, and no
text routes. A text model runs in a separate `lumen-server` process; one process does not
serve both. The endpoint is compiled in with the `image` Cargo feature. The Linux/CUDA
release tarball and Docker image ship a `lumen-server` built with it and the `lbi-convert`
converter.

```sh
lumen pull qwen-image
lumen-server qwen-image
```

`lumen pull qwen-image` downloads the checkpoint from Hugging Face (33 GB, pinned to one
commit; every file is checked against its pinned size and SHA-256 and refused otherwise),
converts it into the three `.lbi` containers under `~/.cache/lumen/qwen-image-2-1/lbi/`,
and removes the downloaded weights, keeping the tokenizer files under `processor/`. A pull
that stops continues where it stopped on the next run; a complete one reports the cache.
It refuses on a `lumen` built without CUDA, since the model runs on CUDA only.
`lumen-server qwen-image` serves what the pull cached; `LUMEN_IMAGE_MODEL_ID`,
`LUMEN_IMAGE_DEVICE` and `LUMEN_IMAGE_PIN_TEXT_ENCODER` apply to it as below, while
`LUMEN_IMAGE_LBI` and `LUMEN_IMAGE_CKPT` are for a checkpoint converted by hand and are
refused beside the name. The rest of this page covers that by-hand path and the endpoint.

## Requirements

- NVIDIA CUDA, compute capability 8.0+ (the CPU path exists for reference checks only
  and takes minutes per image).
- Device memory: the text encoder's language tower (12.9 GiB of BF16 weights), the
  transformer (13.3 GiB) and the VAE decoder (about 1 GiB). Loaded one at a time, a
  generation's device memory peaks at 15.3 GiB for a 1024×1024 image and 21.0 GiB for a
  2048×2048 one (the VAE decode). The server refuses to start on a device with less than
  21.0 GiB in total, naming both amounts. The transformer and the VAE normally stay loaded
  between generations, and the server uses spare room beside them: on a 32 GiB card,
  1024×1024 generations take up to about 30 GiB. See [Device memory](#device-memory).

## Convert the checkpoint

The pipeline reads three `.lbi` containers converted from the Qwen-Image-2.1 checkpoint
directory (the one holding `transformer/`, `vae/`, `text_encoder/` and `processor/`):

```sh
lbi-convert /path/to/Qwen-Image-2.1 /path/to/lbi
# from source: cargo run --release -p lumen-image --bin lbi-convert -- <same arguments>
```

This writes `transformer.lbi`, `vae.lbi` and `text_encoder.lbi`. The checkpoint's
`processor/vocab.json`, `processor/merges.txt` and `processor/added_tokens.json` are read
directly at run time.

## Run

```sh
# from source: cargo build --release -p lumen-server --features bin,cuda,image
LUMEN_IMAGE_LBI=/path/to/lbi LUMEN_IMAGE_CKPT=/path/to/Qwen-Image-2.1 lumen-server --port 8000
```

A model passed as well is refused at startup: run the text model as its own `lumen-server`
on another port, and on a device that cannot hold both, start one server at a time.

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
| `size` | `1024x1024` | `WxH`, each side 32–4096; sides round down to a multiple of 32 |
| `num_inference_steps` | `40` | 1–200 |
| `seed` | `42` | any `u64` |
| `true_cfg_scale` | `1.0` | must be `1.0`: one conditional pass per step, no negative prompt |
| `n` | `1` | must be `1`: one image per request |
| `response_format` | `b64_json` | `b64_json` or `url` (a `data:` URL carrying the same bytes) |
| `output_format` | `png` | `png` only |

The response is `{"created": <unix-seconds>, "data": [{"b64_json": "<PNG>"}]}` (or
`{"url": "data:image/png;base64,…"}`). The same request with the same seed produces the
same bytes.

A request that sends `Accept: image/png` receives the PNG itself (`Content-Type: image/png`)
instead, so curl alone can save it:

```sh
curl -fsS http://localhost:8000/v1/images/generations \
  -H 'content-type: application/json' -H 'accept: image/png' \
  -d '{"prompt":"A red apple on a wooden table"}' -o apple.png
```

`image/png` is weighed against `application/json` as HTTP content negotiation defines it;
with no `Accept` header, or `*/*`, the response is JSON.

## Device memory

Generations run one at a time. On CUDA the server loads the transformer and the VAE on
device 0 at startup and keeps them there between generations (about 14.4 GiB), except
when the transformer has to make room as described below, and the text encoder loads for
each image; the server fails to start if that device cannot hold them. The text encoder's first layers stay
loaded between images when there is room. The first image of a size that runs with the
transformer loaded throughout keeps none, and the server records how much device memory
that size's denoising and decoding need; later images of that size, with the transformer
loaded and a prompt no longer than the measured one, keep as many layers as leave that
much free plus 512 MiB, so each loads only the rest. A denoising or decoding step that
still runs out of memory drops the kept layers and that size's record and runs again. On an RTX 5090,
repeated 1024×1024 images settle with about 10 GiB of the 12.9 GiB kept, and the
encoder's upload drops from 12.9 GiB to 2.9 GiB. Prompt encoding and image
decoding run beside the resident transformer when they fit. When one runs out of device
memory there — a 2048×2048 decode on a 32 GiB card, or any prompt on a card much smaller
than 32 GiB — the transformer is released and the step
retried, along with any text-encoder layers kept beside it. A decode that still does not
fit on its own, as at 3840×2176 on a 32 GiB card, is split into horizontal bands, twice
as many at each retry down to bands of 512 image rows, and the count is remembered for
the size; every band is decoded with enough neighbouring rows that the image is the same
as a one-pass decode, bit for bit. After an encoding the transformer is
loaded again for the denoising steps of the same generation, after a decode by the next
generation. Later work at least that large
releases it up front. Images are identical either way. Any generation that runs out pays
for its failed attempt. A single
out-of-memory event caused by another process on the device, if dropping the kept text
layers does not clear it and the transformer has to be released, lowers that size
threshold for the rest of the server's life.

On CUDA the server opens the three `.lbi` files once at startup and keeps them mapped, so
their pages count toward its resident memory (page cache the system can reclaim).
`lbi-convert` writes each file beside its destination and renames it into place, so
re-converting while the server runs leaves it serving the files it started with; restart
it to load the new ones. Overwriting a file in place (for example with `cp`) while the
server runs can crash it.

`LUMEN_IMAGE_PIN_TEXT_ENCODER=1` (CUDA only) makes the server copy the text encoder's
weights into page-locked host memory at startup (12.9 GiB) and keep them for its
lifetime; the text encoder loads for every image, and loads from page-locked memory
faster. Page-locked memory is not bounded by memlock or cgroup memory limits, so set it
only on a host that can spare that memory beside everything else it runs; the server
fails to start if the copy cannot be made.

A request whose client disconnects is dropped: if it was still queued it never runs, and
if its generation was running it stops at the next denoising step (the disconnect is seen
when the connection carries no further pipelined request behind the image request). Stopping the server (Ctrl-C) stops a generation that is still
denoising the same way and answers it with `503`; one already decoding its image completes.

Errors are the same JSON shape as the text endpoints: validation failures are `400` with
`param` naming the field, a generation failure is `500`.
