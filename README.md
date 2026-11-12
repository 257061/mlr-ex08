# MLR exercise 08 — leaf classifier in the browser

**Live page: <https://257061.github.io/mlr-ex08/>**

Point a phone's rear camera at a leaf and the page classifies it **live, on the
device** — into `apple`, `cherry`, `chestnut` or `maple`. Nothing is uploaded:
frames never leave the browser, the ONNX model runs locally through
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) (WASM, loaded from a CDN).

Open the link on a phone (or scan the QR code at the bottom of the page), press
**Start rear camera**, allow camera access. If the camera is unavailable, **Upload
photo** / **Take photo** classifies a still image through the same code path.

## What it runs

`index.html` is the whole app — one file, plain JavaScript, no build step.

Preprocessing follows `model_info.json` exactly:

1. crop the **central square** of the current frame,
2. resize it to `input_size` × `input_size` (**96 × 96**),
3. scale the RGB bytes to **0–1** (`/255`),
4. standardise each channel with `mean = [0.485, 0.456, 0.406]`,
   `std = [0.229, 0.224, 0.225]`,
5. feed the tensor as **NCHW**, `1 × 3 × 96 × 96`, float32, input name `image`.

The graph's output is `logits` (`1 × 4`), so **softmax is applied in JavaScript**
before the four probabilities are shown. Bars update on every processed video
frame; measured in headless Chrome on a desktop CPU this was **~50 ms per
inference, 10–15 inferences/s** (a phone will be slower, and the page shows its
own measured rate next to the bars).

## Files

| File | Purpose |
|---|---|
| `index.html` | the app (camera, upload fallback, live probabilities, QR code) |
| `leaf_model.onnx` | the trained leaf classifier (MobileNet-style, opset 20) |
| `model_info.json` | class names, input size, per-channel mean/std |
| `check_image.jpg` | 96 × 96 self-test image |
| `check_expected.json` | reference probabilities for `check_image.jpg` |

## Self-check

The page carries its own regression check, which runs `check_image.jpg` through
the page's **own** preprocessing and inference path and compares against
`check_expected.json`. In a browser console on the live page:

```js
await window.leafSelfTest()
// {ok: true, max_abs_diff: 6.4e-8, prob_sum: 1.0,
//  rows: [{cls: 'apple', got: 0.9990653950589254, expected: 0.9990654587745667, …}, …]}
```

Verified against a desktop Chrome and a Python/onnxruntime reference run:
the largest absolute deviation from `check_expected.json` is **6.4e-8**, and the
same pipeline classifies live camera frames correctly.

## iPhone / Safari notes

* `getUserMedia` needs a secure context — GitHub Pages serves HTTPS, so it works.
* The `<video>` element carries `playsinline`/`webkit-playsinline` + `muted`, which
  is what makes an inline (not full-screen) preview play in iOS Safari.
* GitHub Pages sends no COOP/COEP headers, so `SharedArrayBuffer` is unavailable:
  the session is pinned to a **single WASM thread** (`ort.env.wasm.numThreads = 1`),
  which is also the reliable setting on iOS Safari.
* Frames are taken with `requestVideoFrameCallback` where available, otherwise a
  timer; classification pauses while the tab is hidden.
* The square preview is `object-fit: cover`, so **what you see in the square is
  exactly the region the model receives**.

## QR code

The QR code at the bottom of the page is generated locally (the `qrcode` Python
library) and inlined as SVG path data in `index.html` — no third-party QR image
service is involved. It encodes `https://257061.github.io/mlr-ex08/`, the same
address as the clickable link next to it.

## Deployed by

GitHub Pages, from the `main` branch (root of the repository).
