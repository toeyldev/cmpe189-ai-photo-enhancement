r"""
Flask web app for CMPE189 photo enhancement (Check-in 4).

What this file does (high level):
  1. Serves a single-page UI from templates/index.html (route GET /).
  2. Accepts multipart uploads on POST /enhance: an image file + model choice.
  3. Decodes the image to RGB, runs either DnCNN or SwinIR inference (PyTorch),
     then returns the result as a downloadable PNG.

Why DnCNN and SwinIR live in src/:
  - DnCNN architecture + weights helper: src/dncnn_pytorch.py, src/dncnn_weights.py
    (same checkpoint pattern as src/evaluate_model.py, without importing pandas).
  - SwinIR: src/swinir_pytorch.py (clones upstream SwinIR repo on first use, loads weights).

Run from the project root (folder that contains this file):
    pip install -r requirements.txt
    python app.py or .\.venv\Scripts\python app.py if you want to use GPU 

Then open http://127.0.0.1:5000 (or set HOST / PORT env vars — see __main__ block).
"""

from __future__ import annotations  # Enables postponed evaluation of type annotations (helps with forward refs)

import io  # BytesIO for returning encoded PNG via send_file
import os  # Environment variables (HOST/PORT) + basic OS interaction
import sys # Used for importing files from the src directory
from pathlib import Path # Used for handling filesystem paths

import cv2 # Used for reading, resizing and saving images
import numpy as np # Used for array/image math
import torch # Used for building and running neural networks
import time  # perf_counter timing used for UI metrics headers
from flask import Flask, jsonify, make_response, render_template, request, send_file # Used for serving the web app
from skimage.metrics import peak_signal_noise_ratio as psnr # Used for PSNR metric
from skimage.metrics import structural_similarity as ssim # Used for SSIM metric


# ---------------------------------------------------------------------------
# Path setup: model code lives under src/, but we run Flask from repo root.
# Insert src at the front of sys.path so "import dncnn_pytorch" works the same
# way as in run_pipeline.py / notebooks.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dncnn_pytorch import DnCNN  # noqa: E402
from dncnn_weights import ensure_weights  # noqa: E402

# File types accepted by the UI.
# NOTE: This is an extension check only; we still attempt to decode the bytes to validate it's an image.
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}

# Lazy singletons:
# Loading weights into GPU/CPU memory is expensive, so we keep one instance per process.
# This makes repeat requests fast (they typically only pay inference time).
_dncnn_cache: tuple | None = None
_swinir_cache: tuple | None = None

app = Flask(__name__)
# Reject unreasonably large uploads early (32 MiB).
# This is a safety/UX guardrail for the dev server; tune higher if you need bigger images.
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


def allowed_file(name: str) -> bool:
    """Return True if the original filename has an extension we support."""
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_upload_to_rgb(file_storage) -> np.ndarray:
    """
    Turn the browser upload into an H×W×3 uint8 RGB image (NumPy).

    Flow:
      1. Read raw bytes from the Werkzeug FileStorage object.
      2. Wrap bytes in a 1-D uint8 array (what OpenCV expects for imdecode).
      3. cv2.imdecode decodes PNG/JPEG/etc. into BGR (OpenCV's native order).
      4. Convert BGR → RGB so PyTorch models match the same convention as evaluate_model.py.

    Important:
      - Reading from file_storage consumes the stream. Call this once per uploaded file.
      - Any non-image or corrupted image will fail decode and return a user-friendly error.
    """
    raw = file_storage.read()
    if not raw:
        raise ValueError("Empty file.")
    # frombuffer: zero-copy view over the upload bytes (no extra JPEG decode in NumPy).
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image. Use PNG, JPEG, or similar.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_dncnn(model, device, rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Run one forward pass of DnCNN on a single RGB image.

    Tensor layout:
      NumPy (H, W, 3) uint8
        → float32 [0, 1]
        → (1, 3, H, W) on device
      Model output is also [0, 1] range per evaluate_model.py; clip back to uint8.
    """
    img_norm = rgb_uint8.astype(np.float32) / 255.0
    t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(t)
    denoised = out.squeeze().cpu().permute(1, 2, 0).numpy()
    return np.clip(denoised * 255.0, 0, 255).astype(np.uint8)


def load_dncnn_model(weights_path=None):
    """
    Build DnCNN and load pretrained weights (same logic as evaluate_model.load_model).

    We duplicate the load path here so the web app does not import evaluate_model
    at startup (evaluate_model pulls pandas/skimage — not needed to serve the UI).

    Returns:
      (model, device) where device is either "cuda" or "cpu".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = ensure_weights(weights_path)
    state_dict = torch.load(path, map_location=device)
    # KAIR checkpoints sometimes nest weights under a "params" key.
    if "params" in state_dict:
        state_dict = state_dict["params"]
    model = DnCNN(channels=3, num_of_layers=20, features=64)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, device


def get_dncnn():
    """Return cached (model, device) for DnCNN, loading on first use."""
    global _dncnn_cache
    if _dncnn_cache is None:
        _dncnn_cache = load_dncnn_model(weights_path=None)
    return _dncnn_cache


def get_swinir():
    """
    Return cached (model, device) for SwinIR, loading on first use.

    Import is deferred: swinir_pytorch may clone a Git repo and download weights,
    which is slow and only needed when someone actually picks SwinIR.
    """
    global _swinir_cache
    if _swinir_cache is None:
        from swinir_pytorch import load_swinir_model  # noqa: PLC0415

        _swinir_cache = load_swinir_model(weights_path=None)
    return _swinir_cache


@app.route("/")
def index():
    """Serve the single-page UI (templates/index.html)."""
    return render_template("index.html")


@app.route("/enhance", methods=["POST"])
def enhance():
    """
    Main API: multipart form with fields:
      - "image" (file) and "model" ("dncnn" | "swinir")
      - optional "ground_truth" (file): if provided, PSNR/SSIM are computed against it

    Success response:
      - Body: PNG bytes as an attachment (browser download)
      - Headers: timing/device/optional quality metrics (read by templates/index.html)

    Failure response:
      - JSON {"error": "..."} with 4xx/5xx
      - The front-end reads this and shows the message under the button.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in form."}), 400
    f = request.files["image"]
    if not f or not f.filename:
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    model_name = (request.form.get("model") or "dncnn").strip().lower()
    if model_name not in ("dncnn", "swinir"):
        return jsonify({"error": "Model must be 'dncnn' or 'swinir'."}), 400

    gt_file = request.files.get("ground_truth")

    # Total wall-clock time for the full request (decode + load + inference + encode + overhead).
    t0_total = time.perf_counter()

    try:
        # Decode input image bytes -> RGB array for model inference.
        t0_decode = time.perf_counter()
        rgb = decode_upload_to_rgb(f)
        decode_s = time.perf_counter() - t0_decode
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    gt_rgb = None
    if gt_file and gt_file.filename:
        try:
            # If ground truth is provided, decode it now so we can compute PSNR/SSIM later.
            gt_rgb = decode_upload_to_rgb(gt_file)
        except ValueError as e:
            return jsonify({"error": f"Ground truth image error: {e}"}), 400

    h, w = rgb.shape[:2]

    try:
        # Track whether this request incurred a cold-start model load.
        # If the model was already cached, this remains ~0.
        model_load_s = 0.0
        tile_stats = None

        if model_name == "dncnn":
            global _dncnn_cache
            if _dncnn_cache is None:
                t0_load = time.perf_counter()
            model, device = get_dncnn()
            if model_load_s == 0.0 and "_dncnn_cache" in globals() and _dncnn_cache is not None:
                # if we hit the cold-start path above, measure load time now
                if "t0_load" in locals():
                    model_load_s = time.perf_counter() - t0_load

            if device.type == "cuda":
                # These synchronizations make the timing/VRAM stats more accurate on GPU.
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            t0_inf = time.perf_counter()
            out_rgb = run_dncnn(model, device, rgb)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_s = time.perf_counter() - t0_inf
        else:
            # SwinIR uses tiled inference inside swinir_inference (large images, VRAM).
            from swinir_pytorch import swinir_inference  # noqa: PLC0415

            global _swinir_cache
            if _swinir_cache is None:
                t0_load = time.perf_counter()
            model, device = get_swinir()
            if model_load_s == 0.0 and "_swinir_cache" in globals() and _swinir_cache is not None:
                if "t0_load" in locals():
                    model_load_s = time.perf_counter() - t0_load

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            t0_inf = time.perf_counter()
            out_rgb, tile_stats = swinir_inference(model, device, rgb, return_stats=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            infer_s = time.perf_counter() - t0_inf

        # Optional quality metrics:
        # Only compute when GT exists AND dimensions match (same H×W×C).
        psnr_out = ssim_out = psnr_in = ssim_in = None
        if gt_rgb is not None:
            if gt_rgb.shape != out_rgb.shape or gt_rgb.shape != rgb.shape:
                return jsonify({"error": "Ground truth image must match input image dimensions (same width/height/channels)."}), 400
            psnr_in = float(psnr(gt_rgb, rgb))
            ssim_in = float(ssim(gt_rgb, rgb, channel_axis=2))
            psnr_out = float(psnr(gt_rgb, out_rgb))
            ssim_out = float(ssim(gt_rgb, out_rgb, channel_axis=2))

        t_total = time.perf_counter() - t0_total
    except Exception as e:
        # Catch-all so the UI can show a readable error instead of a Flask HTML traceback.
        return jsonify({"error": f"Inference failed: {e!s}"}), 500

    # GPU memory stats (if applicable)
    vram_peak_mb = None
    device_str = "cpu"
    if "device" in locals():
        device_str = str(device)
        if getattr(device, "type", "") == "cuda":
            vram_peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    # OpenCV encoders expect BGR; we kept tensors in RGB throughout inference.
    t0_encode = time.perf_counter()
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", out_bgr)
    if not ok:
        return jsonify({"error": "Failed to encode output image."}), 500
    encode_s = time.perf_counter() - t0_encode

    # Download name is derived from original filename + model choice.
    base = Path(f.filename).stem or "image"
    download_name = f"{base}_enhanced_{model_name}.png"

    response = make_response(send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/png",
        as_attachment=True,
        download_name=download_name,
    ))

    # Simple total time (for backwards compatibility with existing UI)
    response.headers["X-Processing-Time"] = f"{t_total:.2f}"

    # Detailed metrics for the UI.
    # Front-end reads these headers in templates/index.html and renders the "Metrics" table.
    response.headers["X-Device"] = device_str
    response.headers["X-Image-Size"] = f"{w}x{h}"
    response.headers["X-Time-Decode"] = f"{decode_s:.4f}"
    response.headers["X-Time-Model-Load"] = f"{model_load_s:.4f}"
    response.headers["X-Time-Infer"] = f"{infer_s:.4f}"
    response.headers["X-Time-Encode"] = f"{encode_s:.4f}"
    response.headers["X-Time-Total"] = f"{t_total:.4f}"
    if vram_peak_mb is not None:
        response.headers["X-VRAM-Peak-MB"] = f"{vram_peak_mb:.1f}"

    # Tiling stats are only present for SwinIR when tile-based inference is used.
    if tile_stats:
        response.headers["X-Tile-Size"] = str(tile_stats.get("tile_size"))
        response.headers["X-Tile-Overlap"] = str(tile_stats.get("tile_overlap"))
        response.headers["X-Tiles-Processed"] = str(tile_stats.get("tiles_processed"))

    # Quality stats are only present when GT is provided.
    if psnr_out is not None:
        response.headers["X-PSNR-In"] = f"{psnr_in:.3f}"
        response.headers["X-SSIM-In"] = f"{ssim_in:.5f}"
        response.headers["X-PSNR-Out"] = f"{psnr_out:.3f}"
        response.headers["X-SSIM-Out"] = f"{ssim_out:.5f}"
    return response


if __name__ == "__main__":
    # Optional overrides for local dev (e.g. PORT=8080 python app.py).
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    # debug=True: auto-reload on code changes; do not enable on a public server.
    app.run(host=host, port=port, debug=True)
