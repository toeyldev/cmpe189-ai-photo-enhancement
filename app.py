"""
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
    python app.py

Then open http://127.0.0.1:5000 (or set HOST / PORT env vars — see __main__ block).
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file

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

# File types we accept in the upload field (extension check only — not a security boundary).
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}

# Lazy singletons: loading weights into GPU/CPU memory is slow; keep one instance
# per process for each model so repeat requests only pay inference cost.
_dncnn_cache: tuple | None = None
_swinir_cache: tuple | None = None

app = Flask(__name__)
# Reject unreasonably large uploads early (32 MiB); tune if you need bigger images.
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
    """Serve the upload page (templates/index.html)."""
    return render_template("index.html")


@app.route("/enhance", methods=["POST"])
def enhance():
    """
    Main API: multipart form with fields "image" (file) and "model" ("dncnn" | "swinir").

    Success: PNG bytes as attachment (browser download).
    Failure: JSON {"error": "..."} with 4xx/5xx — the front-end reads this to show red text.
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

    try:
        rgb = decode_upload_to_rgb(f)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        if model_name == "dncnn":
            model, device = get_dncnn()
            out_rgb = run_dncnn(model, device, rgb)
        else:
            # SwinIR uses tiled inference inside swinir_inference (large images, VRAM).
            from swinir_pytorch import swinir_inference  # noqa: PLC0415

            model, device = get_swinir()
            out_rgb = swinir_inference(model, device, rgb)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e!s}"}), 500

    # OpenCV encoders expect BGR; we kept tensors in RGB throughout inference.
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", out_bgr)
    if not ok:
        return jsonify({"error": "Failed to encode output image."}), 500

    base = Path(f.filename).stem or "image"
    download_name = f"{base}_enhanced_{model_name}.png"
    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/png",
        as_attachment=True,
        download_name=download_name,
    )


if __name__ == "__main__":
    # Optional overrides for local dev (e.g. PORT=8080 python app.py).
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    # debug=True: auto-reload on code changes; do not enable on a public server.
    app.run(host=host, port=port, debug=True)
