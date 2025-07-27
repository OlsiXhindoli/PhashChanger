import os
import uuid
import logging
from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template,
    send_from_directory,
)
from PIL import Image

from phash_mutator import generate_variants, phash

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception as exc:  # pragma: no cover - optional dependency
    logging.getLogger(__name__).warning("HEIF support unavailable: %s", exc)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)

app = Flask(__name__)

TEMP_DIR = "/tmp"

# Acceptable file extensions
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".pne", ".heic"}


@app.route("/")
def index():
    """Render simple upload form."""

    app.logger.info("Rendering upload form")

    return render_template("index.html")


@app.route("/mutate", methods=["POST"])
def mutate():
    """Handle upload and display variant results."""

    app.logger.info("Received mutation request")

    file = request.files.get("image")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    ext = os.path.splitext(file.filename)[1].lower() or ".png"
    if ext not in ALLOWED_EXTS:
        app.logger.warning("File extension %s not allowed", ext)
        return f"Unsupported file type: {ext}", 400
    app.logger.info("Saving original upload with extension %s", ext)
    orig_name = f"orig_{uuid.uuid4().hex}{ext}"
    path_in = os.path.join(TEMP_DIR, orig_name)
    file.save(path_in)
    app.logger.info("Saved upload to %s", path_in)

    try:
        variants = generate_variants(path_in, n=10)
    except RuntimeError as exc:
        app.logger.error("Variant generation failed: %s", exc)
        return f"Error: {exc}", 500
    app.logger.info("Generated %d variants", len(variants))

    orig_hash = phash(Image.open(path_in).convert("RGB"))
    app.logger.info("Original pHash %s", orig_hash)

    ext = os.path.splitext(file.filename)[1] or ".png"
    orig_name = f"orig_{uuid.uuid4().hex}{ext}"
    path_in = os.path.join(TEMP_DIR, orig_name)
    file.save(path_in)


    return render_template(
        "result.html",
        orig_path=orig_name,
        orig_hash=str(orig_hash),
        variants=variants,
    )


@app.route("/tmp/<path:filename>")
def temp_files(filename):
    """Serve files saved in the temporary directory."""


    app.logger.debug("Serving file %s", filename)

    return send_from_directory(TEMP_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
