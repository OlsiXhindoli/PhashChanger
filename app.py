import os
import uuid
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

app = Flask(__name__)

TEMP_DIR = "/tmp"


@app.route("/")
def index():
    """Render simple upload form."""
    return render_template("index.html")


@app.route("/mutate", methods=["POST"])
def mutate():
    """Handle upload and display variant results."""
    file = request.files.get("image")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    ext = os.path.splitext(file.filename)[1] or ".png"
    orig_name = f"orig_{uuid.uuid4().hex}{ext}"
    path_in = os.path.join(TEMP_DIR, orig_name)
    file.save(path_in)

    variants = generate_variants(path_in, n=10)

    orig_hash = phash(Image.open(path_in).convert("RGB"))

    return render_template(
        "result.html",
        orig_path=orig_name,
        orig_hash=str(orig_hash),
        variants=variants,
    )


@app.route("/tmp/<path:filename>")
def temp_files(filename):
    """Serve files saved in the temporary directory."""
    return send_from_directory(TEMP_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
