#!/usr/bin/env python3
"""
phash_metadata_generator.py

This script prompts you to select an input image (PNG, HEIC, JPEG, JPG, etc.) via a file-dialog,
then generates a fixed set of visually similar variants with different pHashes and randomized EXIF metadata.
It applies two subtle transforms per variant to preserve recognizability while changing the pHash.

You can adjust NUM_VARIANTS to control how many unique images to produce.
Best-practice: generating around 10–15 variants strikes a good balance of diversity vs. processing time.
"""
import os
import io
import random
import string
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import imagehash
import piexif

# Where to save variants (modify as needed)
OUTPUT_DIR = "/Users/olsixhindoli/Desktop/output_variants"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many unique variants to generate
NUM_VARIANTS = 12

# Optional HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def tilt(img, max_angle=1.0):
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def jitter(img, max_shift=2):
    dx = random.uniform(-max_shift, max_shift)
    dy = random.uniform(-max_shift, max_shift)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def crop_and_zoom(img, max_crop=0.03):
    h, w = img.shape[:2]
    top = int(random.uniform(0, max_crop)*h)
    bot = int(random.uniform(0, max_crop)*h)
    left = int(random.uniform(0, max_crop)*w)
    right = int(random.uniform(0, max_crop)*w)
    cropped = img[top:h-bot, left:w-right]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def color_jitter(img, b=0.02, c=0.02, s=0.02):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if random.random() < 0.8:
        pil = ImageEnhance.Brightness(pil).enhance(1 + random.uniform(-b, b))
        pil = ImageEnhance.Contrast(pil).enhance(1 + random.uniform(-c, c))
        pil = ImageEnhance.Color(pil).enhance(1 + random.uniform(-s, s))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def blur_edges(img, radius=20):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for cx, cy in [(0,0),(w,0),(0,h),(w,h)]:
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=radius/4)
    return np.where(mask[..., None]==255, blurred, img)


def film_grain(img, intensity=0.005):
    noise = np.random.randn(*img.shape)*255*intensity
    out = img.astype(np.float32)+noise
    return np.clip(out,0,255).astype(np.uint8)


def compute_phash(pil_img):
    return imagehash.phash(pil_img)


def inject_metadata(jpeg_bytes):
    make = random.choice(['Canon','Nikon','Sony','Huawei','Apple','Samsung'])
    model = f"{make}_{random_string(4)}"
    software = f"phashgen_{random_string(6)}"
    dt = datetime.now().strftime('%Y:%m:%d %H:%M:%S')
    exif_dict = {'0th':{}, 'Exif':{}, 'GPS':{}, '1st':{}, 'Interop':{}}
    exif_dict['0th'][piexif.ImageIFD.Make]=make
    exif_dict['0th'][piexif.ImageIFD.Model]=model
    exif_dict['0th'][piexif.ImageIFD.Software]=software
    exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal]=dt
    exif_bytes = piexif.dump(exif_dict)
    pil = Image.open(io.BytesIO(jpeg_bytes))
    out = io.BytesIO()
    pil.save(out, format='JPEG', exif=exif_bytes)
    return out.getvalue(), (make, model, software, dt)


def generate_variants(input_path):
    pil_orig = Image.open(input_path)
    img0 = cv2.cvtColor(np.array(pil_orig), cv2.COLOR_RGB2BGR)
    base_hash = compute_phash(pil_orig)
    print(f"Original pHash: {base_hash}")

    seen_hashes = {str(base_hash)}
    seen_meta = set()
    saved = 0
    attempts = 0
    max_attempts = NUM_VARIANTS * 5

    transforms = [tilt, jitter, crop_and_zoom, color_jitter, blur_edges, film_grain]

    while saved < NUM_VARIANTS and attempts < max_attempts:
        attempts += 1
        img = img0.copy()
        for fn in random.sample(transforms, k=2): img = fn(img)

        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(90,100)])
        jpeg = buf.tobytes()
        jpeg_exif, meta = inject_metadata(jpeg)
        var_pil = Image.open(io.BytesIO(jpeg_exif))
        hsh = compute_phash(var_pil)

        if str(hsh) in seen_hashes or meta in seen_meta:
            continue
        seen_hashes.add(str(hsh)); seen_meta.add(meta)
        saved += 1
        fname = f"variant_{saved}_{hsh}.jpg"
        with open(os.path.join(OUTPUT_DIR, fname),'wb') as f: f.write(jpeg_exif)
        print(f"Saved {fname}: pHash={hsh}, meta={meta}")

    if saved < NUM_VARIANTS:
        print(f"Warning: only generated {saved}/{NUM_VARIANTS} after {attempts} attempts.")
    else:
        print(f"Generated {saved} variants in {attempts} attempts.")


def main():
    # GUI file-picker, no argparse needed
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Images","*.png *.jpg *.jpeg *.heic *.bmp *.tiff")]
    )
    if not path:
        print("No file selected, exiting.")
        return
    generate_variants(path)

if __name__=='__main__':
    main()
