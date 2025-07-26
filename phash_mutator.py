from pathlib import Path
from typing import List, Dict
import uuid
import random
import numpy as np
from PIL import Image, ImageEnhance
import imagehash
from utils import hamming


def phash(img: Image.Image) -> imagehash.ImageHash:
    """Return perceptual hash of the image."""
    return imagehash.phash(img)


_TRANSFORMS = (
    "tiny_crop",
    "brightness",
    "gaussian_noise",
    "shift_one_pixel",
)


def random_transform(img: Image.Image, op: str) -> Image.Image:
    """Return a new Image after applying one micro-edit."""
    if op == "tiny_crop":
        # crop 1-2 pixels around the border then resize back
        w, h = img.size
        left = random.randint(0, 2)
        top = random.randint(0, 2)
        right = random.randint(0, 2)
        bottom = random.randint(0, 2)
        cropped = img.crop((left, top, w - right, h - bottom))
        return cropped.resize((w, h))
    if op == "brightness":
        factor = 1 + random.uniform(-0.1, 0.1)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    if op == "gaussian_noise":
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 3, arr.shape)
        arr += noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if op == "shift_one_pixel":
        arr = np.array(img)
        dx = random.choice([-1, 1])
        dy = random.choice([-1, 1])
        arr = np.roll(arr, shift=dx, axis=1)
        arr = np.roll(arr, shift=dy, axis=0)
        return Image.fromarray(arr)
    raise ValueError(f"Unknown transform: {op}")


def mutate_once(img: Image.Image):
    """Apply one random_transform picked from _TRANSFORMS."""
    op = random.choice(_TRANSFORMS)
    return random_transform(img, op), op


def generate_variants(
    path_in: str,
    n: int = 10,
    target_bits: int = 14,
    inter_bits: int = 6,
    max_iter: int = 300,
) -> List[Dict]:
    """Return list of unique variants meeting distance constraints."""
    orig = Image.open(path_in).convert("RGB")
    h0 = phash(orig)
    variants = []
    iter_cnt = 0
    while len(variants) < n and iter_cnt < max_iter:
        iter_cnt += 1
        img_tmp = orig.copy()
        ops = []
        for _ in range(3):
            img_tmp, op = mutate_once(img_tmp)
            ops.append(op)
        h1 = phash(img_tmp)
        d0 = hamming(h0, h1)
        if d0 < target_bits:
            continue
        if any(hamming(v["phash_int"], h1) < inter_bits for v in variants):
            continue
        out_path = f"/tmp/mut_{uuid.uuid4().hex}.png"
        img_tmp.save(out_path, "PNG")
        variants.append(
            dict(
                path_out=out_path,
                phash_int=int(str(h1), 16),
                distance_to_original=d0,
                ops_history=ops,
            )
        )
    if len(variants) < n:
        raise RuntimeError(
            f"Generated only {len(variants)} variants after {iter_cnt} attempts."
        )
    return variants
