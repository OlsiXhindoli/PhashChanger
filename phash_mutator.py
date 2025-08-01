from pathlib import Path
from typing import List, Dict, Union
import uuid
import random
import logging
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import imagehash
from utils import hamming

logger = logging.getLogger(__name__)

def phash(img: Image.Image) -> imagehash.ImageHash:
    """Return perceptual hash of the image."""
    return imagehash.phash(img)


_TRANSFORMS = (
    "tiny_crop",
    "brightness",
    "contrast",
    "color",
    "sharpness",
    "gaussian_noise",
    "shift_one_pixel",
    "rotate_small"
)


def random_transform(img: Image.Image, op: str) -> Image.Image:
    """Return a new Image after applying one micro-edit."""
    logger.debug("Applying transform: %s", op)
    if op == "tiny_crop":
        # crop up to 8% from each side then resize back
        w, h = img.size
        max_crop_w = max(1, int(w * 0.08))
        max_crop_h = max(1, int(h * 0.08))
        left = random.randint(0, max_crop_w)
        top = random.randint(0, max_crop_h)
        right = random.randint(0, max_crop_w)
        bottom = random.randint(0, max_crop_h)
        cropped = img.crop((left, top, w - right, h - bottom))
        return cropped.resize((w, h), Image.LANCZOS)
    if op == "brightness":
        factor = 1 + random.uniform(-0.3, 0.3)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)
    if op == "contrast":
        factor = 1 + random.uniform(-0.1, 0.1)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    if op == "color":
        factor = 1 + random.uniform(-0.1, 0.1)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)
    if op == "sharpness":
        factor = 1 + random.uniform(-0.3, 0.3)
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)
    if op == "gaussian_noise":
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 5, arr.shape)
        arr += noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if op == "shift_one_pixel":
        arr = np.array(img)
        dx = random.choice([-3, -2, -1, 1, 2, 3])
        dy = random.choice([-3, -2, -1, 1, 2, 3])
        arr = np.roll(arr, shift=dx, axis=1)
        arr = np.roll(arr, shift=dy, axis=0)
        return Image.fromarray(arr)
    if op == "rotate_small":
        angle = random.uniform(-4.0, 4.0)
        return img.rotate(angle, resample=Image.BICUBIC, expand=False)
    raise ValueError(f"Unknown transform: {op}")


def mutate_once(img: Image.Image):
    """Apply one random_transform picked from _TRANSFORMS."""
    op = random.choice(_TRANSFORMS)
    logger.debug("Chosen transform: %s", op)
    return random_transform(img, op), op


def generate_variants(
    path_in: Union[str, Path],
    n: int = 5,
    target_bits: int = 14,
    inter_bits: int = 6,
    max_iter: int = 300,
) -> List[Dict]:
    """Return list of unique variants meeting distance constraints.

    The function chains up to three random micro-edits to the original
    image per attempt and keeps only those variants whose perceptual hash
    differs sufficiently from both the original and previously accepted
    images.
    """
    logger.info(
        "Generating %d variants from %s (target_bits=%d, inter_bits=%d)",
        n,
        path_in,
        target_bits,
        inter_bits,
    )
    path_in = Path(path_in)
    orig = ImageOps.exif_transpose(Image.open(path_in)).convert("RGB")
    h0 = phash(orig)
    logger.debug("Original pHash %s", h0)
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
            logger.debug(
                "Discard candidate distance %d (<%d)",
                d0,
                target_bits,
            )
            continue
        if any(hamming(v["phash_int"], h1) < inter_bits for v in variants):
            logger.debug("Candidate too similar to existing variants")
            continue
        out_path = Path(f"/tmp/mut_{uuid.uuid4().hex}.png")
        img_tmp.save(out_path, "PNG")
        variants.append(
            dict(
                path_out=str(out_path),
                phash_int=int(str(h1), 16),
                distance_to_original=d0,
                ops_history=ops,
            )
        )
        logger.info(
            "Saved variant %d at %s with distance %d",
            len(variants),
            out_path,
            d0,
        )
    if len(variants) < n:
        msg = (
            f"Generated only {len(variants)} variants after "
            f"{iter_cnt} attempts."
        )
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info("Generated all %d variants in %d attempts", n, iter_cnt)
    return variants
