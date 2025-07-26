from typing import Union
import imagehash


def hamming(
    a: Union[int, imagehash.ImageHash], b: Union[int, imagehash.ImageHash]
) -> int:
    """Return Hamming distance between two 64-bit hashes."""
    if not isinstance(a, int):
        a = int(str(a), 16)
    if not isinstance(b, int):
        b = int(str(b), 16)
    return bin(a ^ b).count("1")
