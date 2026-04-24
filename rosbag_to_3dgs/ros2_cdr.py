from __future__ import annotations

import struct
from typing import Any, Dict, Tuple


BASE_OFFSET = 4


def align(offset: int, alignment: int) -> int:
    return BASE_OFFSET + (((offset - BASE_OFFSET) + (alignment - 1)) & ~(alignment - 1))


def read_u8(blob: bytes, offset: int) -> Tuple[int, int]:
    return struct.unpack_from("<B", blob, offset)[0], offset + 1


def read_bool(blob: bytes, offset: int) -> Tuple[bool, int]:
    return struct.unpack_from("<?", blob, offset)[0], offset + 1


def read_u32(blob: bytes, offset: int) -> Tuple[int, int]:
    offset = align(offset, 4)
    return struct.unpack_from("<I", blob, offset)[0], offset + 4


def read_i32(blob: bytes, offset: int) -> Tuple[int, int]:
    offset = align(offset, 4)
    return struct.unpack_from("<i", blob, offset)[0], offset + 4


def read_f64(blob: bytes, offset: int) -> Tuple[float, int]:
    offset = align(offset, 8)
    return struct.unpack_from("<d", blob, offset)[0], offset + 8


def read_string(blob: bytes, offset: int) -> Tuple[str, int]:
    offset = align(offset, 4)
    length = struct.unpack_from("<I", blob, offset)[0]
    offset += 4
    raw = blob[offset : offset + length]
    offset += length
    return (raw[:-1].decode("utf-8") if length else ""), offset


def parse_header(blob: bytes) -> Dict[str, Any]:
    offset = BASE_OFFSET
    sec, offset = read_i32(blob, offset)
    nsec, offset = read_u32(blob, offset)
    frame_id, offset = read_string(blob, offset)
    return {"stamp_sec": sec, "stamp_nsec": nsec, "frame_id": frame_id, "offset": offset}
