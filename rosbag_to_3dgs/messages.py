from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .ros2_cdr import parse_header, read_bool, read_f64, read_string, read_u8, read_u32


def parse_compressed_image(blob: bytes) -> Dict[str, Any]:
    header = parse_header(blob)
    offset = header["offset"]
    image_format, offset = read_string(blob, offset)
    data_len, offset = read_u32(blob, offset)
    data = blob[offset : offset + data_len]
    return {**header, "format": image_format, "data_len": data_len, "data": data}


def decode_compressed_image(blob: bytes, *, mode: int):
    from data_preparation.shared.io import require_cv2

    cv2 = require_cv2("ROS compressed image decoding")
    message = parse_compressed_image(blob)
    image = cv2.imdecode(np.frombuffer(message["data"], dtype=np.uint8), mode)
    if image is None:
        raise ValueError("Failed to decode compressed image.")
    return {**message, "image": image, "decoded_shape": tuple(int(v) for v in image.shape)}


def parse_odometry(blob: bytes) -> Dict[str, Any]:
    header = parse_header(blob)
    offset = header["offset"]
    child_frame_id, offset = read_string(blob, offset)
    position = []
    for _ in range(3):
        value, offset = read_f64(blob, offset)
        position.append(value)
    orientation = []
    for _ in range(4):
        value, offset = read_f64(blob, offset)
        orientation.append(value)
    return {
        **header,
        "child_frame_id": child_frame_id,
        "position": np.asarray(position, dtype=np.float64),
        "orientation_xyzw": np.asarray(orientation, dtype=np.float64),
    }


def parse_imu(blob: bytes) -> Dict[str, Any]:
    return parse_header(blob)


def parse_pointcloud2(blob: bytes, *, include_data: bool = True) -> Dict[str, Any]:
    header = parse_header(blob)
    offset = header["offset"]
    height, offset = read_u32(blob, offset)
    width, offset = read_u32(blob, offset)
    field_count, offset = read_u32(blob, offset)
    fields = []
    for _ in range(field_count):
        name, offset = read_string(blob, offset)
        field_offset, offset = read_u32(blob, offset)
        datatype, offset = read_u8(blob, offset)
        count, offset = read_u32(blob, offset)
        fields.append({"name": name, "offset": field_offset, "datatype": datatype, "count": count})
    is_bigendian, offset = read_bool(blob, offset)
    point_step, offset = read_u32(blob, offset)
    row_step, offset = read_u32(blob, offset)
    data_len, offset = read_u32(blob, offset)
    result = {
        **header,
        "height": height,
        "width": width,
        "fields": fields,
        "is_bigendian": is_bigendian,
        "point_step": point_step,
        "row_step": row_step,
        "data_len": data_len,
    }
    if include_data:
        result["data"] = blob[offset : offset + data_len]
    return result
