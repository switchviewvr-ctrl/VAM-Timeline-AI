"""Technical AcidBubbles Timeline keyframe codec.

Ported from the old mocap compiler as a technical utility only. It contains no
semantic labels or curation assumptions.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from vam_timeline_ai.io.json_utils import as_float, as_int


SMOOTH_LOCAL = 3


@dataclass(frozen=True)
class TimelineKeyframe:
    time: float
    value: float
    curve_type: int
    control_point_in: float | None = None
    control_point_out: float | None = None
    encoded: str | None = None
    inherited_value: bool = False
    inherited_curve_type: bool = False


def decode_float_le_hex(hex_str: str) -> float:
    if len(hex_str) != 8:
        raise ValueError(f"expected 4-byte float hex, got {hex_str!r}")
    return struct.unpack("<f", bytes.fromhex(hex_str))[0]


def encode_float_le_hex(value: float) -> str:
    return struct.pack("<f", float(value)).hex().upper()


def snap_time(value: float) -> float:
    rounded = round(float(value) * 1000.0) / 1000.0
    return 0.0 if rounded < 0.0 else rounded


def decode_keyframe(
    encoded: str | dict[str, Any],
    last_value: float = 0.0,
    last_curve_type: int = SMOOTH_LOCAL,
    version: int = 283,
) -> TimelineKeyframe:
    if isinstance(encoded, dict):
        return _decode_uncompressed_keyframe(encoded, last_value, last_curve_type)
    if not isinstance(encoded, str) or len(encoded) < 9:
        raise ValueError(f"invalid encoded keyframe {encoded!r}")
    if version <= 230:
        return _decode_legacy_compressed_keyframe(encoded, last_value, last_curve_type)

    encoded_value = ord(encoded[0]) - ord("A")
    has_value = (encoded_value & 1) != 0
    has_curve_type = (encoded_value & 2) != 0
    t = decode_float_le_hex(encoded[1:9])
    index = 9
    if has_value:
        value = decode_float_le_hex(encoded[index:index + 8])
        index += 8
    else:
        value = last_value
    if has_curve_type:
        curve_type = int(encoded[index:index + 2], 16)
    else:
        curve_type = last_curve_type
    return TimelineKeyframe(
        time=float(t),
        value=float(value),
        curve_type=int(curve_type),
        encoded=encoded,
        inherited_value=not has_value,
        inherited_curve_type=not has_curve_type,
    )


def decode_keyframe_sequence(items: list[str | dict[str, Any]] | None, version: int = 283) -> list[TimelineKeyframe]:
    last_v = 0.0
    last_c = SMOOTH_LOCAL
    last_t: float | None = None
    keys: list[TimelineKeyframe] = []
    for item in items or []:
        key = decode_keyframe(item, last_v, last_c, version)
        if key.time < 0:
            continue
        if last_t is not None and abs(key.time - last_t) <= 1e-8:
            continue
        keys.append(key)
        last_t = key.time
        last_v = key.value
        last_c = key.curve_type
    return keys


def encode_keyframe(
    keyframe: TimelineKeyframe,
    last_value: float = 0.0,
    last_curve_type: int = SMOOTH_LOCAL,
) -> str:
    has_value = abs(float(last_value) - float(keyframe.value)) > 1e-7
    has_curve_type = int(last_curve_type) != int(keyframe.curve_type)
    encoded_value = 0
    if has_value:
        encoded_value |= 1
    if has_curve_type:
        encoded_value |= 2
    result = chr(ord("A") + encoded_value) + encode_float_le_hex(keyframe.time)
    if has_value:
        result += encode_float_le_hex(keyframe.value)
    if has_curve_type:
        result += f"{int(keyframe.curve_type) & 0xFF:02X}"
    return result


def encode_keyframe_sequence(
    keyframes: list[TimelineKeyframe],
    *,
    initial_value: float = 0.0,
    initial_curve_type: int = SMOOTH_LOCAL,
) -> list[str]:
    last_v = float(initial_value)
    last_c = int(initial_curve_type)
    out: list[str] = []
    for keyframe in keyframes:
        out.append(encode_keyframe(keyframe, last_v, last_c))
        last_v = float(keyframe.value)
        last_c = int(keyframe.curve_type)
    return out


def _decode_uncompressed_keyframe(obj: dict[str, Any], last_value: float, last_curve_type: int) -> TimelineKeyframe:
    t = as_float(obj.get("t"), as_float(obj.get("time"), 0.0)) or 0.0
    has_v = "v" in obj or "value" in obj
    has_c = "c" in obj or "curveType" in obj
    value = as_float(obj.get("v", obj.get("value")), last_value)
    curve_type = as_int(obj.get("c", obj.get("curveType")), last_curve_type)
    return TimelineKeyframe(
        time=snap_time(t),
        value=float(value if value is not None else last_value),
        curve_type=int(curve_type if curve_type is not None else last_curve_type),
        control_point_in=as_float(obj.get("i", obj.get("controlPointIn"))),
        control_point_out=as_float(obj.get("o", obj.get("controlPointOut"))),
        inherited_value=not has_v,
        inherited_curve_type=not has_c,
    )


def _decode_legacy_compressed_keyframe(encoded: str, last_value: float, last_curve_type: int) -> TimelineKeyframe:
    try:
        size_char = encoded[0]
        if "0" <= size_char <= "9":
            index = ord(size_char) - ord("0")
        elif "a" <= size_char <= "z":
            index = ord(size_char) - ord("a") + 10
        else:
            index = ord(size_char) - ord("A") + 36
        t_bytes = index // 25
        index %= 25
        v_bytes = index // 5
        has_c = (index % 5) != 0
        t = _decode_float_variable(encoded[1:1 + t_bytes * 2])
        value = last_value if v_bytes == 0 else _decode_float_variable(encoded[1 + t_bytes * 2:1 + (t_bytes + v_bytes) * 2])
        c = last_curve_type if not has_c else int(encoded[1 + (t_bytes + v_bytes) * 2:1 + (t_bytes + v_bytes) * 2 + 2], 16)
        return TimelineKeyframe(float(t), float(value), int(c), encoded=encoded, inherited_value=v_bytes == 0, inherited_curve_type=not has_c)
    except Exception:
        return TimelineKeyframe(-1.0, last_value, last_curve_type, encoded=encoded)


def _decode_float_variable(hex_str: str) -> float:
    return decode_float_le_hex(hex_str.ljust(8, "0"))
