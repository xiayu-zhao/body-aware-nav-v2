"""Minimal dependency-free PLY reader (ascii + binary, little/big endian).

Written in pure numpy so the abstraction stage runs on a bare
``pip install -e .``; the as-built lab-floor scan (Sec. VI) is a
multi-gigabyte binary PLY exported from CloudCompare, so the reader
memory-maps the vertex block and supports strided subsampling instead of
materializing the whole cloud.

Only the vertex element is of interest; face/edge elements are skipped.
List properties (which faces use) have no fixed record size, so an element
carrying them cannot be seeked *past* — that is only a problem when such an
element precedes the vertices, which no scanner export does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: PLY scalar type names -> numpy type characters. The ``int8``/``uint8``
#: spellings are the PLY 1.0 aliases some exporters emit.
_PLY_TO_NUMPY: dict[str, str] = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}

_BYTE_ORDER = {
    "ascii": "=",
    "binary_little_endian": "<",
    "binary_big_endian": ">",
}

#: Candidate property names for RGB colour, in order of preference.
_COLOR_NAMES: tuple[tuple[str, str, str], ...] = (
    ("red", "green", "blue"),
    ("r", "g", "b"),
    ("diffuse_red", "diffuse_green", "diffuse_blue"),
)


class PlyError(ValueError):
    """Malformed or unsupported PLY input."""


@dataclass
class PlyProperty:
    name: str
    #: numpy type char for scalar properties; None for list properties
    dtype: str | None
    count_dtype: str | None = None
    item_dtype: str | None = None

    @property
    def is_list(self) -> bool:
        return self.dtype is None


@dataclass
class PlyElement:
    name: str
    count: int
    properties: list[PlyProperty] = field(default_factory=list)
    #: byte offset of the first record; None when an earlier element had
    #: list properties and could not be measured
    offset: int | None = None

    @property
    def is_fixed_size(self) -> bool:
        return all(not p.is_list for p in self.properties)

    def numpy_dtype(self, byte_order: str) -> np.dtype:
        if not self.is_fixed_size:
            raise PlyError(f"element '{self.name}' has list properties")
        return np.dtype(
            [(p.name, byte_order + p.dtype) for p in self.properties]
        )

    def itemsize(self, byte_order: str) -> int:
        return self.numpy_dtype(byte_order).itemsize

    def has(self, name: str) -> bool:
        return any(p.name == name for p in self.properties)


@dataclass
class PlyHeader:
    fmt: str
    elements: list[PlyElement]
    header_bytes: int

    @property
    def byte_order(self) -> str:
        return _BYTE_ORDER[self.fmt]

    def element(self, name: str) -> PlyElement | None:
        for e in self.elements:
            if e.name == name:
                return e
        return None


# ----------------------------------------------------------------------
def read_header(path: str | Path, *, max_header_bytes: int = 1 << 20) -> PlyHeader:
    """Parse the ascii header without touching the payload."""
    raw = bytearray()
    terminator = b"end_header"
    with open(path, "rb") as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                raise PlyError(f"{path}: no 'end_header' found")
            raw.extend(chunk)
            idx = raw.find(terminator)
            if idx != -1:
                # consume through the newline that closes the keyword
                nl = raw.find(b"\n", idx)
                if nl != -1:
                    header_bytes = nl + 1
                    raw = raw[:header_bytes]
                    break
            if len(raw) > max_header_bytes:
                raise PlyError(f"{path}: header exceeds {max_header_bytes} bytes")

    lines = raw.decode("ascii", errors="replace").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise PlyError(f"{path}: missing 'ply' magic (not a PLY file?)")

    fmt: str | None = None
    elements: list[PlyElement] = []
    for line in lines[1:]:
        tok = line.split()
        if not tok or tok[0] == "comment" or tok[0] == "obj_info":
            continue
        key = tok[0]
        if key == "format":
            fmt = tok[1]
            if fmt not in _BYTE_ORDER:
                raise PlyError(f"{path}: unsupported format '{fmt}'")
        elif key == "element":
            elements.append(PlyElement(name=tok[1], count=int(tok[2])))
        elif key == "property":
            if not elements:
                raise PlyError(f"{path}: property outside any element")
            if tok[1] == "list":
                count_t, item_t, name = tok[2], tok[3], tok[4]
                for t in (count_t, item_t):
                    if t not in _PLY_TO_NUMPY:
                        raise PlyError(f"{path}: unknown list type '{t}'")
                elements[-1].properties.append(
                    PlyProperty(
                        name=name,
                        dtype=None,
                        count_dtype=_PLY_TO_NUMPY[count_t],
                        item_dtype=_PLY_TO_NUMPY[item_t],
                    )
                )
            else:
                type_name, name = tok[1], tok[2]
                if type_name not in _PLY_TO_NUMPY:
                    raise PlyError(f"{path}: unknown property type '{type_name}'")
                elements[-1].properties.append(
                    PlyProperty(name=name, dtype=_PLY_TO_NUMPY[type_name])
                )
        elif key == "end_header":
            break

    if fmt is None:
        raise PlyError(f"{path}: header has no 'format' line")
    if not elements:
        raise PlyError(f"{path}: header declares no elements")

    header = PlyHeader(fmt=fmt, elements=elements, header_bytes=header_bytes)

    # Byte offsets are only meaningful (and only computable) for binary
    # payloads made of fixed-size records.
    if fmt != "ascii":
        cursor: int | None = header_bytes
        for el in elements:
            el.offset = cursor
            if cursor is not None and el.is_fixed_size:
                cursor += el.count * el.itemsize(header.byte_order)
            else:
                cursor = None
    return header


# ----------------------------------------------------------------------
def _resolve_stride(count: int, stride: int, max_points: int | None) -> int:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if max_points is not None and max_points > 0 and count > 0:
        stride = max(stride, -(-count // max_points))  # ceil division
    return stride


def _read_vertex_binary(
    path: str | Path, header: PlyHeader, el: PlyElement, stride: int
) -> np.ndarray:
    if el.offset is None:
        raise PlyError(
            f"{path}: cannot locate vertex data — an earlier element has "
            "list properties (unsupported layout)"
        )
    if not el.is_fixed_size:
        raise PlyError(f"{path}: vertex element has list properties (unsupported)")
    if el.count == 0:
        return np.empty(0, dtype=el.numpy_dtype(header.byte_order))
    mm = np.memmap(
        path,
        mode="r",
        dtype=el.numpy_dtype(header.byte_order),
        offset=el.offset,
        shape=(el.count,),
    )
    return mm[::stride] if stride > 1 else mm


def _read_vertex_ascii(
    path: str | Path, header: PlyHeader, el: PlyElement, stride: int
) -> np.ndarray:
    dtype = el.numpy_dtype("=")
    rows: list[tuple] = []
    ncols = len(el.properties)
    with open(path, "r", encoding="ascii", errors="replace") as f:
        # re-scan past the header (byte offsets are unreliable in text mode)
        for line in f:
            if line.strip() == "end_header":
                break
        for i in range(el.count):
            line = f.readline()
            if not line:
                raise PlyError(f"{path}: truncated ascii payload at vertex {i}")
            if i % stride:
                continue
            tok = line.split()
            if len(tok) < ncols:
                raise PlyError(
                    f"{path}: vertex {i} has {len(tok)} fields, expected {ncols}"
                )
            rows.append(tuple(tok[:ncols]))
    return np.array(rows, dtype=dtype) if rows else np.empty(0, dtype=dtype)


def read_ply(
    path: str | Path,
    *,
    stride: int = 1,
    max_points: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read the vertex element of a PLY file.

    Args:
        path: file to read.
        stride: keep every ``stride``-th vertex (uniform subsample).
        max_points: raise ``stride`` until at most this many vertices are
            returned. Combines with ``stride`` by taking the coarser of the
            two.

    Returns:
        ``(xyz, fields)`` where ``xyz`` is (N, 3) float64 and ``fields``
        maps every remaining vertex property name to its (N,) array.
    """
    path = Path(path)
    header = read_header(path)
    el = header.element("vertex")
    if el is None:
        raise PlyError(f"{path}: no 'vertex' element")
    for axis in ("x", "y", "z"):
        if not el.has(axis):
            raise PlyError(f"{path}: vertex element lacks '{axis}'")

    stride = _resolve_stride(el.count, stride, max_points)
    if header.fmt == "ascii":
        rec = _read_vertex_ascii(path, header, el, stride)
    else:
        rec = _read_vertex_binary(path, header, el, stride)

    xyz = np.stack(
        [np.asarray(rec["x"], dtype=np.float64),
         np.asarray(rec["y"], dtype=np.float64),
         np.asarray(rec["z"], dtype=np.float64)],
        axis=1,
    )
    fields = {
        p.name: np.asarray(rec[p.name])
        for p in el.properties
        if p.name not in ("x", "y", "z")
    }
    return xyz, fields


def extract_colors(fields: dict[str, np.ndarray]) -> np.ndarray | None:
    """Pull an (N, 3) float colour array in [0, 1] out of vertex fields.

    Integer channels are assumed 0-255 and normalized; float channels are
    assumed already normalized and are clipped.
    """
    for names in _COLOR_NAMES:
        if all(n in fields for n in names):
            cols = np.stack([fields[n] for n in names], axis=1)
            if np.issubdtype(cols.dtype, np.integer):
                return cols.astype(np.float64) / 255.0
            return np.clip(cols.astype(np.float64), 0.0, 1.0)
    return None


def write_ply(
    path: str | Path,
    xyz: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    comments: list[str] | None = None,
) -> Path:
    """Write a binary little-endian PLY (x y z float32 [+ rgb uchar]).

    The counterpart of :func:`read_ply`, used to land clouds exported from
    USD -- which can only be read under Isaac's Python -- in a format the
    rest of BodyNav reads with no Isaac dependency at all.

    Args:
        xyz: (N, 3) point positions.
        colors: (N, 3) RGB, either uint8 in [0, 255] or float in [0, 1].
        comments: lines recorded in the header. Provenance belongs here --
            the source stage and prim, so an exported cloud is never
            anonymous.
    """
    path = Path(path)
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise PlyError(f"xyz must be (N, 3), got {xyz.shape}")

    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape != xyz.shape:
            raise PlyError(
                f"colors {colors.shape} does not match xyz {xyz.shape}"
            )
        if colors.dtype.kind == "f":
            colors = np.clip(colors * 255.0 + 0.5, 0, 255)
        colors = colors.astype(np.uint8)
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    rec = np.empty(len(xyz), dtype=np.dtype(fields))
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if colors is not None:
        rec["red"], rec["green"], rec["blue"] = colors.T

    header = ["ply", "format binary_little_endian 1.0"]
    header += [f"comment {c}" for c in (comments or [])]
    header.append(f"element vertex {len(xyz)}")
    header += [
        f"property {'float' if dt.endswith('f4') else 'uchar'} {name}"
        for name, dt in fields
    ]
    header.append("end_header")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(rec.tobytes())
    return path
