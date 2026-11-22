"""Bake the Blender-exported, partly open ship mesh into the existing FVX/FVM pair.

Run with numpy, scipy, libigl and Pillow. The deck planking, cloth and ropes
are two-sided sheets in the source. A narrow solid shell gives them an inside
for exact picking and local subtraction. Texture charts stay in source metres.
"""
import gzip
import hashlib
import io
import json
import struct
import sys
from pathlib import Path

import igl
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/blender"))
from surface_mapping import affine_uv, rgba_mips, srgb_decode

ASSET = "sailing-galleon"
FOLDER = ROOT / "public/assets/galleons"
blob = (FOLDER / (ASSET + ".glb")).read_bytes()
size = struct.unpack_from("<I", blob, 12)[0]
model = json.loads(blob[20:20 + size])
binary = memoryview(blob)[28 + size:]


def view(index):
    entry = model["bufferViews"][index]
    at = entry.get("byteOffset", 0)
    return binary[at:at + entry["byteLength"]]


def accessor(index):
    a = model["accessors"][index]
    dtype = np.dtype({5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}[a["componentType"]])
    width = {"VEC3": 3, "VEC2": 2, "SCALAR": 1}[a["type"]]
    stride = model["bufferViews"][a["bufferView"]].get("byteStride", width * dtype.itemsize)
    return np.ndarray((a["count"], width), dtype, view(a["bufferView"]), a.get("byteOffset", 0), (stride, dtype.itemsize))


positions, triangles, coords, vertex_normals, face_materials = [], [], [], [], []


def collect(index, parent):
    node = model["nodes"][index]
    frame = np.eye(4)
    frame[:3, :3] = Rotation.from_quat(node.get("rotation", [0, 0, 0, 1])).as_matrix() @ np.diag(node.get("scale", [1, 1, 1]))
    frame[:3, 3] = node.get("translation", [0, 0, 0])
    world = parent @ frame
    if "mesh" in node:
        for part in model["meshes"][node["mesh"]]["primitives"]:
            attrs = part["attributes"]
            p = accessor(attrs["POSITION"])
            f = accessor(part["indices"]).reshape(-1, 3).astype(np.int64)
            triangles.append(f + sum(map(len, positions)))
            positions.append(p @ world[:3, :3].T + world[:3, 3])
            n = accessor(attrs["NORMAL"]) @ np.linalg.inv(world[:3, :3])
            vertex_normals.append(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12))
            coords.append(accessor(attrs["TEXCOORD_0"]) if "TEXCOORD_0" in attrs else np.zeros((len(p), 2)))
            face_materials.extend([part["material"]] * len(f))
    for child in node.get("children", []):
        collect(child, world)


for root in model["scenes"][model.get("scene", 0)]["nodes"]:
    collect(root, np.eye(4))
vertices, faces, uv = np.vstack(positions), np.vstack(triangles), np.vstack(coords)
material = np.array(face_materials)
area = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
valid = np.linalg.norm(area, axis=1) > 1e-9
faces, material = faces[valid], material[valid]
normals = area[valid] / np.linalg.norm(area[valid], axis=1)[:, None]
tree = igl.AABB()
tree.init(vertices, faces)
h = .10
band = 2 * h
thickness = .70 * h
# Field instances and previously saved cuts use this source-space pivot.
pivot = np.array([.25, 16.75, 3.5])
radius = np.maximum(pivot - vertices.min(axis=0), vertices.max(axis=0) - pivot)
dims = (8 * np.ceil((2 * radius / h + 8) / 8)).astype(int)
origin = pivot - .5 * dims * h
bricks = np.indices(dims // 8).reshape(3, -1).T * 8
distance, _, _ = tree.squared_distance(vertices, faces, origin + (bricks + 4) * h)
bricks = bricks[np.sqrt(distance) <= h * 4 * np.sqrt(3) + thickness + band]
offsets = np.indices((9, 9, 9)).reshape(3, -1).T[:, ::-1]
strides = np.array([1, dims[0] + 1, (dims[0] + 1) * (dims[1] + 1)])
keys, inverse = np.unique((bricks @ strides)[:, None] + offsets @ strides, return_inverse=True)
by_brick = inverse.reshape(-1, 729)
smooth_normals = np.vstack(vertex_normals)


def oct16(v):
    xy = v / np.maximum(np.abs(v).sum(axis=1), 1e-12)[:, None]
    reflected = (1 - np.abs(xy[:, [1, 0]])) * np.where(xy[:, :2] < 0, -1, 1)
    xy = np.where(xy[:, 2:] < 0, reflected, xy[:, :2])
    q = np.rint((xy * .5 + .5) * 65535).astype("<u4")
    return q[:, 0] | (q[:, 1] << 16)


records = np.zeros((len(keys), 4), dtype="<u4")
chart_ids = np.empty(len(keys), dtype="<u4")
# Query in bounded batches. All incident bricks reference the same record at
# their shared corners, including the interpolated, source-authored normal.
for start in range(0, len(keys), 65536):
    batch = slice(start, start + 65536)
    key = keys[batch]
    ijk = np.column_stack((key % strides[1], key // strides[1] % (dims[1] + 1), key // strides[2]))
    pts = origin + ijk * h
    distance2, owners, near = tree.squared_distance(vertices, faces, pts)
    distance = np.sqrt(distance2)
    grad = (pts - near) / np.maximum(distance[:, None], 1e-12)
    tri = faces[owners]
    weights = igl.barycentric_coordinates(near, vertices[tri[:, 0]], vertices[tri[:, 1]], vertices[tri[:, 2]])
    normal = np.einsum("bi,bij->bj", weights, smooth_normals[tri])
    degenerate = ~np.isfinite(normal).all(axis=1) | (np.linalg.norm(normal, axis=1) < 1e-8)
    normal[degenerate] = normals[owners[degenerate]]
    normal[(normals[owners] * grad).sum(axis=1) < 0] *= -1
    records[batch, 0] = np.clip(distance - thickness, -band, band).astype("<f4").view("<u4")
    records[batch, 1] = oct16(grad)
    records[batch, 2] = oct16(normal)
    records[batch, 3] = material[owners] + 1
    chart_ids[batch] = owners + 1

# No hit cell can use a corner more than a cell diagonal from the surface.
# Keep its distance for traversal, omit irrelevant normal and material data.
air = records[:, 0].view("<f4") > np.sqrt(3) * h
records[air, 1:] = 0
chart_ids[air] = 0

# A missing brick reads as air. Omit bricks whose samples are all at the
# exterior band limit so empty rigging space costs no GPU node storage.
active = (records[:, 0].view("<f4")[by_brick] < np.float32(band)).any(axis=1)
bricks, by_brick = bricks[active], by_brick[active]
payload = b"".join(b.astype("<i4").tobytes() + records[indices].tobytes() for b, indices in zip(bricks, by_brick))
common = dict(version=1, asset=ASSET, origin=origin.tolist(), dimensions=dims.tolist(), cellSize=h,
              brickCount=len(bricks), brickCells=8)
layers, surfaces, palette = [], [], []
texture_size = 2048
unused = 0xffffffff


def pixels(info):
    image = model["images"][model["textures"][info["index"]]["source"]]
    # Convert glTF top-origin UV and image together to the FVM bottom-origin convention.
    return np.asarray(Image.open(io.BytesIO(view(image["bufferView"]))).convert("RGB")
                      .resize((texture_size, texture_size), Image.Resampling.LANCZOS), dtype=np.float32)[::-1] / 255


uv[:, 1] = 1 - uv[:, 1]
for m in model["materials"]:
    pm = m.get("pbrMetallicRoughness", {})
    base = pm.get("baseColorFactor", [1, 1, 1])[:3]
    rough = pm.get("roughnessFactor", 1)
    metallic = pm.get("metallicFactor", 1)
    color_layer = normal_layer = unused
    fallback = base
    if "baseColorTexture" in pm:
        color = srgb_decode(pixels(pm["baseColorTexture"]))
        alpha = pixels(pm["metallicRoughnessTexture"])[:, :, 1] if "metallicRoughnessTexture" in pm else np.ones(color.shape[:2])
        color_layer = len(layers)
        layers.append(rgba_mips(color, alpha, texture_size))
        fallback = (np.median(color.reshape(-1, 3), axis=0) * base).tolist()
    if "normalTexture" in m:
        normal_layer = len(layers)
        layers.append(rgba_mips(pixels(m["normalTexture"]), np.ones((texture_size, texture_size)), texture_size, normal=True))
    surfaces.append(dict(name=m["name"], baseColorFactor=base, roughnessFactor=rough, metallicFactor=metallic,
                         colorRoughnessLayer=color_layer, normalLayer=normal_layer, normalStrength=m.get("normalTexture", {}).get("scale", 1)))
    palette.append(dict(name=m["name"], baseColorLinear=fallback, roughness=rough, metallic=metallic))

maps = np.zeros((len(faces) + 1, 16), dtype="<u4")
for i, f in enumerate(faces):
    chart = affine_uv(vertices[f], uv[f])
    row = maps[i + 1]
    if chart:
        u, v, tangent, sign, _ = chart
        row.view("<f4")[:12] = [*u, *v, *tangent, sign]
    else:
        row.view("<f4")[11] = 1
    row[12] = material[i]
nodes = np.stack((chart_ids, records[:, 2]), axis=-1).astype("<u4")[by_brick]
sha = lambda b: hashlib.sha256(b).hexdigest()


def file_bytes(magic, header, data):
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    return magic + struct.pack("<I", len(encoded)) + encoded + data


mapping = dict(**common, sourceSha256=sha(blob), fvxPayloadSha256=sha(payload), nodeStrideBytes=8, mappingStrideBytes=64,
               mappingCount=len(maps), mappingByteOffset=0, nodeByteOffset=maps.nbytes, textureByteOffset=maps.nbytes + nodes.nbytes,
               textureWidth=texture_size, textureHeight=texture_size, textureLayerCount=len(layers), mipLevelCount=texture_size.bit_length(),
               textureFormat="rgba8unorm-srgb", textureLayout="layer-major-mip-major", textureOrigin="blender-uv-bottom-left", materials=surfaces)
companion = file_bytes(b"FVM1", mapping, maps.tobytes() + nodes.tobytes() + b"".join(layers))
header = dict(**common, units="metres", upAxis="Y", bandWidth=band, nodeStrideBytes=16, materials=palette,
              surfaceMapping=dict(url="/assets/galleons/sailing-galleon.fvm.gz", sha256=sha(companion)))
for ext, data in [("fvx", file_bytes(b"FVX1", header, payload)), ("fvm", companion)]:
    (FOLDER / f"{ASSET}.{ext}.gz").write_bytes(gzip.compress(data, compresslevel=6, mtime=0))
(ROOT / "lib/core/imported-assets/metadata/sailing-galleon-volume.json").write_text(json.dumps(header, indent=2) + "\n")
