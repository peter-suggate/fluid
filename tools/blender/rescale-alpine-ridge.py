"""Uniformly scale a baked ridge without changing its sampled zero set.

Run inside Blender with --factor positive-number. This scales native geometry,
FVX metre coordinates and distances together; normals, topology, UVs, materials
and sparse node indices are invariant. The source exporter must use the same
final world scale for a fresh rebuild.
"""
import argparse, gzip, json, math, struct, sys
from pathlib import Path
import bpy
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--factor', type=float, required=True)
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
if not math.isfinite(a.factor) or a.factor <= 0:
    p.error('--factor must be finite and positive')
root = Path(__file__).resolve().parents[2]
folder = root / 'public/assets/alpine-lake'
name = 'alpine-mountain-ridge'
raw = gzip.decompress((folder / (name + '.fvx.gz')).read_bytes())
assert raw[:4] == b'FVX1'
length = struct.unpack_from('<I', raw, 4)[0]
header = json.loads(raw[8:8 + length])
payload = bytearray(raw[8 + length:])
stride = 12 + 729 * 16
assert len(payload) == header['brickCount'] * stride
distances = np.ndarray((header['brickCount'], 729), dtype='<f4', buffer=payload,
                       offset=12, strides=(stride, 16))
distances *= a.factor
assert np.isfinite(distances).all()
header['origin'] = [v * a.factor for v in header['origin']]
header['cellSize'] *= a.factor
header['bandWidth'] *= a.factor
assert np.max(np.abs(distances)) <= header['bandWidth'] * (1 + 1e-6)
encoded = json.dumps(header, separators=(',', ':')).encode()
encoded += b' ' * ((-len(encoded)) % 4)
result = b'FVX1' + struct.pack('<I', len(encoded)) + encoded + payload

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(folder / (name + '.glb')))
parts = [o for o in bpy.context.scene.objects if o.type == 'MESH']
for obj in parts:
    obj.data.transform(obj.matrix_world)
    obj.matrix_world.identity()
    for v in obj.data.vertices:
        v.co *= a.factor
    obj.data.update()
bpy.ops.object.select_all(action='DESELECT')
for obj in parts:
    obj.select_set(True)
bpy.context.view_layer.objects.active = parts[0]
bpy.ops.export_scene.gltf(filepath=str(folder / (name + '.glb')),
                         export_format='GLB', use_selection=True,
                         export_apply=True, export_yup=True,
                         export_normals=True, export_materials='EXPORT')
bounds = [(min(v.co[k] for obj in parts for v in obj.data.vertices),
           max(v.co[k] for obj in parts for v in obj.data.vertices)) for k in range(3)]
with (folder / (name + '.fvx.gz')).open('wb') as stream:
    with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as zipped:
        zipped.write(result)
(root / 'lib/core/imported-assets/metadata' / (name + '-volume.json')).write_text(
    json.dumps(header, indent=2) + '\n')
manifest = json.loads((folder / 'manifest.json').read_text())
for asset in manifest['assets']:
    if asset['id'] == name:
        asset['cellSize'] = header['cellSize']
(folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

# Preserve the editable multi-asset source; scale only the ridge object.
bpy.ops.wm.open_mainfile(filepath=str(root / 'assets/alpine-lake/alpine-runtime.blend'))
obj = bpy.data.objects[name]
for v in obj.data.vertices:
    v.co *= a.factor
obj.data.update()
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(root / 'assets/alpine-lake/alpine-runtime.blend'))
print(json.dumps({'asset': name, 'factor': a.factor,
                  'nativeBlenderBounds': bounds, 'cellSize': header['cellSize'],
                  'dimensions': header['dimensions'],
                  'nodeDistancesScaled': int(distances.size)}), flush=True)
