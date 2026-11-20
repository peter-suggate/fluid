"""Bake Poly Haven's 1K Kloofendal 48d Pure Sky HDR to the renderer's 60 KiB
hemisphere. Run in Blender with -- <path to source .hdr>. The sun direction
is the zero meridian. Direct illumination is supplied by the scene light.
"""
import base64
import json
from pathlib import Path
import sys
import bpy
import numpy as np

source = bpy.data.images.load(sys.argv[sys.argv.index('--') + 1])
width, height = source.size
pixels = np.empty(width * height * 4, dtype=np.float32)
source.pixels.foreach_get(pixels)
sky = pixels.reshape(height, width, 4)[height // 2:, :, :3]
sun_row, sun_col = np.unravel_index(sky.mean(axis=2).argmax(), sky.shape[:2])
sky = np.roll(sky, -int(sun_col), axis=1)
# Remove only the solar disc, retaining the aureole and cloud silver lining.
sky = np.minimum(sky, 4)
mean = np.average(sky.mean(axis=1), axis=0, weights=np.cos((np.arange(sky.shape[0])+.5)*np.pi/(2*sky.shape[0])))
# Retain the source's horizontal resolution for the cloud edges seen from
# the terrace. Allocate finer angular spacing near the horizon, where cloud
# banks are foreshortened. Area filtering keeps the zenith free of aliasing.
cols, rows = 1024, 120
sky = sky.reshape(sky.shape[0], cols, width // cols, 3).mean(axis=2)
u = np.arange(rows+1)/rows
edges = sky.shape[0]*u/(1.8-.8*u)
bands = []
for start, end in zip(edges, edges[1:]):
    samples = np.arange(int(np.floor(start)), min(sky.shape[0], int(np.ceil(end))))
    weights = np.minimum(end, samples+1) - np.maximum(start, samples)
    bands.append(np.average(sky[samples], axis=0, weights=weights))
sky = np.stack(bands)
# Endpoints in sqrt radiance with four interpolants for each 4x4 block.
quant = np.array([31, 63, 31])
packed = bytearray()
for row in range(0, rows, 4):
    for col in range(0, cols, 4):
        block = np.sqrt(sky[row:row+4, col:col+4].reshape(16,3) / 4)
        luma = block @ np.array([.2126,.7152,.0722])
        ends = np.rint(block[[luma.argmin(), luma.argmax()]] * quant).astype(int)
        for r,g,b in ends:
            packed.extend(int(r*2048 + g*32 + b).to_bytes(2,'little'))
        ramp = np.linspace(ends[0]/quant, ends[1]/quant, 4)
        indices = ((block[:,None,:] - ramp[None,:,:])**2).sum(axis=2).argmin(axis=1)
        packed.extend(sum(int(v) << (2*i) for i,v in enumerate(indices)).to_bytes(4,'little'))
target = Path(__file__).resolve().parents[2]/'lib/core/imported-assets/metadata/kloofendal-sky.json'
target.write_text(json.dumps(dict(width=cols, height=rows, meanLinear=mean.tolist(), sunElevation_rad=float((sun_row+.5)*np.pi/height),
    blocks=base64.b64encode(packed).decode()),separators=(',',':'))+'\n')
