"""Deterministic, portable whitewash maps shared by Cycles, glTF and FVM.

The atlas uses circumference U and tower-height V. It contains reflectance,
roughness and tangent normals only, never baked illumination or shadows.
"""
from pathlib import Path
import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def whitewash_material(size=2048):
    existing = bpy.data.materials.get('Atlantic salt-weathered whitewash')
    if existing:
        return existing
    rng = np.random.default_rng(20261031)
    u, v = np.meshgrid(np.arange(size) / size, np.arange(size) / size)

    def noise(nx, ny):
        grid = rng.random((ny + 1, nx + 1))
        grid[:, -1] = grid[:, 0]
        x, y = u * nx, v * ny
        ix, iy = x.astype(int), y.astype(int)
        fx, fy = x - ix, y - iy
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        return ((grid[iy, ix] * (1-fx) + grid[iy, ix+1] * fx) * (1-fy)
                + (grid[iy+1, ix] * (1-fx) + grid[iy+1, ix+1] * fx) * fy)

    broad = noise(26, 32)
    flakes = .5 * noise(75, 105) + .3 * noise(180, 240) + .2 * noise(420, 470)
    streaks = noise(130, 5) * .65 + noise(47, 9) * .35
    base_damp = np.clip((.23-v) / .23, 0, 1)
    peeling = np.clip((broad*.55 + flakes*.45 - (.68 - base_damp*.14)) * 18, 0, 1)
    runoff = np.clip((streaks - .49) * 2.8, 0, .65) * (.3 + .7*broad)
    grain = noise(650, 680)
    lime = np.stack([.76 + .035*grain, .75 + .032*grain, .69 + .033*grain], axis=-1)
    stone = np.stack([.32 + .12*flakes, .28 + .11*flakes, .21 + .09*flakes], axis=-1)
    rgb = lime*(1-peeling[..., None]) + stone*peeling[..., None]
    rgb *= (1 - .38*runoff - .24*base_damp)[..., None]
    # Sparse old horizontal pointing survives underneath the limewash.
    joints = np.exp(-((np.sin(v * np.pi * 43)) / .055)**2) * peeling * .09
    rgb -= joints[..., None]
    height = .0018*grain - .006*peeling - .003*joints
    dv, du = np.gradient(height, 9/size, 7/size)
    normal = np.stack([-du, -dv, np.ones_like(du)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1)[..., None]
    rough = np.clip(.76 + .14*grain - .13*base_damp, .55, .96)
    folder = ROOT/'assets/atlantic-lighthouse/textures'
    folder.mkdir(parents=True, exist_ok=True)

    def image(name, data, srgb=False):
        im = bpy.data.images.new(name, width=size, height=size, alpha=True)
        im.colorspace_settings.name = 'sRGB' if srgb else 'Non-Color'
        rgba = np.ones((size, size, 4), dtype=np.float32)
        # Image pixels are encoded for glTF and FVM; Blender decodes sRGB.
        rgba[:, :, :3] = data
        im.pixels.foreach_set(rgba.ravel())
        im.filepath_raw = str(folder/(name+'.png'))
        im.file_format = 'PNG'
        im.save()
        im.pack()
        return im

    encoded = np.where(rgb <= .0031308, rgb*12.92, 1.055*rgb**(1/2.4)-.055)
    images = [image('whitewash-color', encoded, True),
              image('whitewash-roughness', np.repeat(rough[..., None], 3, axis=-1)),
              image('whitewash-normal', normal*.5+.5)]
    mat = bpy.data.materials.new('Atlantic salt-weathered whitewash')
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    for im, socket in zip(images, ['Base Color', 'Roughness', 'Normal']):
        tex = nodes.new('ShaderNodeTexImage'); tex.image = im
        if socket == 'Normal':
            mapping = nodes.new('ShaderNodeNormalMap')
            links.new(tex.outputs['Color'], mapping.inputs['Color'])
            links.new(mapping.outputs['Normal'], bsdf.inputs[socket])
        else:
            links.new(tex.outputs['Color'], bsdf.inputs[socket])
    return mat


def tower_uv(mesh, origin=(0, 0, 0)):
    """Cylindrical atlas without a seam spanning a triangle."""
    import math
    uv = mesh.uv_layers.new(name='Whitewash cylindrical atlas')
    for face in mesh.polygons:
        coords = [mesh.vertices[mesh.loops[i].vertex_index].co for i in face.loop_indices]
        us = [(math.atan2(p.y+origin[1], p.x+origin[0]-7)/math.tau) % 1 for p in coords]
        if max(us)-min(us) > .5:
            us = [u+1 if u < .5 else u for u in us]
        for loop, p, u in zip(face.loop_indices, coords, us):
            uv.data[loop].uv = (u, (p.z+origin[2]-5.8)/9)
