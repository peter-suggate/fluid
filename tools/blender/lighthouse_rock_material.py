"""Shared photographed coastal-rock material for Cycles and voxel FVM baking.

The CC0 source images and attribution live in assets/alpine-lake/textures.
UVs affect surface appearance only; the collision/FVX geometry is unchanged.
"""
from pathlib import Path
import bpy
import json
import struct

ROOT=Path(__file__).resolve().parents[2]
TEXTURES=ROOT/'assets/alpine-lake/textures'
ROCK_FACTOR=(.58,.55,.50,1)


def photographed_rock_material():
    material=bpy.data.materials.new('Atlantic coastal rock — CC0 mineral photographs')
    material.use_nodes=True
    nodes,links=material.node_tree.nodes,material.node_tree.links
    bsdf=nodes.get('Principled BSDF')
    def texture(suffix,color):
        node=nodes.new('ShaderNodeTexImage')
        path=(TEXTURES.parent/'alpine-strata-neutral.png') if suffix=='diff' else TEXTURES/('aerial_rocks_02_'+suffix+'_2k.jpg')
        node.image=bpy.data.images.load(str(path),check_existing=True)
        node.image.colorspace_settings.name=color
        node.image.pack()
        node.extension='REPEAT'
        return node
    diffuse=texture('diff','sRGB')
    tint=nodes.new('ShaderNodeMixRGB');tint.blend_type='MULTIPLY';tint.inputs[0].default_value=1
    tint.inputs[2].default_value=ROCK_FACTOR
    links.new(diffuse.outputs['Color'],tint.inputs[1]);links.new(tint.outputs[0],bsdf.inputs['Base Color'])
    rough=texture('rough','Non-Color');links.new(rough.outputs['Color'],bsdf.inputs['Roughness'])
    normal=texture('nor_gl','Non-Color');mapping=nodes.new('ShaderNodeNormalMap');mapping.inputs['Strength'].default_value=.6
    links.new(normal.outputs['Color'],mapping.inputs['Color']);links.new(mapping.outputs['Normal'],bsdf.inputs['Normal'])
    bsdf.inputs['Metallic'].default_value=0
    return material


def assign_rock_uv(mesh,origin=(0,0,0)):
    """Dominant-face charts at a four-metre repeat, for real voxel surface UVs."""
    mesh.update()
    uv=mesh.uv_layers.get('Coastal mineral charts') or mesh.uv_layers.new(name='Coastal mineral charts')
    for face in mesh.polygons:
        axis=max(range(3),key=lambda a:abs(face.normal[a]))
        a,b=((1,2),(0,2),(0,1))[axis]
        for loop in face.loop_indices:
            co=mesh.vertices[mesh.loops[loop].vertex_index].co
            uv.data[loop].uv=((co[a]+origin[a])*.25,(co[b]+origin[b])*.25)


def write_glb_rock_factor(path):
    """Blender exports the image but drops MixRGB's constant; encode glTF's factor."""
    blob=path.read_bytes();chunks=[];offset=12
    while offset<len(blob):
        length,kind=struct.unpack_from('<II',blob,offset)
        data=blob[offset+8:offset+8+length]
        if kind==0x4e4f534a:
            model=json.loads(data)
            for material in model['materials']:
                if material.get('name','').startswith('Atlantic coastal rock'):
                    material['pbrMetallicRoughness']['baseColorFactor']=list(ROCK_FACTOR)
            data=json.dumps(model,separators=(',',':')).encode()
            data+=b' '*((-len(data))%4)
        chunks.append(struct.pack('<II',len(data),kind)+data);offset+=8+length
    content=b''.join(chunks)
    path.write_bytes(struct.pack('<4sII',b'glTF',2,12+len(content))+content)
