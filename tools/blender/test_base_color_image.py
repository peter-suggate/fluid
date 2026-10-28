"""Actual Blender → GLB → Blender image transfer regression."""
import sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bpy
import numpy as np
from base_color_image import linear_base_color_image

linear=np.array([[[.001,.003,.18,1],[.045,.13,.5,1]],
                 [[.8,1,0,1],[.02,.04,.12,1]]],dtype=np.float32)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add();obj=bpy.context.object
mat=bpy.data.materials.new('Linear colour transfer fixture');mat.use_nodes=True;obj.data.materials.append(mat)
tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=linear_base_color_image('Linear optical samples',linear)
mat.node_tree.links.new(tex.outputs['Color'],mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
with tempfile.TemporaryDirectory(prefix='fluid-base-colour-') as temporary:
    path=str(Path(temporary)/'linear-colour.glb')
    bpy.ops.export_scene.gltf(filepath=path,export_format='GLB',use_selection=True)
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=path)
    imported=next(o for o in bpy.context.selected_objects if o.type=='MESH').data.materials[0]
    image=next(n.image for n in imported.node_tree.nodes if n.type=='TEX_IMAGE')
    assert image.colorspace_settings.name=='sRGB'
    raw=np.array(image.pixels[:],dtype=np.float32).reshape(2,2,4)[:,:,:3]
    decoded=np.where(raw<=.04045,raw/12.92,((raw+.055)/1.055)**2.4)
    # PNG is8-bit: allow its maximum quantisation error in linear space.
    np.testing.assert_allclose(decoded,linear[:,:,:3],atol=.0045,rtol=0)
    assert abs(float(decoded[0,1,1])-.13)<.002
    print('BASE_COLOR_ROUNDTRIP: linear optical values preserved through GLB')
