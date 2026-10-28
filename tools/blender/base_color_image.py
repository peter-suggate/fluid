"""Create glTF-compatible LDR base-colour images from linear optical values.

Blender generated-image pixels are encoded file values when packed/exported.
glTF base-colour images are sRGB, so linear procedural values need the transfer
function before packing. Roughness/normal data must not use this helper.
"""
import bpy
import numpy as np


def linear_base_color_image(name, linear_rgba):
    pixels=np.asarray(linear_rgba,dtype=np.float32)
    if pixels.ndim!=3 or pixels.shape[2]!=4 or not np.isfinite(pixels).all():
        raise ValueError('Base colour image requires finite H×W×4 linear RGBA')
    if np.any(pixels<0) or np.any(pixels>1):
        raise ValueError('LDR base colour image requires values in [0,1]')
    encoded=pixels.copy();rgb=pixels[:,:,:3]
    encoded[:,:,:3]=np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055)
    image=bpy.data.images.new(name,width=pixels.shape[1],height=pixels.shape[0])
    image.colorspace_settings.name='sRGB';image.file_format='PNG'
    image.pixels.foreach_set(encoded.ravel());image.pack()
    return image
