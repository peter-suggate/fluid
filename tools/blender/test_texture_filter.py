"""Analytic filtering fixtures; runnable in ordinary NumPy Python or Blender."""
import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from texture_filter import TextureMipChain,uv_cell_metric

class TextureFilterTests(unittest.TestCase):
    def test_checker_minification_and_magnification(self):
        checker=np.indices((32,32)).sum(axis=0)%2
        linear=np.repeat(np.where(checker,.36,.04)[:,:,None],3,axis=2)
        texture=TextureMipChain(linear)
        metric=uv_cell_metric([(0,0,0),(2,0,0),(0,4,0)],[(0,0),(1,0),(0,1)],.25)
        self.assertAlmostEqual(texture.lod(metric),2.)
        for uv in [(0,0),(.173,.491),(-.38,2.12),(.5,.5)]:
            np.testing.assert_allclose(texture.sample(uv,texture.lod(metric)),[.2]*3,atol=1e-7)
        np.testing.assert_allclose(texture.sample((.5/32,.5/32),0),[.04]*3,atol=1e-7)
        np.testing.assert_allclose(texture.sample((1.5/32,.5/32),0),[.36]*3,atol=1e-7)
        rough=TextureMipChain(np.repeat(np.where(checker,.8,.2)[:,:,None],3,axis=2))
        np.testing.assert_allclose(rough.sample((.123,.456),2),[.5]*3,atol=1e-7)

    def test_world_uv_footprint_rotation_scaling_shear(self):
        t=TextureMipChain(np.zeros((32,32,3)))
        base=uv_cell_metric([(0,0,0),(1,0,0),(0,1,0)],[(0,0),(1,0),(1,1)],1)
        rotated=uv_cell_metric([(7,2,3),(7,5,3),(7,2,6)],[(0,0),(1,0),(1,1)],3)
        np.testing.assert_allclose(base,rotated,atol=1e-12)
        self.assertAlmostEqual(t.lod(base),math.log2(32*(1+math.sqrt(5))/2))
        self.assertEqual(uv_cell_metric([(0,0,0)]*3,[(0,0)]*3,1),(0.,0.,0.))

    def test_normal_mean_vectors_are_not_renormalized_between_mips(self):
        vectors=np.zeros((4,4,3),dtype=np.float32);vectors[:,:,0]=1
        vectors[:2,:,0]=np.array([1,-1,1,-1]);vectors[:2,:,2]=.1
        vectors/=np.linalg.norm(vectors,axis=-1,keepdims=True)
        texture=TextureMipChain((vectors+1)/2,normal=True)
        expected=vectors.mean(axis=(0,1));expected/=np.linalg.norm(expected)
        np.testing.assert_allclose(texture.sample((.17,.81),2),expected,atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(texture.sample((.17,.81),2))),1.,places=6)
        mirrored=TextureMipChain(np.tile([[[.5,.5,1.]]],(3,5,1)),normal=True)
        np.testing.assert_allclose(mirrored.sample((0,0),10),[0,0,1],atol=1e-6)

    def test_non_power_of_two_area_mean_and_periodic_wrap(self):
        pixels=np.arange(45,dtype=np.float32).reshape(3,5,3)/45
        texture=TextureMipChain(pixels)
        np.testing.assert_allclose(texture.sample((.13,.19),10),pixels.mean(axis=(0,1)),atol=1e-7)
        np.testing.assert_allclose(texture.sample((.13,.19),1.3),texture.sample((2.13,-3.81),1.3),atol=1e-7)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TextureFilterTests))
    if not result.wasSuccessful():raise RuntimeError('Texture footprint fixture failed')
