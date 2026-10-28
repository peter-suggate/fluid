"""FVM projection and combined sRGB-array optical encoding fixtures."""
import sys,unittest,gzip,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from surface_mapping import affine_uv,rgba_mips,srgb_decode
from mesh_sampling import atomic_write

class SurfaceMappingTests(unittest.TestCase):
 def test_atomic_writer_preserves_previous_asset_on_failure(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'asset.fvm.gz';atomic_write(path,b'previous',compressed=True)
   before=path.read_bytes()
   with self.assertRaises(TypeError):atomic_write(path,object(),compressed=True)
   self.assertEqual(path.read_bytes(),before);self.assertEqual(gzip.decompress(before),b'previous')
   self.assertEqual(list(Path(folder).iterdir()),[path])
   atomic_write(path,b'replacement',compressed=True);self.assertEqual(gzip.decompress(path.read_bytes()),b'replacement')
 def test_affine_projection_is_exact_and_normal_invariant(self):
  vertices=np.array([(2,3,4),(4,4,4),(3,5,7)],dtype=float);uvs=np.array([(2,-1),(3,2),(-2,4)],dtype=float)
  u,v,t,sign,n=affine_uv(vertices,uvs)
  for weights in [(1,0,0),(0,1,0),(0,0,1),(.2,.3,.5)]:
   q=np.array(weights)@vertices;expected=np.array(weights)@uvs
   for offset in [-3.,0.,2.]:np.testing.assert_allclose([u[:3]@(q+n*offset)+u[3],v[:3]@(q+n*offset)+v[3]],expected,atol=1e-12)
  self.assertAlmostEqual(np.linalg.norm(t),1);self.assertAlmostEqual(t@n,0);self.assertIn(sign,[-1,1])
 def test_separate_uv_charts_and_degenerate_mapping(self):
  p=[(0,0,0),(1,0,0),(0,1,0)];uv=[(0,0),(1,0),(0,1)]
  first=affine_uv(p,uv);other=affine_uv(p,np.asarray(uv)+[8,-3])
  self.assertEqual(other[0][3]-first[0][3],8);self.assertEqual(other[1][3]-first[1][3],-3)
  self.assertIsNone(affine_uv(p,[(0,0)]*3));self.assertIsNone(affine_uv([(0,0,0)]*3,uv))
 def test_color_roughness_mips_decode_to_linear_area_mean(self):
  checker=np.indices((8,8)).sum(axis=0)%2;rgb=np.repeat(np.where(checker,.36,.04)[...,None],3,axis=2);alpha=np.where(checker,.8,.2)
  data=rgba_mips(rgb,alpha,8);self.assertEqual(len(data),4*(64+16+4+1))
  last=np.frombuffer(data[-4:],np.uint8)/255
  np.testing.assert_allclose(srgb_decode(last[:3]),[.2]*3,atol=.004);self.assertAlmostEqual(last[3],.5,delta=1/255)
 def test_normal_srgb_storage_preserves_signed_mean_before_normalizing(self):
  n=np.zeros((4,4,3));n[:2,:,0]=1;n[2:,:,1]=-1
  data=rgba_mips((n+1)/2,np.ones((4,4)),4,normal=True)
  mean=srgb_decode(np.frombuffer(data[-4:-1],np.uint8)/255)*2-1
  np.testing.assert_allclose(mean,[.5,-.5,0],atol=.009)
  np.testing.assert_allclose(mean/np.linalg.norm(mean),[2**-.5,-2**-.5,0],atol=.012)

suite=unittest.defaultTestLoader.loadTestsFromTestCase(SurfaceMappingTests)
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():raise RuntimeError('FVM surface mapping fixture failed')
