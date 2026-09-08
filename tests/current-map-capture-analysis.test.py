"""Independent CPU fixtures for the read-only map capture evaluator."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('capture_analysis',Path(__file__).resolve().parents[1]/'tools/analyze-current-map-capture.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MapEvaluatorTests(unittest.TestCase):
    def snapshot(self):
        result = module.MapSnapshot.__new__(module.MapSnapshot)
        result.origin = np.zeros(3)
        result.spacing = 1.0
        result.trace_substeps = None
        result.chain = None
        result.dims = np.array([9,9,9])
        result.layout = {'dimensions':[8,8,8]}
        result.bank = 0
        result.banks = [np.zeros((9,9,9,3)),np.zeros((9,9,9,3))]
        result.velocity = np.zeros((9,9,9,3))
        return result

    def test_affine_map_and_full_jacobian(self):
        snapshot = self.snapshot()
        gradient = np.array([[.02,.03,-.01],[-.04,.01,.06],[.01,-.02,.03]])
        translation = np.array([.1,-.2,.3])
        z,y,x = np.indices((9,9,9))
        coords = np.stack([x,y,z],axis=-1)
        snapshot.banks[0] = coords@gradient.T+translation
        point = np.array([3.25,4.1,2.9])
        mapped,jacobian = snapshot.evaluate(point)
        np.testing.assert_allclose(mapped,point+gradient@point+translation,atol=1e-14)
        np.testing.assert_allclose(jacobian,np.eye(3)+gradient,atol=1e-14)

    def test_half_spacing_preserves_physical_jacobian_and_half_fine_velocity_offset(self):
        snapshot = self.snapshot()
        snapshot.spacing = .5
        z,y,x = np.indices((9,9,9))
        coordinates = .5*np.stack([x,y,z],axis=-1)
        gradient = np.array([[.1,.2,0],[0,-.1,.3],[.2,0,.05]])
        snapshot.banks[0] = coordinates@gradient.T
        point = np.array([1.7,2.1,1.9])
        mapped,jacobian = snapshot.evaluate(point)
        np.testing.assert_allclose(mapped,point+gradient@point,atol=1e-14)
        np.testing.assert_allclose(jacobian,np.eye(3)+gradient,atol=1e-14)
        snapshot.velocity = coordinates+.5
        np.testing.assert_allclose(snapshot.frozen_velocity(point),point,atol=1e-14)
        snapshot.banks[1][:] = 0
        snapshot.trace_substeps = 3
        # RK2 for u=x, independently compounded for the stored global schedule.
        dt=.06
        expected=point*(1-dt/3+.5*(dt/3)**2)**3
        np.testing.assert_allclose(snapshot.direct_composition(point,dt),expected,atol=1e-14)

    def test_noncommuting_chain_and_jacobian_order(self):
        snapshot = self.snapshot()
        z,y,x = np.indices((9,9,9))
        points = np.stack([x,y,z],axis=-1)
        first = np.array([[1,.12,0],[0,1,0],[0,0,1]])
        last = np.array([[1,0,0],[.2,1,0],[0,0,1]])
        snapshot.chain = [points@(first-np.eye(3)).T,points@(last-np.eye(3)).T]
        point = np.array([2.2,2.1,3.4])
        mapped,jacobian = snapshot.evaluate(point)
        np.testing.assert_allclose(mapped,first@last@point,atol=1e-14)
        np.testing.assert_allclose(jacobian,first@last,atol=1e-14)
        np.testing.assert_allclose(snapshot.evaluate(point,chain_count=1)[0],first@point,atol=1e-14)

    def test_nonaffine_derivative_matches_independent_finite_differences(self):
        snapshot = self.snapshot()
        snapshot.banks[0] = np.random.default_rng(41).normal(0,.02,(9,9,9,3))
        point = np.array([3.18,4.22,2.76])
        _,analytic = snapshot.evaluate(point)
        numerical = np.column_stack([(snapshot.evaluate(point+np.eye(3)[a]*1e-5)[0]
                                     -snapshot.evaluate(point-np.eye(3)[a]*1e-5)[0])/2e-5 for a in range(3)])
        np.testing.assert_allclose(analytic,numerical,atol=5e-10,rtol=0)

    def test_frozen_velocity_uses_centered_lattice_and_previous_map_bank(self):
        snapshot = self.snapshot()
        z,y,x = np.indices((9,9,9))
        snapshot.velocity = np.stack([.1*x+.05,.2*y+.1,-.1*z-.05],axis=-1)
        point = np.array([3.2,4.1,2.8])
        np.testing.assert_allclose(snapshot.frozen_velocity(point),[.32,.82,-.28],atol=1e-14)
        snapshot.velocity[:] = [.2,-.3,.1]
        snapshot.banks[0][:] = [8,8,8]  # Must not use this accepted output again.
        snapshot.banks[1][:] = [.1,.2,.3]
        np.testing.assert_allclose(snapshot.direct_composition(point,.05),point-.05*np.array([.2,-.3,.1])+[.1,.2,.3],atol=1e-14)


if __name__ == '__main__':
    unittest.main()
