"""CPU-only density-authority ray QA fixtures; no rendering or GPU required."""
import importlib.util
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np


spec = importlib.util.spec_from_file_location('ray_field', Path(__file__).resolve().parents[1]/'tools/analyze-current-map-ray-field.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def configuration():
    return dict(h=.5, origin=[-4, 0, -4], dimensions=[16, 16, 16], dt=.025,
                scene=dict(container=dict(width_m=8, height_m=8, depth_m=8, fillFraction=.125),
                           voxelDomain=dict(finestCellSize_m=.5), rigidBodies=[],
                           fluid=dict(initialCondition='tank-fill', gravity_m_s2=dict(x=0, y=-9.8, z=0),
                                      initialLiquidVolumes=[dict(shape='sphere', center_m=dict(x=0, y=4, z=0), radius_m=1)])))


def snapshot(gradient=None, translation=None):
    result = module.MappedSnapshot.__new__(module.MappedSnapshot)
    result.origin = np.array([-2., -2., -2.])
    result.spacing = .5
    result.dims = np.array([41, 41, 41])
    result.bank = 0
    result.chain = None
    z, y, x = np.indices(tuple(result.dims[::-1]))
    points = result.origin+result.spacing*np.stack([x, y, z], axis=-1)
    gradient = np.zeros((3, 3)) if gradient is None else gradient
    translation = np.zeros(3) if translation is None else translation
    result.banks = [points@gradient.T+translation, np.zeros(points.shape)]
    return result


class RayFieldTests(unittest.TestCase):
    def test_batch_evaluator_matches_independent_scalar_chain(self):
        first = np.array([[0, .04, 0], [0, 0, .03], [0, 0, 0]])
        last = np.array([[0, 0, 0], [.02, 0, 0], [0, 0, 0]])
        captured = snapshot(first, [.01, -.02, .03])
        captured.chain = [captured.banks[0], snapshot(last).banks[0]]
        points = np.array([[3.2, 4.1, 2.3], [4.25, 3.73, 5.1], [-2.1, 0, 0], [0, -2, 0]])
        actual_points, actual_jacobians = module.evaluate_many(captured, points)
        scalar = [captured.evaluate(point) for point in points]
        np.testing.assert_allclose(actual_points, [p[0] for p in scalar], atol=2e-14, rtol=0)
        np.testing.assert_allclose(actual_jacobians, [p[1] for p in scalar], atol=2e-14, rtol=0)

    def test_candidate_batch_composes_new_increment_before_accepted_chain(self):
        captured = snapshot(np.array([[0, .04, 0], [0, 0, .03], [0, 0, 0]]))
        captured.chain = [captured.banks[0]]
        captured.banks[1] = snapshot(np.array([[0, 0, 0], [.02, 0, 0], [0, 0, 0]]), [.1, -.2, .05]).banks[0]
        points = np.array([[3.2, 4.1, 2.3], [4.25, 3.73, 5.1]])
        actual_points, actual_jacobians = module.evaluate_many(captured, points, bank=1)
        scalar = [captured.evaluate(point, bank=1) for point in points]
        np.testing.assert_allclose(actual_points, [p[0] for p in scalar], atol=2e-14, rtol=0)
        np.testing.assert_allclose(actual_jacobians, [p[1] for p in scalar], atol=2e-14, rtol=0)

    def test_batched_velocity_matches_mirrored_boundary_scalar_queries(self):
        captured = snapshot()
        captured.layout = dict(dimensions=[16, 16, 16])
        captured.velocity = np.zeros(tuple(captured.dims[::-1])+(3,))+[1, 2, 3]
        points = np.array([[-.25, .2, 2], [16.25, 8, 4], [7, -.125, 9], [1, 17, 2]])
        actual = module.frozen_velocity_many(captured, points)
        expected = np.array([captured.frozen_velocity(point) for point in points])
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
        self.assertAlmostEqual(actual[0, 0], -.5)
        self.assertAlmostEqual(actual[2, 1], -.5)

    def test_translated_sphere_has_no_radius_or_trajectory_fit_error(self):
        seed = module.SeedField(configuration())
        displacement = np.array([.13, -.6, -.17])
        captured = snapshot(translation=-displacement/seed.h)
        density = module.DensityAuthority(captured, seed)
        center = np.array([0., 4., 0.])+displacement
        metrics = module.sphere_metrics(density, center, 1, 1, seed.lower, seed.upper,
                                        module.antipodal_directions(38), 33, 1e-9)
        self.assertEqual(metrics['uniqueRoots'], 38)
        self.assertLess(metrics['radialError_m']['maxAbs'], 1e-9)
        self.assertLess(metrics['referenceCenteredChordError_m']['maxAbs'], 2e-9)

    def test_volume_preserving_shear_is_measured_not_fitted_away(self):
        seed = module.SeedField(configuration())
        inverse = np.array([[1, .3, 0], [0, 1, 0], [0, 0, 1.]])
        center = np.array([0., 4., 0.])
        center_fine = (center-seed.origin)/seed.h
        captured = snapshot(inverse-np.eye(3), (np.eye(3)-inverse)@center_fine)
        density = module.DensityAuthority(captured, seed)
        directions = module.antipodal_directions(38)
        metrics = module.sphere_metrics(density, center, 1, 1, seed.lower, seed.upper, directions, 33, 1e-9)
        expected = 1/np.linalg.norm(directions@inverse.T, axis=1)
        np.testing.assert_allclose([s['radius_m'] for s in metrics['samples']], expected, atol=1e-9, rtol=0)
        self.assertGreater(metrics['radialError_m']['maxAbs'], .1)

    def test_compression_includes_determinant_in_half_density_root(self):
        seed = module.SeedField(configuration())
        inverse = np.diag([1.1, .95, 1.05])
        center = np.array([0., 4., 0.])
        center_fine = (center-seed.origin)/seed.h
        captured = snapshot(inverse-np.eye(3), (np.eye(3)-inverse)@center_fine)
        density = module.DensityAuthority(captured, seed)
        directions = module.antipodal_directions(38)
        metrics = module.sphere_metrics(density, center, 1, 1, seed.lower, seed.upper, directions, 33, 1e-9)
        # q0*J=.5 implies seed radius squared = r²+r*width*(1-1/J).
        expected = np.sqrt(1+seed.width*(1-1/np.linalg.det(inverse)))/np.linalg.norm(directions@inverse.T, axis=1)
        np.testing.assert_allclose([s['radius_m'] for s in metrics['samples']], expected, atol=1e-9, rtol=0)

    def test_stationary_pool_height_roots_remain_flat_through_wall_inset(self):
        seed = module.SeedField(configuration())
        metrics = module.pool_metrics(module.DensityAuthority(snapshot(), seed), 1, [0, 4, 0], 1,
                                      seed.lower, seed.upper, seed.h, 5, 33, 1e-9)
        self.assertEqual(metrics['uniqueRoots'], 25)
        self.assertLess(metrics['heightError_m']['maxAbs'], 1e-9)

    def test_missing_and_multiple_crossings_are_not_silently_selected(self):
        starts = np.zeros((3, 3))
        starts[:, 0] = [0, 1, 2]
        direction = np.tile([0., 1., 0.], (3, 1))

        def field(points):
            x, y = points[:, 0], points[:, 1]
            return np.where(x == 0, 1., np.where(x == 1, .5+.4*np.cos(4*np.pi*y), 1-y))

        roots = module.descending_roots(field, starts, direction, 1, 65, 1e-9)
        np.testing.assert_array_equal(roots['descending_counts'], [0, 2, 1])
        self.assertTrue(np.isnan(roots['roots'][0]))
        self.assertTrue(np.isnan(roots['roots'][1]))
        self.assertAlmostEqual(roots['roots'][2], .5, places=8)

    def test_reference_contact_does_not_create_a_free_fall_comparison(self):
        metrics = module.sphere_metrics(lambda p: np.ones(len(p)), [0, 2, 0], 1, 1,
                                        [-4, 0, -4], [4, 8, 4], module.antipodal_directions(6), 33, 1e-9)
        self.assertIn('contact', metrics['referenceScope'])
        self.assertNotIn('radialError_m', metrics)

    def test_compressed_capture_uses_relocated_chain_count_not_capacity(self):
        count, base = 125, 17
        schedule, chain_base = base+9*count, base+9*count+1
        chain_count = chain_base+3*count  # Archive stores one slot, capacity is 32.
        raw = np.zeros(chain_count-base+1, dtype='<f4')
        raw[schedule-base] = 3
        raw[chain_count-base] = 1
        raw[chain_base-base:chain_count-base].reshape(-1, 3)[:] = [.03, -.04, .02]
        meta = dict(baseWords=base, retainedControl=[1, 0, 1, 0],
                    map=dict(nodeCount=count, nodeDimensions=[5, 5, 5], originFine=[0, 0, 0], spacingFine=1,
                             coefficientBaseWords=[base, base+3*count], immutableVelocityBaseWords=base+6*count,
                             traceSubstepCountBaseWords=schedule, chainBaseWords=chain_base,
                             chainCountBaseWords=chain_count, chainCapacity=32))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory/'current-map.json').write_text(json.dumps(meta))
            (directory/'current-map.bin.gz').write_bytes(gzip.compress(raw.tobytes()))
            captured = module.MappedSnapshot(directory)
            self.assertEqual(captured.trace_substeps, 3)
            self.assertEqual(len(captured.chain), 1)
            points, jacobians = module.evaluate_many(captured, [[2, 2, 2]])
            np.testing.assert_allclose(points, [[2.03, 1.96, 2.02]], atol=1e-8, rtol=0)
            np.testing.assert_allclose(jacobians, [np.eye(3)], atol=1e-14, rtol=0)

    def test_explicit_acceptance_limits_reject_errors_and_incomplete_roots(self):
        report = dict(frames=[dict(step=6, sphere=dict(uniqueRoots=6, sampledRays=6,
                                                       radialError_m=dict(maxAbs=.002)),
                                  pool=dict(uniqueRoots=4, sampledRays=4, heightError_m=dict(maxAbs=.000001)))])
        result = module.check_limits(report, .001, .000002)
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['violations']), 1)
        self.assertEqual(result['violations'][0]['field'], 'sphere')
        report['frames'][0]['sphere']['radialError_m']['maxAbs'] = .0001
        report['frames'][0]['pool']['uniqueRoots'] = 3
        result = module.check_limits(report, .001, .000002)
        self.assertFalse(result['passed'])
        self.assertIn('roots', result['violations'][0]['reason'])


if __name__ == '__main__':
    unittest.main()
