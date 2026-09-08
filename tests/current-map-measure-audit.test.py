"""Independent cell-integration and stored-bank decoding fixtures."""
import gzip
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np


spec = importlib.util.spec_from_file_location('measure_audit', Path(__file__).resolve().parents[1]/'tools/audit-current-map-measure.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MeasureAuditTests(unittest.TestCase):
    def test_quadrature_integrates_independent_mixed_polynomial_on_physical_cells(self):
        cells = np.array([[0, 0, 0], [2, 3, 1], [-2, 1, 4]])
        origin, h = np.array([-.3, .4, .1]), .2

        def density(points):
            return points[:, 0]**7+points[:, 1]**3*points[:, 2]**2

        lo = origin+h*cells
        hi = lo+h
        average = lambda axis, power: (hi[:, axis]**(power+1)-lo[:, axis]**(power+1))/(h*(power+1))
        expected = average(0, 7)+average(1, 3)*average(2, 2)
        actual = module.integrate_cell_means(density, origin, h, cells, 4, point_budget=64)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)

    def test_compact_compressed_capture_selects_accepted_measure_bank(self):
        metadata = dict(baseWords=100, retainedControl=[2, 1, 5, 0], measure=dict(baseWords=103, dimensions=[2, 1, 1]))
        values = np.arange(19, dtype='<f4')
        with tempfile.TemporaryDirectory() as path:
            path = Path(path)
            (path/'current-map.bin.gz').write_bytes(gzip.compress(values.tobytes()))
            actual = module.read_measure(path, metadata)
            np.testing.assert_array_equal(actual.reshape(-1), values[11:19])


if __name__ == '__main__':
    unittest.main()
