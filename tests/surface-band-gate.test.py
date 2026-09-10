"""WP0 self-test: the gate reproduces the shadow replay's tables.

Runs tools/surface-band-gate.py on the replay capture
artifacts/surface-band-replay/quarter/coarse with the replay's own phi
(arms A and D) and asserts the numbers in
docs/HANDOFF_FLUID_SURFACE_REVIEW_2026-09-08.md, section "Results", within
0.05 mm, plus the replay.json values they were rounded from. Skipped when the
capture is not on this machine (artifacts/ is not versioned).

    python3.11 tests/surface-band-gate.test.py
"""
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / 'artifacts' / 'surface-band-replay' / 'quarter' / 'coarse'
REPLAY = CAPTURE / 'replay-normal'

_spec = importlib.util.spec_from_file_location('surface_band_gate', ROOT / 'tools' / 'surface-band-gate.py')
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

# Handoff "Results" table, steps 1-7: RMS / max in mm.
DOC_CONTOUR = {1: (3.23, 6.4), 2: (3.67, 6.8), 3: (4.19, 7.2), 4: (4.60, 8.4), 5: (4.74, 8.9), 6: (4.84, 9.2), 7: (5.19, 10.1)}
DOC_MESH = {1: (1.85, 3.8), 2: (2.89, 8.3), 3: (12.27, 39.9), 4: (12.19, 27.6), 5: (15.66, 32.9), 6: (15.10, 36.6), 7: (14.31, 37.2)}
DOC_BAND_A = {1: (1.90, 3.3), 2: (1.66, 3.6), 3: (1.57, 5.3), 4: (1.57, 5.2), 5: (1.50, 5.0), 6: (1.88, 8.7), 7: (2.36, 12.3)}
DOC_BAND_D = {1: (2.23, 3.6), 2: (2.34, 4.5), 3: (2.44, 6.1), 4: (2.66, 6.4), 5: (2.79, 6.5), 6: (3.29, 14.0), 7: (3.51, 13.4)}
TOLERANCE_MM = 0.05


@unittest.skipUnless(REPLAY.exists(), f'replay capture not present: {REPLAY}')
class SurfaceBandGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = json.loads((REPLAY / 'replay.json').read_text())
        cls.by_step = {entry['step']: entry for entry in cls.replay['steps']}
        cls.arm_a = gate.evaluate(CAPTURE, steps=range(1, 8), band='replay-normal/phi-A')
        cls.records_a = {r['step']: r for r in cls.arm_a['steps']}

    def assert_pair(self, record, prefix, expected, label):
        rms, maximum = expected
        self.assertAlmostEqual(record[f'{prefix}.sphere.rms_mm'], rms, delta=TOLERANCE_MM, msg=f'{label} rms')
        self.assertAlmostEqual(record[f'{prefix}.sphere.max_mm'], maximum, delta=TOLERANCE_MM, msg=f'{label} max')

    def test_band_arm_a_matches_the_handoff_table_and_replay_json(self):
        for step, expected in DOC_BAND_A.items():
            record = self.records_a[step]
            self.assert_pair(record, 'band', expected, f'A step {step}')
            error = self.by_step[step]['arms']['A']['surface']['sphere']['radialError_m']
            self.assertAlmostEqual(record['band.sphere.rms_mm'], 1e3 * error['rms'], delta=1e-3)
            self.assertAlmostEqual(record['band.sphere.max_mm'], 1e3 * error['maxAbs'], delta=1e-3)
            pool = self.by_step[step]['arms']['A']['surface']['pool']['heightError_m']
            self.assertAlmostEqual(record['band.pool.max_mm'], 1e3 * pool['maxAbs'], delta=1e-3)

    def test_native_contour_and_mesh_columns(self):
        for step in range(1, 8):
            record = self.records_a[step]
            self.assert_pair(record, 'contour', DOC_CONTOUR[step], f'contour step {step}')
            self.assert_pair(record, 'mesh', DOC_MESH[step], f'mesh step {step}')
            contour = self.by_step[step]['solverNativeTrilinear']['sphere']['radialError_m']
            self.assertAlmostEqual(record['contour.sphere.rms_mm'], 1e3 * contour['rms'], delta=1e-3)
            mesh = self.by_step[step]['shippingMesh']['radialError_m']
            self.assertAlmostEqual(record['mesh.sphere.max_mm'], 1e3 * mesh['maxAbs'], delta=1e-3)

    def test_solver_against_reference_matches_replay_json(self):
        for step in range(1, 8):
            record = self.records_a[step]
            solver = self.by_step[step]['solverVsReference']
            self.assertAlmostEqual(record['solver.massOutsideReference_ml'], 1e6 * solver['massOutsideReference_m3'], delta=1e-2)
            self.assertAlmostEqual(record['solver.fractionError_max'], solver['fractionError']['maxAbs'], delta=1e-4)
        self.assertGreater(self.records_a[5]['solver.massOutsideReference_ml'], 600, 'the 685 ml smear the handoff cites')

    def test_arm_a_passes_the_pre_contact_limits(self):
        limits = json.loads((ROOT / 'tests' / 'surface-band-limits.json').read_text())
        results, failed = gate.check_limits(self.arm_a, limits, criteria={'C2.4'})
        self.assertEqual(failed, [], json.dumps(failed, indent=1))
        self.assertEqual(len(results), 17, 'every C2.4 rule evaluated: 5 + 5 + 1 + 1 + 5 step checks')

    def test_band_arm_d_mass_agreement_after_impact(self):
        report = gate.evaluate(CAPTURE, steps=[1, 7, 8, 12, 20], band='replay-normal/phi-D')
        records = {r['step']: r for r in report['steps']}
        for step in (1, 7):
            self.assert_pair(records[step], 'band', DOC_BAND_D[step], f'D step {step}')
        for step in (8, 12, 20):
            self.assertLess(abs(records[step]['band.minusSolver_L']), 0.005, f'D step {step}: {records[step]["band.minusSolver_L"]} L')
            replay_value = 1e3 * self.by_step[step]['arms']['D']['bandMinusSolver_m3']
            self.assertAlmostEqual(records[step]['band.minusSolver_L'], replay_value, delta=1e-3)
        results, failed = gate.check_limits(report, json.loads((ROOT / 'tests' / 'surface-band-limits.json').read_text()), criteria={'C3.1'})
        self.assertEqual(failed, [], json.dumps(failed, indent=1))


# WP0 HEAD baseline (ec5af4bb), coarse arm, replayed 9 September: handoff
# "Replay on the HEAD captures". Arm A comes from the default run
# (replay-normal), arm D from the sign-class run (replay-topology).
HEAD_CAPTURE = ROOT / 'artifacts' / 'surface-band' / 'baseline' / 'coarse'
HEAD_REPLAY_A = HEAD_CAPTURE / 'replay-normal'
HEAD_REPLAY_D = HEAD_CAPTURE / 'replay-topology'
DOC_HEAD_A = {1: (1.90, 3.3), 2: (1.66, 3.6), 3: (1.57, 5.3), 4: (1.57, 5.2), 5: (1.50, 5.0), 6: (1.88, 8.7), 7: (2.36, 12.1)}
DOC_HEAD_D = {1: (3.94, 5.3), 2: (3.93, 6.5), 3: (4.04, 7.7), 4: (4.21, 9.1), 5: (4.27, 9.2), 6: (4.54, 14.5), 7: (4.76, 17.1)}


@unittest.skipUnless((HEAD_REPLAY_A / 'replay.json').exists(), f'HEAD replay not present: {HEAD_REPLAY_A}')
class SurfaceBandGateHeadTests(unittest.TestCase):
    """The HEAD oracles: arm A unchanged from the afternoon, arm D with sign classes."""

    def assert_pair(self, record, prefix, expected, label):
        rms, maximum = expected
        self.assertAlmostEqual(record[f'{prefix}.sphere.rms_mm'], rms, delta=TOLERANCE_MM, msg=f'{label} rms')
        self.assertAlmostEqual(record[f'{prefix}.sphere.max_mm'], maximum, delta=TOLERANCE_MM, msg=f'{label} max')

    def test_head_arm_a_is_the_afternoon_band_and_passes_c2_4(self):
        report = gate.evaluate(HEAD_CAPTURE, steps=range(1, 8), band='replay-normal/phi-A')
        records = {r['step']: r for r in report['steps']}
        for step, expected in DOC_HEAD_A.items():
            self.assert_pair(records[step], 'band', expected, f'HEAD A step {step}')
            if step <= 6:  # the afternoon table; step 7's max moved 0.13 mm with the pool rising under the sphere
                self.assert_pair(records[step], 'band', DOC_BAND_A[step], f'afternoon A step {step}')
            self.assertLessEqual(records[step]['band.pool.max_mm'], 0.02 if step <= 6 else 6.0, f'pool step {step}')
        limits = json.loads((ROOT / 'tests' / 'surface-band-limits.json').read_text())
        results, failed = gate.check_limits(report, limits, criteria={'C2.4'})
        self.assertEqual(failed, [], json.dumps(failed, indent=1))

    @unittest.skipUnless((HEAD_REPLAY_D / 'replay.json').exists(), f'HEAD sign-class replay not present: {HEAD_REPLAY_D}')
    def test_head_arm_d_sign_classes_hold_mass_without_fills(self):
        replay = json.loads((HEAD_REPLAY_D / 'replay.json').read_text())
        by_step = {entry['step']: entry for entry in replay['steps']}
        report = gate.evaluate(HEAD_CAPTURE, steps=[1, 2, 3, 4, 5, 6, 7, 8, 12, 20], band='replay-topology/phi-D')
        records = {r['step']: r for r in report['steps']}
        for step, expected in DOC_HEAD_D.items():
            self.assert_pair(records[step], 'band', expected, f'HEAD D step {step}')
            correction = by_step[step]['arms']['D']['correction']
            self.assertEqual((correction['filledCells'], correction['drainedCells']), (0, 0), f'step {step} fills/drains')
            if step <= 5:
                self.assertLessEqual(records[step]['band.pool.max_mm'], 0.02, f'pool step {step}')
        shifts = [abs(v) for step in range(2, 21) for v in by_step[step]['arms']['D']['correction']['bodyShift_mm']]
        self.assertLess(max(shifts), 2.5, f'body shifts after step 1: {max(shifts)} mm')
        self.assertLess(abs(by_step[1]['arms']['D']['correction']['bodyShift_mm'][1] - 2.65), 0.1, 'step-1 sphere shift absorbs the 1.0 L initialisation deficit')
        for step in (1, 6, 8, 12, 20):
            self.assertLess(abs(records[step]['band.minusSolver_L']), 0.005, f'D step {step}: {records[step]["band.minusSolver_L"]} L')
        results, failed = gate.check_limits(report, json.loads((ROOT / 'tests' / 'surface-band-limits.json').read_text()), criteria={'C3.1'})
        self.assertEqual(failed, [], json.dumps(failed, indent=1))


if __name__ == '__main__':
    unittest.main()
