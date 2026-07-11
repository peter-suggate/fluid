import { cameraBasis, dot, length, mulberry32 } from "./math";
import { cloneScene, defaultCamera, defaultScene, parseScene, serializeScene, validateScene } from "./model";
import { advanceRigidBodies, initializeRigidBody, massProperties, rigidDiagnostics } from "./rigid-body";

export interface ValidationResult {
  id: string;
  name: string;
  measured: string;
  threshold: string;
  passed: boolean;
}

export function runShellValidation(): ValidationResult[] {
  const results: ValidationResult[] = [];
  const errors = validateScene(defaultScene);
  results.push({ id: "S2-01", name: "Default SI scene", measured: `${errors.length} errors`, threshold: "0 errors", passed: errors.length === 0 });

  const encoded = serializeScene(defaultScene);
  const roundTrip = serializeScene(parseScene(encoded));
  results.push({ id: "S2-02", name: "Scene JSON round-trip", measured: `${encoded.length} bytes`, threshold: "byte-identical", passed: encoded === roundTrip });

  const basis = cameraBasis(defaultCamera);
  const maxDot = Math.max(Math.abs(dot(basis.forward, basis.right)), Math.abs(dot(basis.forward, basis.up)), Math.abs(dot(basis.right, basis.up)));
  const maxNorm = Math.max(Math.abs(length(basis.forward) - 1), Math.abs(length(basis.right) - 1), Math.abs(length(basis.up) - 1));
  const basisError = Math.max(maxDot, maxNorm);
  results.push({ id: "S2-04", name: "Camera orthonormality", measured: basisError.toExponential(2), threshold: "< 1e-10", passed: basisError < 1e-10 });

  const rngA = mulberry32(defaultScene.randomSeed);
  const rngB = mulberry32(defaultScene.randomSeed);
  let mismatch = 0;
  for (let i = 0; i < 1000; i += 1) if (rngA() !== rngB()) mismatch += 1;
  results.push({ id: "S2-05", name: "Seed reproducibility", measured: `${mismatch} mismatches`, threshold: "0 mismatches", passed: mismatch === 0 });

  const invalid = cloneScene(defaultScene);
  invalid.container.width_m = -1;
  invalid.fluid.density_kg_m3 = 0;
  invalid.container.fillFraction = 1.2;
  const invalidErrors = validateScene(invalid);
  results.push({ id: "S2-06", name: "Invalid input rejection", measured: `${invalidErrors.length} rejections`, threshold: ">= 3 rejections", passed: invalidErrors.length >= 3 });

  const sphereDescription = cloneScene(defaultScene).rigidBodies[0];
  sphereDescription.shape = "sphere";
  sphereDescription.dimensions_m = { x: 0.1, y: 0.1, z: 0.1 };
  sphereDescription.density_kg_m3 = 1000;
  const properties = massProperties(sphereDescription);
  const expectedMass = 1000 * 4 * Math.PI * 0.1 ** 3 / 3;
  const massError = Math.abs(properties.mass_kg - expectedMass) / expectedMass;
  results.push({ id: "R3-01", name: "Sphere analytic mass", measured: massError.toExponential(2), threshold: "< 1e-12 relative", passed: massError < 1e-12 });

  const freeFallScene = cloneScene(defaultScene);
  freeFallScene.container.width_m = 20; freeFallScene.container.depth_m = 20; freeFallScene.rigidBodies = [];
  const falling = initializeRigidBody({ ...sphereDescription, position_m: { x: 0, y: 10, z: 0 }, linearVelocity_m_s: { x: 0, y: 0, z: 0 } });
  for (let i = 0; i < 250; i += 1) advanceRigidBodies([falling], freeFallScene, 0.001);
  const expectedY = 10 + 0.5 * defaultScene.fluid.gravity_m_s2.y * 0.25 ** 2;
  const fallError = Math.abs(falling.position_m.y - expectedY) / Math.abs(expectedY);
  results.push({ id: "R3-03", name: "Rigid free fall", measured: `${(fallError * 100).toFixed(4)}%`, threshold: "< 1% position error", passed: fallError < 0.01 });

  const finite = rigidDiagnostics([falling], freeFallScene.fluid.gravity_m_s2).nanCount;
  results.push({ id: "R3-09", name: "Rigid state finite", measured: `${finite} invalid values`, threshold: "0 NaN/∞", passed: finite === 0 });
  return results;
}
