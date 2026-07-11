import { cameraBasis, dot, length, mulberry32 } from "./math";
import { cloneScene, defaultCamera, defaultScene, parseScene, serializeScene, validateScene } from "./model";

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
  return results;
}
