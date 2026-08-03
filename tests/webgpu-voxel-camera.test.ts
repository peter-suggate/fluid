import assert from "node:assert/strict";
import test from "node:test";
import { defaultCamera } from "../lib/model";
import { voxelViewProjectionMatrix } from "../lib/webgpu-renderer";
import {
  CAMERA_TAN_HALF_FOV,
  cameraApertureShaderLibrary,
  cameraTanHalfFov,
  tanHalfFovFor35mmFocalLength,
  viewportRay,
} from "../lib/webgpu-camera";
import { cameraBasis } from "../lib/math";

function transform(matrix: Float32Array, point: readonly [number, number, number, number]) {
  return [0, 1, 2, 3].map((row) => matrix[row] * point[0] + matrix[4 + row] * point[1] + matrix[8 + row] * point[2] + matrix[12 + row] * point[3]);
}

test("voxel view projection centers the camera target in WebGPU clip space", () => {
  const matrix = voxelViewProjectionMatrix(defaultCamera, 16 / 9, 0.01, 100);
  const target = defaultCamera.target_m;
  const clip = transform(matrix, [target.x, target.y, target.z, 1]);
  assert.ok(clip[3] > 0);
  assert.ok(Math.abs(clip[0] / clip[3]) < 1e-6);
  assert.ok(Math.abs(clip[1] / clip[3]) < 1e-6);
  assert.ok(clip[2] / clip[3] > 0 && clip[2] / clip[3] < 1);
});

test("voxel projection exactly matches the hybrid ray camera away from screen center", () => {
  const aspect = 16 / 9;
  const matrix = voxelViewProjectionMatrix(defaultCamera, aspect, 0.01, 100);
  const basis = cameraBasis(defaultCamera);
  const depth = 3.7;
  for (const ndc of [[-.73, .41], [.52, -.67], [.12, .88]] as const) {
    const point: readonly [number, number, number, number] = [
      basis.position.x + basis.forward.x * depth + basis.right.x * ndc[0] * depth * aspect * CAMERA_TAN_HALF_FOV + basis.up.x * ndc[1] * depth * CAMERA_TAN_HALF_FOV,
      basis.position.y + basis.forward.y * depth + basis.right.y * ndc[0] * depth * aspect * CAMERA_TAN_HALF_FOV + basis.up.y * ndc[1] * depth * CAMERA_TAN_HALF_FOV,
      basis.position.z + basis.forward.z * depth + basis.right.z * ndc[0] * depth * aspect * CAMERA_TAN_HALF_FOV + basis.up.z * ndc[1] * depth * CAMERA_TAN_HALF_FOV,
      1,
    ];
    const clip = transform(matrix, point);
    assert.ok(Math.abs(clip[0] / clip[3] - ndc[0]) < 1e-6);
    assert.ok(Math.abs(clip[1] / clip[3] - ndc[1]) < 1e-6);
  }
});

/**
 * The invariance the aperture field exists to preserve.
 *
 * Every golden fingerprint and A/B image in this repo was captured before the
 * camera had a lens to author, so the one property that cannot bend is that a
 * document which says nothing draws exactly what it drew. Bit-for-bit, not
 * within a tolerance: the ray is the same expression evaluated on the same
 * literal, and an `Object.is` failure here means someone changed the default.
 */
test("an unauthored aperture reproduces the historical rays exactly", () => {
  assert.equal(cameraTanHalfFov(defaultCamera), CAMERA_TAN_HALF_FOV);
  assert.equal(cameraTanHalfFov({ tanHalfFov: undefined }), CAMERA_TAN_HALF_FOV);
  const authored = { ...defaultCamera, tanHalfFov: CAMERA_TAN_HALF_FOV };
  for (const [x, y] of [[-.73, .41], [.52, -.67], [0, 0]] as const) {
    const before = viewportRay(defaultCamera, x, y, 16 / 9);
    const after = viewportRay(authored, x, y, 16 / 9);
    assert.deepEqual(after.direction, before.direction);
    assert.deepEqual(after.origin, before.origin);
  }
  // The shader accessor's fallback is the same number, so a host that never
  // learned to fill the lane draws the same frame as one that did.
  assert.match(cameraApertureShaderLibrary(), /select\(0\.72,authored,authored>0\.0\)/);
  // The hero pond's 50 mm, which is the one authored aperture in the repo.
  assert.equal(tanHalfFovFor35mmFocalLength(50), 0.24);
});
