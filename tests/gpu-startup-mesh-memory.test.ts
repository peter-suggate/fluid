import assert from "node:assert/strict";
import test from "node:test";
import {
  interpretSurfaceMeshState, surfaceMeshReserveQuads,
  SVO_SURFACE_MESH_INITIAL_BYTES, SVO_SURFACE_MESH_QUAD_BYTES,
  SVO_SURFACE_MESH_STATE, SVO_SURFACE_MESH_STATE_BYTES, SVO_SURFACE_MESH_MODE,
} from "../lib/svo/features/primary-visibility/svo-surface-mesh";

test("a large scene can outgrow the smaller startup mesh without a capacity failure", () => {
  const words = new Uint32Array(SVO_SURFACE_MESH_STATE_BYTES / 4);
  const W = SVO_SURFACE_MESH_STATE;
  words[W.building] = 1;
  words[W.mode] = SVO_SURFACE_MESH_MODE.initial;
  words[W.errorFlags] = 1;
  words[W.overflowLength] = SVO_SURFACE_MESH_INITIAL_BYTES / SVO_SURFACE_MESH_QUAD_BYTES;
  const receipt = interpretSurfaceMeshState(words, {
    arenaBytes: SVO_SURFACE_MESH_INITIAL_BYTES, maximumBytes: 128 * 1024 * 1024, growable: true,
  });
  assert.equal(receipt.status.state, "pending");
  assert.ok(receipt.grow);
  // The same old receipt cannot trigger another growth after the binding grows.
  assert.equal(interpretSurfaceMeshState(words, {
    arenaBytes: SVO_SURFACE_MESH_INITIAL_BYTES * 2, maximumBytes: 128 * 1024 * 1024, growable: true,
  }).grow, undefined);
  assert.equal(surfaceMeshReserveQuads(1) * SVO_SURFACE_MESH_QUAD_BYTES, 16 * 1024 * 1024);
});
