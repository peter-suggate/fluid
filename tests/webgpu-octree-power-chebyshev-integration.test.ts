import assert from "node:assert/strict";
import test from "node:test";
import { WebGPUOctreeProjection, resolveOctreePowerProjectionPolicy } from "../lib/webgpu-octree";

test("automatic pressure policy retains Chebyshev for compatibility geometry", () => {
  const importedGeometryPolicy = resolveOctreePowerProjectionPolicy(
    "authoritative", [1, 1, 1], false, 0, true, true, true,
  );
  assert.equal(importedGeometryPolicy.authoritative, false);
  assert.match(importedGeometryPolicy.fallbackReason!, /imported\/seeded geometry/);

  const source = WebGPUOctreeProjection.toString().replace(/\s+/g, "");
  assert.match(source,
    /this\.leafSolver=requested===["']auto["']\?this\.powerPolicy\.authoritative\?["']mgpcg["']:["']chebyshev["']/,
    "auto must select Chebyshev when the scene cannot admit power authority");
  assert.match(source,
    /requested===["']mgpcg["']&&!this\.powerPolicy\.authoritative\?["']chebyshev["']:requested/,
    "an unavailable paper solve must fail closed to the compatibility solver");
});

test("compatibility Chebyshev remains a compact row-parallel solve", () => {
  const encode = WebGPUOctreeProjection.prototype.encode.toString();
  assert.match(encode, /const useChebyshev=this\.leafSolver===["']chebyshev["']/);
  assert.match(encode, /pressure\.dispatchWorkgroupsIndirect\(this\.solveDispatch,0\)/,
    "Chebyshev must remain row-parallel over the compact publication dispatch");
});
