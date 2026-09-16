import assert from "node:assert/strict";
import test from "node:test";
import { packAdaptivitySurfaceParameters } from "../packing";
import { sparseCM12ActivityPolicy } from "../policy";

test("surface publication policy preserves physical thresholds and invalidates proofs only on changes", () => {
  const words = new ArrayBuffer(80);
  const f = new Float32Array(words); const u = new Uint32Array(words);
  const policy = sparseCM12ActivityPolicy({freezeTopology:true});
  const signature = packAdaptivitySurfaceParameters(f,u,0,policy,0.1,0.025,8);
  assert.equal(f[0],Math.fround(0.1));
  assert.equal(u[3],0x80000000);
  assert.deepEqual([...f.slice(4,8)],[0.125,0.25,0.5,1]);
  assert.equal(f[18],0);
  packAdaptivitySurfaceParameters(f,u,0,policy,0.1,0.025,8,signature);
  assert.equal(f[18],0);
  // Curvature tolerance shapes the initial atlas and nothing the shader reads,
  // so changing it rebuilds the solver rather than invalidating live proofs.
  packAdaptivitySurfaceParameters(f,u,0,{...policy,curvatureTolerance:0.5},0.1,0.025,8,signature);
  assert.equal(f[18],0);
  // A published proof threshold does invalidate them.
  packAdaptivitySurfaceParameters(f,u,0,
    {...policy,surfaceDisplacementToleranceCells:2},0.1,0.025,8,signature);
  assert.equal(f[18],1);
});
