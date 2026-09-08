import assert from "node:assert/strict";
import test from "node:test";
import { createSparseCM12CurrentMapLayout } from "../lib/methods/adaptive-mass/sparse-cm12-current-map.wgsl";
import {
  createSparseCM12CurrentMapVelocityLayout,
  createSparseCM12CurrentMapVelocityWGSL,
  SPARSE_CM12_CURRENT_MAP_VELOCITY_EXTENSION_SWEEPS,
} from "../lib/methods/adaptive-mass/sparse-cm12-current-map-velocity.wgsl";

/** Independent CPU numerical oracle for the bounded harmonic extension.
 * It verifies the fixed-data and Laplace contracts, not GPU dispatch wiring.
 */
function relax(n: number, velocity: Float32Array, valid: Uint8Array, sweeps: number) {
  let source = velocity.slice(), destination = new Float32Array(source.length);
  for (let step = 0; step < sweeps; ++step) {
    for (let z = 0; z < n; ++z) for (let y = 0; y < n; ++y) for (let x = 0; x < n; ++x) {
      const id = x+n*(y+n*z);
      if (valid[id]) { destination.set(source.subarray(3*id,3*id+3),3*id);continue; }
      for (let component = 0; component < 3; ++component) {
        let sum = 0;
        for (const [dx,dy,dz] of [[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]) {
          const q = [x+dx!,y+dy!,z+dz!];
          if (q.some(v => v<0||v>=n)) continue;
          sum = Math.fround(sum+source[3*(q[0]!+n*(q[1]!+n*q[2]!))+component]!);
        }
        destination[3*id+component] = sum/6;
      }
    }
    [source,destination] = [destination,source];
  }
  return source;
}

function boundaryData(n: number, value: (x: number,y: number,z: number)=>readonly number[]) {
  const velocity = new Float32Array(3*n**3), valid = new Uint8Array(n**3);
  for (let z = 0; z < n; ++z) for (let y = 0; y < n; ++y) for (let x = 0; x < n; ++x) {
    if ([x,y,z].every(v => v>0&&v<n-1)) continue;
    const id = x+n*(y+n*z);valid[id] = 1;velocity.set(value(x,y,z),3*id);
  }
  return { velocity,valid };
}

test("velocity extension reuses disjoint map scratch arenas without allocation", () => {
  const map = createSparseCM12CurrentMapLayout(128,[12,8,12],8);
  const layout = createSparseCM12CurrentMapVelocityLayout(map);
  assert.equal(layout.nodeCount,map.nodeCount);
  assert.equal(layout.velocityBaseWords,map.immutableVelocityBaseWords);
  assert.equal(layout.scratchBaseWords,map.scratchBaseWords);
  assert.equal(layout.fixedMaskBaseWords,map.nodalBaseWords);
  assert.equal(SPARSE_CM12_CURRENT_MAP_VELOCITY_EXTENSION_SWEEPS%2,0);
  assert.throws(() => createSparseCM12CurrentMapVelocityLayout({ ...map,scratchBaseWords:map.nodalBaseWords }));
  assert.match(createSparseCM12CurrentMapVelocityWGSL(map),/fn cm12CurrentMapInitializeVelocity/);
});

test("unknown cache nodes recover uniform flow while original samples remain exact", () => {
  const n = 5, { velocity,valid } = boundaryData(n,() => [1.25,-26.15106773376465,.5]);
  const before = new Uint32Array(velocity.buffer), maskBefore = valid.slice();
  const result = relax(n,velocity,valid,SPARSE_CM12_CURRENT_MAP_VELOCITY_EXTENSION_SWEEPS);
  const after = new Uint32Array(result.buffer);
  for (let id = 0; id < n**3; ++id) for (let component = 0; component < 3; ++component) {
    if (valid[id]) assert.equal(after[3*id+component],before[3*id+component]);
    assert.ok(Math.abs(result[3*id+component]!-[1.25,-26.15106773376465,.5][component]!)<2e-5);
  }
  assert.deepEqual(valid,maskBefore,"computed values must not become fixed data");
});

test("valid stationary zeros and moving data produce the harmonic shear between them", () => {
  const n = 5, { velocity,valid } = boundaryData(n,x => [0,-26*x/(n-1),0]);
  const result = relax(n,velocity,valid,SPARSE_CM12_CURRENT_MAP_VELOCITY_EXTENSION_SWEEPS);
  for (let z = 0; z < n; ++z) for (let y = 0; y < n; ++y) for (let x = 0; x < n; ++x) {
    const at = 3*(x+n*(y+n*z));
    assert.equal(result[at],0);assert.equal(result[at+2],0);
    assert.ok(Math.abs(result[at+1]!+26*x/(n-1))<2e-5);
    if (x===0) assert.equal(result[at+1],velocity[at+1],"valid pool zeros cannot inherit the moving boundary");
  }
});

test("unknown velocities continue relaxing instead of freezing their first arrival", () => {
  const n = 5, { velocity,valid } = boundaryData(n,() => [0,-10,0]);
  const first = relax(n,velocity,valid,1), final = relax(n,velocity,valid,64);
  const centre = 3*(2+n*(2+n*2))+1;
  assert.equal(first[centre],0);
  assert.ok(final[centre]! < -9.99999);
  assert.ok(final.every(v => v>=-10&&v<=0),"convex relaxation cannot overshoot its Dirichlet data");
});
