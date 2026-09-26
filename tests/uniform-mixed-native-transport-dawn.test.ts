import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
import type { FluidRefinementRegion } from "../lib/core/model";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { readMixedTexture, readMixedBuffer } from "./helpers/uniform-mixed-native-fields";
import { geometricSeamRows, transportSeamReference, type Triple } from "./helpers/uniform-geometric-seam";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("graded transport uses native evolved MAC tracing and borrowed field storage", { timeout: 240000 }, async t => {
  await acquireWebGPUExclusiveLock("dawn-test", "Uniform native mixed transport");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });
    const scene = structuredClone(sceneDocument(getSceneDefinition("sparse-cm12-long-dam-break")));
    scene.container.width_m = 1; scene.container.height_m = .4; scene.container.depth_m = .4;
    scene.voxelDomain.finestCellSize_m = .05; scene.nominalResolution.length_m = .05;
    scene.fluid.initialDamBreakDimensions_m = { x: .4, y: .3, z: .4 };
    scene.solidVoxels = [...solidVoxelShellForScene(scene)];
    const region = (maxX: number): FluidRefinementRegion => ({ id: "fine", rule: "minimum-cell-size", minimumCellSize_cells: 1, maximumCellSize_cells: 1,
      min_m: { x: -.5, y: 0, z: -.2 }, max_m: { x: maxX, y: .4, z: .2 } });
    let fineResult: Float32Array | undefined;
    for (const mode of ["native", "fine", "graded", "coarse"] as const) {
      const solver = await WebGPUUniformReferenceSolver.createAsync(device, scene, "balanced", undefined, {
        ...uniformGeometricSolverOptions({}, scene), volumePages: 16, geometricRedistance: false, volumeDustThreshold: 0, orphanDustThreshold: 0,
        pressureCycleBudget: "fixed", pressureCycleDispatch: "direct",
        pressureSchedule: { fullCycles: 1, vCycles: 0, preSweeps: 6, postSweeps: 6, residualTolerance: 0 },
      }, () => {});
      try {
        for (let frame = 1; frame <= 3; frame++) { assert.ok(solver.advanceTo(frame / 30)); await solver.awaitFrameCompletion(); }
        const input = await readMixedTexture(device, solver.volumeTexture);
        const inputPhi = await readMixedTexture(device, solver.vertexPhiTexture!);
        // Dust flooring is disabled for exact transport comparison. Native
        // sharpening can leave signed cancellation residue in the input.
        assert.ok(input.every(v => Number.isFinite(v) && v >= -1e-7));
        await solver.prepareMixedTransportForQA(mode === "coarse" ? [] : [region(mode === "graded" ? -.3 : .5)]);
        const capture = solver.captureMixedTransportForQA(1 / 30, mode === "native");
        const output = await readMixedTexture(device, capture.volume), departures = await readMixedTexture(device, capture.departures);
        const [nx, ny, nz] = capture.layout.lattice.dimensions;
        const index = (p: Triple) => p[0] + nx * (p[1] + ny * p[2]);
        const cells = geometricSeamRows(capture.layout, () => [0, 0, 0]).cells;
        const mixedVelocity=await readMixedTexture(device,capture.velocity),fineVelocity=await readMixedTexture(device,capture.fineVelocity);
        const haloIndex=(q:readonly number[])=>q[0]!+(nx+2)*(q[1]!+(ny+2)*q[2]!);
        let checkedFaces=0;
        for(const c of cells)for(let axis=0;axis<3;axis++){
          const plane=c.min[axis]!+c.width,u=(axis+1)%3,v=(axis+2)%3;
          const adjacent=cells.filter(other=>other.min[axis]===plane
            &&other.min[u]!<c.min[u]!+c.width&&other.min[u]!+other.width>c.min[u]!
            &&other.min[v]!<c.min[v]!+c.width&&other.min[v]!+other.width>c.min[v]!);
          const patches=adjacent.length?adjacent.map(other=>({width:Math.min(c.width,other.width),u:Math.max(c.min[u]!,other.min[u]!),v:Math.max(c.min[v]!,other.min[v]!)}))
            :[{width:c.width,u:c.min[u]!,v:c.min[v]!}];
          for(const patch of patches){
            const anchor=[...c.min];anchor[axis]=plane-1;anchor[u]=patch.u;anchor[v]=patch.v;
            let expectedVelocity=0;
            for(let y=0;y<patch.width;y++)for(let x=0;x<patch.width;x++){
              const q=anchor.map(n=>n+1);q[u]!+=x;q[v]!+=y;expectedVelocity+=fineVelocity[4*haloIndex(q)+axis]!;
            }
            expectedVelocity/=patch.width**2;
            const actualVelocity=mixedVelocity[4*index(anchor as unknown as Triple)+axis]!;
            assert.ok(Number.isFinite(actualVelocity)&&Math.abs(actualVelocity-expectedVelocity)<1e-6,`${mode}: canonical velocity mismatch at ${anchor}/${axis}`);
            checkedFaces++;
          }
        }
        const negativeVelocity=await readMixedBuffer(device,capture.boundaryVelocity);
        for(const c of cells)for(let axis=0;axis<3;axis++)if(c.min[axis]===0){
          const u=(axis+1)%3,v=(axis+2)%3;let expectedVelocity=0;
          for(let y=0;y<c.width;y++)for(let x=0;x<c.width;x++){
            const q=c.min.map(n=>n+1);q[axis]=0;q[u]!+=x;q[v]!+=y;expectedVelocity+=fineVelocity[4*haloIndex(q)+axis]!;
          }
          expectedVelocity/=c.width**2;
          const [x,y,z]=c.min,at=axis===0?y+ny*z:axis===1?ny*nz+x+nx*z:ny*nz+nx*nz+x+nx*y;
          assert.ok(Math.abs(negativeVelocity[at]!-expectedVelocity)<1e-6,`${mode}: negative boundary velocity mismatch`);
        }
        assert.ok(checkedFaces>0);
        const traces = new Map<string, Triple>();
        for (const c of cells) {
          const center = c.min.map(v => v + c.width / 2) as unknown as Triple;
          traces.set(center.join(","), center.map((v, a) => v - departures[index(c.min) * 4 + a]!) as unknown as Triple);
        }
        assert.ok([...traces.values()].some(delta => delta.some(v => Math.abs(v) > 1e-4)), "trace samples moving evolved water");
        const rows = geometricSeamRows(capture.layout, center => traces.get(center.join(","))!);
        const mass = Float32Array.from(cells, c => {
          let value = 0;
          for (let z = 0; z < c.width; z++) for (let y = 0; y < c.width; y++) for (let x = 0; x < c.width; x++) value += input[index([c.min[0] + x, c.min[1] + y, c.min[2] + z])]!;
          return value;
        });
        const expected = transportSeamReference(rows, mass);
        const actual = Float32Array.from(cells, c => output[index(c.min)]! * c.capacity);
        for (let i = 0; i < actual.length; i++) assert.ok(Number.isFinite(actual[i]) && actual[i]! >= -1e-7 && Math.abs(actual[i]! - expected[i]!) < 3e-5 * Math.max(1, expected[i]!), `${mode} cell ${i}: ${actual[i]} vs ${expected[i]}`);
        const sum = (a: Float32Array) => a.reduce((n, v) => n + v, 0);
        assert.ok(Math.abs(sum(actual) - sum(mass)) < 2e-6 * sum(mass), `${mode} conserved volume`);
        t.diagnostic(`${mode}: ${cells.length} owners, max volume error ${Math.max(...actual.map((v, i) => Math.abs(v - expected[i]!)))}, relative mass error ${Math.abs(sum(actual) - sum(mass)) / sum(mass)}, added buffers ${capture.allocatedBytes} bytes`);
        assert.equal(capture.allocatedBytes, capture.layout.tiles.length * 12 + 16 + 36 * 16);
        assert.ok(capture.allocatedBytes < solver.info.allocatedBytes * .03);
        if (mode === "native") fineResult = output;
        if (mode === "fine") for (let i = 0; i < output.length; i++) assert.ok(Math.abs(output[i]! - fineResult![i]!) < 3e-5, `native endpoint cell ${i}`);
        if (mode === "graded") assert.ok(capture.layout.fineTiles.length && capture.layout.transitionTiles.length && capture.layout.coarseTiles.length);
        if(mode!=="native"){
          const restored=solver.captureMixedProlongationForQA();
          const restoredVolume=await readMixedTexture(device,restored.volume);
          const compactPhi=await readMixedTexture(device,capture.phi),restoredPhi=await readMixedTexture(device,restored.phi);
          assert.ok(restoredPhi.every(Number.isFinite),`${mode}: reverse handoff left invalid phi`);
          for(let z=0;z<=nz;z++)for(let y=0;y<=ny;y++)for(let x=0;x<=nx;x++){
            const p=[x,y,z],at=x+(nx+1)*(y+(ny+1)*z);
            const owner=cells.filter(c=>c.min.every((v,a)=>p[a]!>=v&&p[a]!<=v+c.width)).sort((a,b)=>b.width-a.width)[0]!;
            if(owner.min.every((v,a)=>(p[a]!-v)%owner.width===0)){
              assert.equal(compactPhi[at],inputPhi[at],`${mode}: restriction changed canonical phi`);
              assert.equal(restoredPhi[at],inputPhi[at],`${mode}: return handoff changed canonical phi`);
            }
          }
          assert.ok(Math.abs(sum(restoredVolume)-sum(actual))<2e-6*sum(actual),`${mode}: reverse cell handoff changed volume`);
          cells.forEach(c=>{
            const value=output[index(c.min)]!;
            for(let z=0;z<c.width;z++)for(let y=0;y<c.width;y++)for(let x=0;x<c.width;x++)
              assert.equal(restoredVolume[index([c.min[0]+x,c.min[1]+y,c.min[2]+z])],value);
          });
        }else assert.throws(()=>solver.captureMixedProlongationForQA(),/terminal mixed transport result/);
        assert.throws(() => solver.advanceTo(4 / 30), /terminal mixed/);
      } finally { solver.destroy(); }
    }
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
