import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { shapePatches } from "../lib/core/voxel-editor/geometry";
import { sphMethod } from "../lib/methods/sph/method";
import { SphSolver } from "../lib/methods/sph/solver";
import { sphKernelNormalization } from "../lib/methods/sph/parameters";
import { readFloatTexture3D, smokeRenderHybridPresentation } from "../lib/harness/webgpu-smoke-readbacks";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("Traditional SPH physics and shared scene publication", { timeout: 240_000 }, async t => {
  await acquireWebGPUExclusiveLock("dawn-test", "Traditional SPH");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
    const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const base = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
    Object.assign(base.container, { width_m: 1, height_m: 1, depth_m: 1, fillFraction: 0.5, top: "closed", fluidWallMode: "free-slip" });
    base.voxelDomain.finestCellSize_m = 1 / 12; base.rigidBodies = []; base.solidVoxels = []; base.terrain = undefined;
    Object.assign(base.fluid, { initialCondition: "tank-fill", initialLiquidVolumes: [], initialBrickSeeds_m: [], initialHeightField: undefined,
      initialVelocity_m_s: undefined, inflow: undefined, surfaceTension_N_m: 0, dynamicViscosity_Pa_s: 0, gravity_m_s2: { x: 0, y: -9.81, z: 0 } });
    const create = async (scene = base, values = {}) => await sphMethod.createSolverAsync!(device!, scene, "balanced", values, undefined, () => {}) as SphSolver;
    const readBuffer = async (buffer: GPUBuffer) => {
      const target = device!.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const encoder = device!.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, target, 0, buffer.size); device!.queue.submit([encoder.finish()]);
      await target.mapAsync(GPUMapMode.READ); const values = new Float32Array(target.getMappedRange().slice(0)); target.unmap(); target.destroy(); return values;
    };
    const advance = async (solver: SphSolver, target: number) => {
      let advances = 0;
      while (solver.info.submittedTime_s! < target - 1e-9) { assert.ok(++advances < 2000); solver.advanceTo(target, []); await solver.awaitFrameCompletion(); }
      return advances;
    };
    const drop = () => {
      const scene = structuredClone(base); scene.container.fillFraction = 0;
      scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: 0, y: 0.65, z: 0 }, radius_m: 0.18 }]; return scene;
    };
    await t.test("GPU density matches a direct poly6 sum away from boundaries", async () => {
      for (const sampling of [1, 8]) {
      const scene = drop(); scene.fluid.gravity_m_s2.y = 0; const solver = await create(scene, { particlesPerCell: String(sampling) });
      try {
        const particles = await readBuffer(solver.debug.sphParticles), rho = await readBuffer(solver.debug.sphDensity), cell = 1/12;
        const h = sampling === 8 ? cell : 2*cell;
        const norm = 315 / (64 * Math.PI * h**3) * sphKernelNormalization([cell,cell,cell], sampling === 8 ? 2 : 1);
        for (let i = 0; i < particles.length / 8; i += 7) {
          let expected = 0;
          for (let j = 0; j < particles.length / 8; j++) {
            let r2 = 0; for (let a = 0; a < 3; a++) r2 += (particles[i*8+a]-particles[j*8+a])**2;
            expected += scene.fluid.density_kg_m3*cell**3/sampling*norm*Math.max(0,1-r2/h**2)**3;
          }
          assert.ok(Math.abs(rho[2*i]-expected) < 0.002, `${rho[2*i]} vs ${expected}`);
        }
      } finally { solver.destroy(); }
      }
    });
    await t.test("free fall follows analytic momentum and conserves material", async () => {
      const solver = await create(drop());
      try {
        await advance(solver, 0.05); const particles = await readBuffer(solver.debug.sphParticles);
        let vy=0; for(let i=0;i<particles.length;i+=8) vy+=particles[i+5]; vy/=particles.length/8;
        assert.ok(Math.abs(vy+9.81*0.05)<0.001, `mean vy ${vy}`);
        assert.ok(Math.abs(solver.info.sphMaterialDrift!)<1e-6); assert.ok(solver.info.lastSubsteps!>1);
      } finally { solver.destroy(); }
    });
    await t.test("compressed particles have symmetric pressure forces matching a CPU reference", async () => {
      const scene = drop(); scene.fluid.gravity_m_s2.y = 0;
      const solver = await create(scene, { particlesPerCell: "8" });
      try {
        solver.applyRuntimeValues({ artificialViscosity: 0 });
        const data = await readBuffer(solver.debug.sphParticles), n = data.length / 8, h = 1/12;
        for (let i=0;i<n;i++) for(let a=0;a<3;a++) { const centre = a===1 ? 0.65 : 0.5; data[8*i+a] = centre + 0.7*(data[8*i+a]-centre); }
        device!.queue.writeBuffer(solver.debug.sphParticles,0,data);
        const mass=scene.fluid.density_kg_m3*h**3/8, coefficient=315/(64*Math.PI*h**3)*sphKernelNormalization([h,h,h]);
        const rho=Array.from({length:n},(_,i)=>{
          let sum=0; for(let j=0;j<n;j++){let r2=0;for(let a=0;a<3;a++)r2+=(data[8*i+a]-data[8*j+a])**2;sum+=mass*coefficient*Math.max(0,1-r2/h**2)**3;}return sum;
        });
        const pressure=rho.map(r=>20**2*Math.max(0,r-scene.fluid.density_kg_m3)); assert.ok(Math.max(...pressure)>0);
        await advance(solver,1e-6); const forces=await readBuffer(solver.debug.sphForces); const total=[0,0,0];
        for(let i=0;i<n;i++){
          const expected=[0,0,0];
          for(let j=0;j<n;j++)if(j!==i){const d=[0,1,2].map(a=>data[8*i+a]-data[8*j+a]), r=Math.hypot(...d);if(r>=h||r===0)continue;
            const scale=mass*(pressure[i]+pressure[j])/(2*rho[i]*rho[j])*45/(Math.PI*h**6)*(h-r)**2/r;
            for(let a=0;a<3;a++)expected[a]+=scale*d[a];
          }
          for(let a=0;a<3;a++){assert.ok(Math.abs(forces[4*i+a]-expected[a]) < 0.02+Math.abs(expected[a])*1e-4);total[a]+=forces[4*i+a];}
        }
        assert.ok(total.every(v=>Math.abs(v/n)<0.001), `net acceleration ${total.map(v=>v/n)}`);
      } finally { solver.destroy(); }
    });
    await t.test("physical viscosity dissipates shear and surface tension stays finite", async () => {
      const energies: number[]=[];
      for(const viscosity of [0,10]){
        const scene=drop();scene.fluid.gravity_m_s2.y=0;scene.fluid.dynamicViscosity_Pa_s=viscosity;
        const solver=await create(scene);
        try {
          solver.applyRuntimeValues({artificialViscosity:0});
          const data=await readBuffer(solver.debug.sphParticles);
          for(let i=0;i<data.length;i+=8)data[i+4]=data[i+1]>0.65 ? 0.1 : -0.1;
          device!.queue.writeBuffer(solver.debug.sphParticles,0,data);await advance(solver,0.01);energies.push(solver.info.sphKineticEnergy_J!);
        }finally{solver.destroy();}
      }
      assert.ok(energies[1]<0.95*energies[0], `viscous energy ${energies}`);
      const scene=drop();scene.fluid.surfaceTension_N_m=0.072;const solver=await create(scene);
      try{await advance(solver,0.02);assert.ok((await readBuffer(solver.debug.sphParticles)).every(Number.isFinite));assert.ok(Math.abs(solver.info.sphMaterialDrift!)<1e-6);}finally{solver.destroy();}
    });
    await t.test("hydrostatic pool remains bounded with modest density compression", async () => {
      const solver = await create();
      try {
        await advance(solver, 0.2);
        assert.ok(solver.info.maxSpeed_m_s! < 0.5, `rest speed ${solver.info.maxSpeed_m_s}`);
        assert.ok(solver.info.sphMaxCompression! < 0.15, `compression ${solver.info.sphMaxCompression}`);
        assert.ok(Math.abs(solver.info.sphMaterialDrift!) < 1e-6);
      } finally { solver.destroy(); }
    });
    await t.test("authored dam advances through impact with finite particles", async () => {
      const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32"))), solver = await create(scene), start = performance.now();
      try {
        const advances = await advance(solver, 0.5);
        assert.ok(solver.info.maxSpeed_m_s! > 0.1); assert.ok(solver.info.maxSpeed_m_s! < 10);
        assert.ok(Math.abs(solver.info.sphMaterialDrift!) < 1e-6); assert.ok(solver.info.sphMaxCompression! < 0.3, `dam compression ${solver.info.sphMaxCompression}, speed ${solver.info.maxSpeed_m_s}`);
        const particles = await readBuffer(solver.debug.sphParticles); assert.ok(particles.every(Number.isFinite));
        t.diagnostic(`SPH dam: ${advances} advances, ${((performance.now()-start)/advances).toFixed(2)} ms/advance including fence, ${solver.info.sphParticleCount} particles, ${solver.info.allocatedBytes} bytes`);
      } finally { solver.destroy(); }
    });
    await t.test("fixed voxel barriers stop fast particles without losing material", async () => {
      const scene=drop();scene.fluid.gravity_m_s2.y=0;scene.fluid.initialVelocity_m_s={x:4,y:0,z:0};
      scene.solidVoxels=shapePatches([8,0,0],[9,12,12],"fill","box");
      const solver=await create(scene);
      try{
        await advance(solver,0.15);const data=await readBuffer(solver.debug.sphParticles);
        for(let i=0;i<data.length;i+=8)assert.ok(data[i]<8/12, `particle crossed wall at ${data[i]}`);
        assert.ok(Math.abs(solver.info.sphMaterialDrift!)<1e-6);
      }finally{solver.destroy();}
    });
    await t.test("empty scenes and scenes with ignored rigid/source features open", async () => {
      const scene = structuredClone(base); scene.container.fillFraction = 0; scene.fluid.inflow = { enabled: true } as NonNullable<typeof scene.fluid.inflow>;
      scene.rigidBodies = [{}] as typeof scene.rigidBodies; const solver = await create(scene);
      try {
        solver.advanceTo(0.001, [{}] as Parameters<SphSolver["advanceTo"]>[1]); await solver.awaitFrameCompletion();
        assert.equal(solver.info.sphParticleCount, 0); assert.equal(solver.info.volumeCellSum, 0); assert.equal(solver.info.sphMaterialDrift, 0);
      } finally { solver.destroy(); }
    });
    await t.test("live dt and stiffness edits preserve particles and the running clock", async () => {
      const scene = drop(); scene.fluid.gravity_m_s2.y=0; const solver = await create(scene);
      try {
        const particles = solver.debug.sphParticles;
        await advance(solver, 0.01); scene.numerics.fixedDt_s = 0.02; solver.applySceneUniforms(scene); solver.applyRuntimeValues({ soundSpeed: 20 });
        await advance(solver, 0.03); assert.equal(solver.debug.sphParticles, particles); assert.ok(Math.abs(solver.info.completedTime_s!-0.03)<1e-9);
        assert.ok(Math.abs(solver.info.sphMaterialDrift!)<1e-6);
      } finally { solver.destroy(); }
    });
    await t.test("open-top outflow has an explicit material ledger", async () => {
      const scene = drop(); scene.container.top="open"; scene.fluid.gravity_m_s2.y=0; scene.fluid.initialVelocity_m_s={x:0,y:2,z:0};
      const solver = await create(scene);
      try { await advance(solver,0.25); assert.ok(solver.info.sphEscapedVolume_m3!>0); assert.ok(Math.abs(solver.info.sphMaterialDrift!)<1e-6); }
      finally { solver.destroy(); }
    });
    await t.test("failed particle health retains the last valid renderer publication", async () => {
      const solver = await create();
      try {
        const before = await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 13,13,13);
        device!.queue.writeBuffer(solver.debug.sphParticles, 0, new Float32Array([NaN])); solver.advanceTo(0.01, []);
        const encoder=device!.createCommandEncoder();const health=solver.captureSimulationHealth(encoder);device!.queue.submit([encoder.finish()]);
        await assert.rejects(health(), /non-finite/);
        await assert.rejects(solver.awaitFrameCompletion(), /non-finite/);
        assert.deepEqual(await readFloatTexture3D(device!, solver.denseLevelSetVolumeSource.vertexPhi, 13,13,13), before);
        assert.equal(solver.info.completedTime_s, 0);
      } finally { solver.destroy(); }
    });
    await t.test("the shared renderer consumes the SPH water surface", async () => {
      const solver = await create();
      try {
        await advance(solver,0.01); const rendered = await smokeRenderHybridPresentation(device!,solver,base,[]);
        assert.ok(rendered.frontInterfacePixels>0); assert.ok(rendered.pairedInterfacePixels>0);
        assert.equal(rendered.rendererValidationErrorCount,0); assert.equal(rendered.rendererUncapturedErrorCount,0);
      } finally { solver.destroy(); }
    });
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
