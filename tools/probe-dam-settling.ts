/** Headless late-time dam-break diagnostics. Run with WEBGPU_NODE_MODULE set.
 * node --import tsx tools/probe-dam-settling.ts [--steps=360] [--values='{}']
 * Samples physical h-cell velocities and the canonical top surface, without rendering.
 */
import assert from "node:assert/strict";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import { createHighResolutionDamBreakScene } from "../lib/core/scenes";
import { createUniformSolver, withUniformDevice, advanceUniform, readUniformFields } from "../tests/helpers/uniform-geometric";
import { readMixedTexture } from "../tests/helpers/uniform-mixed-native-fields";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";

const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const steps = Number(option("steps") ?? 360), dt = Number(option("dt") ?? 1 / 60);
const values = JSON.parse(option("values") ?? "{}");
const switchStep = Number(option("switch-step") ?? 0);
const switchValues = JSON.parse(option("switch-values") ?? "{}");
const scene = createHighResolutionDamBreakScene();
scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
paceDawnEventPolling();
await withUniformDevice("dam settling", async device => {
 const particleDrift = process.argv.includes("--particle-drift");
 const quadraticDrift = process.argv.includes("--quadratic-drift");
 const source = !particleDrift && !quadraticDrift ? device : new Proxy(device, { get(target, key) {
  if (key === "createShaderModule") return (descriptor: GPUShaderModuleDescriptor) => {
   let code = descriptor.code;
   if (code.includes("fn nbAdvect(i:u32)")) {
    assert.ok(code.includes("if(!ballistic){"));
    if (particleDrift) code = code.replace("if(!ballistic){", "if(false){");
    if (quadraticDrift) {
     assert.ok(code.includes("let k=sampleVelocity(x)/h;"));
     code = code.replace("let k=sampleVelocity(x)/h;", "var probe=Particle();probe.position=vec4f(x,0);let k=nbApicSample(probe).velocity.xyz/h;");
    }
   }
   return target.createShaderModule({ ...descriptor, code });
  };
  const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
 } });
 const solver = await createUniformSolver(source, scene, values);
 try {
  const { nx, ny, nz } = solver.info, h = scene.voxelDomain.finestCellSize_m;
  console.log(JSON.stringify({ values, dt, nx, ny, nz }));
  for (let step = 1; step <= steps; step++) {
   if (step === switchStep) solver.applyRuntimeValues(resolveMethodValues(uniformNarrowBandMethod, "balanced", { ...uniformNarrowBandMethod.appDefaults, ...values, ...switchValues, timeStep: "scene" }));
   await advanceUniform(solver, step * dt);
   if (step % 30 !== 0 && step !== steps) continue;
   const stats = await solver.readStats();
   const fields = await readUniformFields(device, solver);
   const velocity = await readMixedTexture(device, solver.velocityTexture);
   const volumeBudget = (solver as unknown as { mixedFrame: { surfaceVolume: { budget: GPUBuffer } } }).mixedFrame.surfaceVolume.budget;
   const budgetRead = device.createBuffer({ size: 36, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
   const budgetEncoder = device.createCommandEncoder(); budgetEncoder.copyBufferToBuffer(volumeBudget, 0, budgetRead, 0, 36); device.queue.submit([budgetEncoder.finish()]);
   await budgetRead.mapAsync(GPUMapMode.READ); const budget = Array.from(new Float32Array(budgetRead.getMappedRange())); budgetRead.unmap(); budgetRead.destroy();
   const occupied = fields.density.reduce((sum, v) => sum + v, 0);
   let energy = 0, count = 0, max = 0, sinks = 0, sinkFlux = 0;
   const at = (x: number, y: number, z: number, a: number) => x < 0 || y < 0 || z < 0 ? 0 : velocity[4 * (x + nx * (y + ny * z)) + a]!;
   for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (fields.widthAt(x, y, z) !== 1) continue;
    const fraction = fields.density[x + nx * (y + ny * z)]!;
    if (fraction > 0.5) {
     for (let a = 0; a < 3; a++) { const u = at(x, y, z, a); energy += u * u; max = Math.max(max, Math.abs(u)); }
     count++;
    }
    const div = (at(x, y, z, 0) - at(x - 1, y, z, 0) + at(x, y, z, 1) - at(x, y - 1, z, 1) + at(x, y, z, 2) - at(x, y, z - 1, 2)) * dt / h;
    if (fraction < 0.5 && div < -1) { sinks++; sinkFlux -= div; }
   }
   const heights = new Float32Array((nx + 1) * (nz + 1));
   for (let z = 0; z <= nz; z++) for (let x = 0; x <= nx; x++) {
    for (let y = ny - 1; y >= 0; y--) {
     const a = fields.vertex(x, y, z), b = fields.vertex(x, y + 1, z);
     if (a <= 0 && b > 0 && fields.vertex(x, Math.max(0, y - 4), z) < 0) { heights[x + (nx + 1) * z] = y - a / (b - a); break; }
    }
   }
   let rough = 0, n = 0;
   for (let z = 4; z < nz - 4; z++) for (let x = 4; x < nx - 4; x++) {
    const i = x + (nx + 1) * z;
    const r = heights[i]! - (heights[i - 2]! + heights[i + 2]! + heights[i - 2 * (nx + 1)]! + heights[i + 2 * (nx + 1)]!) / 4;
    rough += r * r; n++;
   }
   assert.ok(Number.isFinite(energy));
   console.log(JSON.stringify({ step, time: step * dt, rms: Math.sqrt(energy / Math.max(1, count)), max, sinks, sinkFlux,
    rough: Math.sqrt(rough / n), budget, occupied, volume: stats.volumeCellSum, drift: stats.volumeDrift,
    control: solver.info.narrowBandVolumeControlRate_per_s, fine: solver.info.uniformMixedFineTiles,
    particles: solver.narrowBandFlipInfo?.particles }));
  }
 } finally { solver.destroy(); }
});
