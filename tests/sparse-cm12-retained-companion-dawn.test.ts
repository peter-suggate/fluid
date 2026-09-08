import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { SPARSE_CM12_RETAINED_DENSITY_COMPANION_WGSL } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";
import { evaluateRetainedSceneDensity, evaluateRetainedScenePhi, retainedSceneDensity } from
  "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const live = new Set<GPU>();
interface Probe { readonly phi: number; readonly width: number; readonly a: number; readonly b: number }
const density = ({ phi, width, a, b }: Probe) => a * Math.max(0, Math.min(1, .5 - phi / width)) + b;
const sign = (value: number) => value === 0 ? 0 : Math.sign(value);

(dawnModule ? test : test.skip)("production retained companion has exactly the density's signs and endpoint zeros",
  { timeout: 30_000 }, async t => {
    await acquireWebGPUExclusiveLock("dawn-test", "retained-companion-endpoints");
    let gpu: GPU | undefined, device: GPUDevice | undefined;
    const buffers: GPUBuffer[] = [];
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href); Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
      // Dyadic coefficients, samples and width make this matrix exact in both
      // binary32 and binary64. No epsilon can conceal a false zero or sign.
      const probes: Probe[] = [];
      for (const a of [0, .125, .25, .5, .75, 1]) for (const b of [0, .125, .25, .5, .625, 1])
        for (const relativePhi of [-2, -.5, -.375, -.25, -.125, 0, .125, .25, .375, .5, 2])
          probes.push({ phi: relativePhi / 16, width: 1 / 16, a, b });
      // Nearby representable values must take the strict branches on the
      // correct side; checking only exact .5 would miss a widened tie band.
      const epsilon = 2 ** -22;
      for (const [a, b] of [[.5 - epsilon, 0], [.5 + epsilon, 0],
        [.5, .5 - epsilon], [.5, .5 + epsilon], [0, .5 - epsilon], [0, .5 + epsilon]])
        for (const relativePhi of [-.5, 0, .5])
          probes.push({ phi: relativePhi / 16, width: 1 / 16, a, b });
      const matrixCount = probes.length;
      // Reproduce the independently captured empty point in a half-drained
      // support. A tiny seed mean in that support does not wet its centre.
      const sphere = retainedSceneDensity({ generation: 1, transitionWidth: .05,
        domain: { lower: [-.8, 0, -.8], upper: [.8, 1.6, .8] },
        primitives: [{ kind: "ellipsoid", center: [-.15, .8, 0], radii: [.25, .25, .25] }] });
      const point = [-.175, .675, -.275] as const;
      assert.equal(evaluateRetainedSceneDensity(sphere, point), 0);
      probes.push({ phi: Math.fround(evaluateRetainedScenePhi(sphere, point)), width: sphere.transitionWidth, a: .5, b: 0 });

      const input = new Float32Array(probes.flatMap(p => [p.phi, p.width, p.a, p.b]));
      const source = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const output = device.createBuffer({ size: 4 * probes.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readback = device.createBuffer({ size: 4 * probes.length, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      buffers.push(source, output, readback); device.queue.writeBuffer(source, 0, input);
      const module = device.createShaderModule({ label: "production retained density companion", code: /* wgsl */ `
@group(0) @binding(0) var<storage,read> probes:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> result:array<f32>;
${SPARSE_CM12_RETAINED_DENSITY_COMPANION_WGSL}
@compute @workgroup_size(64)
fn check(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=arrayLength(&probes)){return;}
  let probe=probes[gid.x];
  result[gid.x]=cm12RetainedDensityEvolvedPhi(probe.x,probe.y,probe.zw);
}` });
      const info = await module.getCompilationInfo();
      assert.deepEqual(info.messages.filter(message => message.type === "error"), []);
      const pipeline = await device.createComputePipelineAsync({ label: "retained companion exact density oracle",
        layout: "auto", compute: { module, entryPoint: "check" } });
      const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: source } }, { binding: 1, resource: { buffer: output } },
      ] });
      const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(Math.ceil(probes.length / 64)); pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, output.size); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ); const result = new Float32Array(readback.getMappedRange()).slice(); readback.unmap();

      await t.test("414 exact dry, ramp, saturated, constant, compressed and near-endpoint density cases", () => {
        for (let i = 0; i < matrixCount; i++) {
          const p = probes[i]!, actualDensity = density(p), phi = result[i]!;
          const label = JSON.stringify({ ...p, actualDensity, phi });
          assert.ok(Number.isFinite(phi), label);
          assert.equal(sign(phi), sign(.5 - actualDensity), label);
          assert.equal(phi === 0, actualDensity === .5, label);
          if (p.b === .5 || p.a + p.b === .5)
            assert.equal(phi, p.width * (.5 - actualDensity), label);
        }
      });
      await t.test("true half-density plateaus stay zero and constant supports never divide by zero", () => {
        const plateaus = probes.slice(0, matrixCount).map((p, i) => ({ p, i })).filter(({ p }) => density(p) === .5);
        assert.ok(plateaus.length > 20);
        for (const { i } of plateaus) assert.equal(result[i], 0);
        for (let i = 0; i < matrixCount; i++) if (probes[i]!.a === 0) assert.ok(Number.isFinite(result[i]));
      });
      await t.test("interior inverse keeps the seed's analytic companion unchanged at a=1,b=0", () => {
        for (let i = 0; i < matrixCount; i++) if (probes[i]!.a === 1 && probes[i]!.b === 0)
          assert.equal(result[i], probes[i]!.phi);
      });
      await t.test("captured half-drained sphere support's empty centre is strictly outside", () => {
        assert.equal(density(probes[matrixCount]!), 0);
        assert.equal(result[matrixCount], Math.fround(sphere.transitionWidth * .5));
        assert.ok(result[matrixCount]! > 0);
      });
      await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []);
    } finally {
      for (const buffer of buffers) buffer.destroy();
      device?.destroy(); if (gpu) live.delete(gpu); await releaseWebGPUExclusiveLock();
    }
  });
