// Run only via tools/gpu/fluid-gpu-run.sh check. Shader ablations are local
// to this probe's device; they never change production files or settings.
import assert from "node:assert/strict";
import { coarseVolumeScene, type CoarseVolumeCase } from "../tests/helpers/uniform-coarse-surface-volume";
import { advanceUniform, createUniformSolver, withUniformDevice } from "../tests/helpers/uniform-geometric";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import { createNbflipVolumeProbe } from "./probe-nbflip-volume";
import type { UniformMixedSurface } from "../lib/methods/uniform/uniform-mixed-surface";
import type { UniformNarrowBandFlip } from "../lib/methods/uniform/uniform-narrow-band-flip";

const kinds = (process.env.O6_CASES ?? "film,slab,resolved-slab,drop").split(",") as CoarseVolumeCase[];
const variants = (process.env.O6_VARIANTS ?? "baseline,no-smoothing,no-redistance,cubic").split(",");
for (const kind of kinds) for (const variant of variants) {
  await withUniformDevice(`O6 ${kind} ${variant}`, async device => {
    let substitutions = 0;
    const noSmoothing = variant.includes("no-smoothing");
    const wrapped = new Proxy(device, { get(target, key) {
      if (key === "createShaderModule") return (descriptor: GPUShaderModuleDescriptor) => {
        let code = descriptor.code;
        if (noSmoothing && code.includes("result.smoothing=umSurfaceSmoothing(")) {
          code = code.replace("result.smoothing=umSurfaceSmoothing(vec3i(vertex),wi,c,t,closed,params.hDt.xyz,eps);", "result.smoothing=0.0;");
          substitutions++;
        }
        if (variant.includes("no-fast")) {
          code = code.replaceAll("fn umCheapAdvected(vertex:vec3u,width:u32)->vec2f{", "fn umCheapAdvected(vertex:vec3u,width:u32)->vec2f{if(params.hDt.w>=0.0){return vec2f(0.0);}")
            .replaceAll("fn umEulerAdvected(vertex:vec3u,width:u32,foot:UMFoot)->vec2f{", "fn umEulerAdvected(vertex:vec3u,width:u32,foot:UMFoot)->vec2f{if(params.hDt.w>=0.0){return vec2f(0.0);}");
        }
        if (variant.includes("unlimited") && code.includes("fn umCubicPhi(")) code = code.replace("return clamp(value,low,high);", "return value;");
        if (variant.includes("no-coupling")) code = code.replace(
          "let bulk=particleSurfaceBulk(owner.tile,vec3i(origin),vec3i(q),false);var value=bulk.x;",
          "let bulk=vec3f(bandPhi(vec3f(q)),0.0,0.0);var value=bulk.x;");
        return target.createShaderModule({ ...descriptor, code });
      };
      const value = Reflect.get(target, key, target);
      return typeof value === "function" ? value.bind(target) : value;
    } });
    const rest = process.env.O6_REST === "1";
    const solver = await createUniformSolver(wrapped, coarseVolumeScene(kind, variant !== "fine", rest), {
      volumeControlSeconds: 0, ...(variant.includes("cubic") ? { surfaceAdvection: "best" } : {}),
    });
    try {
      if (noSmoothing) assert.ok(substitutions > 0, "smoothing ablation matched");
      const frame = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame;
      if (variant.includes("no-redistance")) {
        const advance = frame.advance.bind(frame);
        frame.advance = (p, ...args) => advance({ ...p, redistance: false }, ...args);
      }
      const rows = [];
      const stageStep = Number(process.env.O6_STAGE_STEP ?? 0);
      const detail = frame as unknown as { fields: { phiScratch: GPUTexture }; surface: UniformMixedSurface; narrowBandFlip: UniformNarrowBandFlip };
      const live = await createNbflipVolumeProbe(device, solver, { gauge: true });
      const initialPhiVolume = (await live.read()).reduce((a, b) => a + b, 0);
      const allocatedBytes = solver.info.allocatedBytes;
      const scratch = stageStep ? await createNbflipVolumeProbe(device, solver, { gauge: true, phi: detail.fields.phiScratch }) : undefined;
      const captures: { name: string; buffer: GPUBuffer }[] = [];
      let currentStep = 0;
      const capture = (name: string, encoder: GPUCommandEncoder, probe = live) => {
        if (currentStep !== stageStep || !probe) return;
        probe.encode(encoder);
        const buffer = device.createBuffer({ size: probe.output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        encoder.copyBufferToBuffer(probe.output, 0, buffer, 0, buffer.size);
        captures.push({ name, buffer });
      };
      if (stageStep) {
        const encode = detail.surface.encode.bind(detail.surface);
        detail.surface.encode = (encoder, group) => {
          capture("before-advection", encoder);
          const traced = new Proxy(encoder, { get(target, key) {
            if (key === "beginComputePass") return (descriptor?: GPUComputePassDescriptor) => {
              const pass = target.beginComputePass(descriptor);
              return new Proxy(pass, { get(target, key) {
                if (key === "end") return () => { target.end(); capture(descriptor?.label ?? "surface-pass", encoder, scratch); };
                const value = Reflect.get(target,key,target);return typeof value === "function" ? value.bind(target) : value;
              } });
            };
            const value = Reflect.get(target,key,target);return typeof value === "function" ? value.bind(target) : value;
          } });
          encode(traced, group);
        };
        for (const name of ["reconstruct", "clearAir", "retire", "redistance"] as const) {
          const original = detail.narrowBandFlip[name].bind(detail.narrowBandFlip);
          detail.narrowBandFlip[name] = encoder => {
            if (name === "reconstruct") capture("before-reconstruct-resolved-scratch",encoder,scratch);
            original(encoder); capture(name, encoder);
          };
        }
      }
      const steps = Number(process.env.O6_STEPS ?? (kind === "film" ? 60 : kind.includes("drop") ? 12 : 30));
      for (let step = 0; step <= steps; step++) {
        currentStep = step;
        if (step) await advanceUniform(solver, step / 60);
        const stats = await solver.readStats();
        rows.push({ step, volume: stats.volumeCellSum, speed: stats.maxSpeed_m_s, fine: stats.uniformMixedFineTiles, particles: detail.narrowBandFlip.count });
      }
      console.log(JSON.stringify({ kind, variant, rest, initialPhiVolume, allocatedBytes, drift: rows.at(-1)!.volume! / initialPhiVolume - 1, rows }));
      let previous: Float32Array | undefined;
      for (const { name, buffer } of captures) {
        await buffer.mapAsync(GPUMapMode.READ);
        const cells = new Float32Array(buffer.getMappedRange().slice(0));
        const changes = previous ? Array.from(cells, (v, i) => ({ i, delta: v - previous![i]! })).filter(c => Math.abs(c.delta) > 1e-5).sort((a, b) => a.delta - b.delta) : [];
        console.log(JSON.stringify({ stage: name, step: stageStep, kind, variant, volume: cells.reduce((a, b) => a + b, 0), changed: changes.length,
          losses: changes.slice(0, 12).map(c => ({ ...c, xyz: [c.i % solver.info.nx, Math.floor(c.i / solver.info.nx) % solver.info.ny, Math.floor(c.i / (solver.info.nx * solver.info.ny))] })) }));
        previous = cells; buffer.unmap(); buffer.destroy();
      }
      live?.destroy(); scratch?.destroy();
    } finally { solver.destroy(); }
  });
}
