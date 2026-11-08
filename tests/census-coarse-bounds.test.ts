import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { createUniformTroughScene } from "../lib/core/uniform-trough-scenes";
import { UniformMixedDynamicClassifier } from "../lib/methods/uniform/uniform-mixed-dynamic";
import { advanceUniform, createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

// classifyCoarse spreads each 4h owner's face-velocity bounds over the
// workgroup (a lane an (owner, axis, sign) face, then a lane a part of a
// sixteen-part seam face). On a recorded frame, the census it writes (the
// census words, bound cubes and tile records) must be bit-identical to the
// serial reference's (one lane an owner, the faces in series), both launched
// over the same inputs from the same saved outputs (regionList first, which
// lists the regions classifyCoarse takes). The trough dam-break on a
// census start holds 4h tiles beside h ones, so seam faces are in the frame.
const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };

type Classifier = {
  work: GPUBuffer; bounds: GPUBuffer; serialCoarse?: GPUComputePipeline; pipelines: Map<string, GPUComputePipeline>;
  group: { group: GPUBindGroup }; ownership: { bindGroup: GPUBindGroup };
  censusLaunches(): { pipeline: GPUComputePipeline; x: number }[];
};

async function words(device: GPUDevice, buffer: GPUBuffer): Promise<Uint32Array> {
  const read = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, read, 0, buffer.size); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); return new Uint32Array(read.getMappedRange().slice(0));
  } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
}

function firstDifference(a: Uint32Array, b: Uint32Array): string {
  let n = 0, first = -1;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { n++; if (first < 0) first = i; }
  return n ? `${n} words differ, first at ${first}: ${a[first]} vs ${b[first]}` : "";
}

test("classifyCoarse's spread face bounds give the serial loop's census bit for bit", gpu, async () => {
  UniformMixedDynamicClassifier.serialCoarseBounds = true;
  try {
    await withUniformDevice("census coarse bounds", async device => {
      const scene = createUniformTroughScene("dam-break");
      scene.fluid.tiers = { ...scene.fluid.tiers, start: "census" };
      const solver = await createUniformSolver(device, scene, {});
      try {
        const c = (solver as unknown as { mixedDynamic: Classifier }).mixedDynamic;
        assert.ok(c.serialCoarse, "the reference is compiled");
        const spread = c.pipelines.get("classifyCoarse")!, regions = c.pipelines.get("regionList")!;
        const launch = c.censusLaunches().find(l => l.pipeline === spread)!;
        const saved = [c.work, c.bounds].map(b => device.createBuffer({ size: b.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }));
        try {
          for (let step = 1; step <= 3; step++) {
            await advanceUniform(solver, step * scene.numerics.fixedDt_s);
            await solver.awaitFrameCompletion();
            assert.equal(solver.info.simulationPipelineError, undefined);
            assert.ok((solver.info.uniformMixedFineTiles ?? 0) > 0, "the frame holds h tiles beside the 4h ones");
            { const e = device.createCommandEncoder(); e.copyBufferToBuffer(c.work, 0, saved[0]!, 0, c.work.size); e.copyBufferToBuffer(c.bounds, 0, saved[1]!, 0, c.bounds.size); device.queue.submit([e.finish()]); }
            const results: Uint32Array[][] = [];
            for (const pipeline of [spread, c.serialCoarse!]) {
              const e = device.createCommandEncoder();
              e.copyBufferToBuffer(saved[0]!, 0, c.work, 0, c.work.size); e.copyBufferToBuffer(saved[1]!, 0, c.bounds, 0, c.bounds.size);
              const pass = e.beginComputePass(); pass.setBindGroup(0, c.ownership.bindGroup); pass.setBindGroup(1, c.group.group);
              pass.setPipeline(regions); pass.dispatchWorkgroups(1);
              pass.setPipeline(pipeline); pass.dispatchWorkgroups(launch.x); pass.end();
              device.queue.submit([e.finish()]);
              results.push([await words(device, c.work), await words(device, c.bounds)]);
            }
            const [[work0, bounds0], [work1, bounds1]] = results as [[Uint32Array, Uint32Array], [Uint32Array, Uint32Array]];
            assert.equal(firstDifference(work0, work1), "", `census words, step ${step}`);
            assert.equal(firstDifference(bounds0, bounds1), "", `bound cubes and tile records, step ${step}`);
            const before = await words(device, saved[1]!);
            assert.ok(bounds0.some((w, i) => w !== before[i]) || work0.some(w => w !== 0), `classifyCoarse wrote the census, step ${step}`);
          }
        } finally { for (const b of saved) b.destroy(); }
      } finally { solver.destroy(); }
    });
  } finally { UniformMixedDynamicClassifier.serialCoarseBounds = false; }
});
