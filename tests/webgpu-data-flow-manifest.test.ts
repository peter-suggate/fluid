import assert from "node:assert/strict";
import test from "node:test";
import {
  GPUDataFlowAudit,
  classifyGPUDataFlowBuffer,
  parseGPUDataFlowWGSL,
} from "../tools/webgpu-data-flow-manifest";
import { summarizePowerDamPerformance } from "../tools/power-dam-performance-report";

const fakeBuffer = (label: string, size: number, usage = 128): GPUBuffer =>
  ({ label, size, usage }) as GPUBuffer;

test("WGSL data-flow parser recovers storage access and numeric workgroup sizes", () => {
  const parsed = parseGPUDataFlowWGSL(`
    @group(0) @binding(1) var<storage, read> source: array<u32>;
    @group(0)@binding(2)var<storage,read_write>destination:array<u32>;
    @group(1) @binding(0) var<uniform> params: vec4u;
    @compute @workgroup_size(64) fn trace() {}
    @compute@workgroup_size(8, 4, 2)fn scatter(){}
  `);
  assert.equal(parsed.bindings.get("0:1")?.access, "read");
  assert.equal(parsed.bindings.get("0:2")?.access, "read_write");
  assert.equal(parsed.bindings.get("1:0")?.access, "uniform");
  assert.deepEqual(parsed.workgroupSizes.get("trace"), [64, 1, 1]);
  assert.deepEqual(parsed.workgroupSizes.get("scatter"), [8, 4, 2]);
});

test("buffer labels identify controls, generations, topology, payloads, and scratch", () => {
  assert.deepEqual(classifyGPUDataFlowBuffer("Fine topology generation control"),
    ["control", "generation", "topology"]);
  assert.deepEqual(classifyGPUDataFlowBuffer("Power face velocity scratch"),
    ["topology", "payload", "scratch"]);
  assert.deepEqual(classifyGPUDataFlowBuffer("Live row indirect dispatch"),
    ["topology", "indirect"]);
});

test("manifest joins actual bindings and logical work sources to timestamp labels", () => {
  const audit = new GPUDataFlowAudit();
  const source = fakeBuffer("Fine phi generation payload", 4096);
  const destination = fakeBuffer("Fine transport outcome scratch", 2048);
  const control = fakeBuffer("Fine transport control", 64);
  const indirect = fakeBuffer("Fine transport indirect dispatch", 96, 256);
  for (const buffer of [source, destination, control, indirect]) {
    audit.registry.recordBuffer(buffer, {
      label: buffer.label,
      size: buffer.size,
      usage: buffer.usage,
    });
  }
  const module = { label: "transport module" } as GPUShaderModule;
  audit.registry.recordShader(module, {
    code: `
      @group(0) @binding(0) var<storage,read> source:array<u32>;
      @group(0) @binding(1) var<storage,read_write> outcome:array<u32>;
      @group(0) @binding(2) var<storage,read_write> control:array<u32>;
      @compute @workgroup_size(64) fn trace() {}
    `,
  });
  const pipeline = { label: "Trace pipeline" } as GPUComputePipeline;
  audit.registry.recordPipeline(pipeline, {
    label: "Trace pipeline",
    layout: "auto",
    compute: { module, entryPoint: "trace" },
  });
  const group = { label: "Trace group" } as GPUBindGroup;
  audit.registry.recordBindGroup(group, {
    layout: {} as GPUBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: source } },
      { binding: 1, resource: { buffer: destination, offset: 256, size: 1024 } },
      { binding: 2, resource: { buffer: control } },
    ],
  });

  assert.equal(audit.createEncoderSession(), undefined);
  audit.start();
  const pass = audit.createEncoderSession()!.beginPass("Trace fine characteristic");
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.direct(4);
  pass.indirect(indirect, 24);
  audit.stop();

  const manifest = audit.report(2, {
    "Trace fine characteristic": { samples: 2, total_ms: 8 },
  });
  assert.equal(manifest.schemaVersion, 1);
  assert.equal(manifest.measuredAdvances, 2);
  assert.equal(manifest.passes.length, 1);
  const recorded = manifest.passes[0]!;
  assert.equal(recorded.samples, 2);
  assert.equal(recorded.dispatches, 2);
  assert.equal(recorded.dispatchesPerAdvance, 1);
  assert.equal(recorded.totalPerAdvance_ms, 4);
  assert.equal(recorded.readBoundBytesUpperBound, 4096);
  assert.equal(recorded.writableBoundBytesUpperBound, 1024 + 64);
  assert.equal(recorded.paths[0]?.bindings[0]?.access, "read");
  assert.equal(recorded.paths[0]?.bindings[1]?.access, "read_write");
  assert.deepEqual(recorded.paths.find((path) => path.work.kind === "direct")?.work, {
    kind: "direct",
    workgroups: [4, 1, 1],
    workgroupSize: [64, 1, 1],
    logicalInvocationCapacity: 256,
  });
  const indirectWork = recorded.paths.find((path) => path.work.kind === "indirect")?.work;
  assert.equal(indirectWork?.kind, "indirect");
  if (indirectWork?.kind === "indirect") {
    assert.equal(indirectWork.offset, 24);
    assert.equal(indirectWork.logicalCount, "gpu-authored");
  }
  const authorityLabels = recorded.authorityBufferIds.map((id) =>
    manifest.buffers.find((buffer) => buffer.id === id)?.label);
  assert.deepEqual(authorityLabels.sort(), [
    "Fine phi generation payload",
    "Fine transport control",
  ]);
  assert.deepEqual(manifest.limitations, {
    boundBytes: "binding-range-upper-bound",
    indirectLogicalCount: "gpu-authored-not-read-back",
  });
});

test("stopped data-flow audit adds no recurring samples", () => {
  const audit = new GPUDataFlowAudit();
  audit.start();
  audit.stop();
  assert.equal(audit.createEncoderSession(), undefined);
  assert.deepEqual(audit.report(1).passes, []);
});

test("explicit bind-group unbinding removes stale data-flow edges", () => {
  const audit = new GPUDataFlowAudit();
  const buffer = fakeBuffer("temporary payload", 256);
  const group = {} as GPUBindGroup;
  audit.registry.recordBindGroup(group, {
    layout: {} as GPUBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer } }],
  });
  audit.start();
  const pass = audit.createEncoderSession()!.beginPass("Unbound dispatch");
  pass.setBindGroup(0, group);
  pass.setBindGroup(0, null);
  pass.direct(1);
  const path = audit.report(1).passes[0]?.paths[0];
  assert.deepEqual(path?.bindings, []);
});

test("power-dam JSON summary retains the machine-readable manifest", () => {
  const manifest = new GPUDataFlowAudit().report(3);
  const summary = summarizePowerDamPerformance({
    scenario: "minimal-power-dam-break",
    method: "octree",
    phase: "result",
    steps: 62,
    simulationWall_ms: 620,
    gpuDataFlowManifest: manifest,
  });
  assert.equal(summary.dataFlow, manifest);
  assert.equal(summary.dataFlow?.schemaVersion, 1);
});
