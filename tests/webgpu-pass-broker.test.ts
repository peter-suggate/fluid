import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { PassBroker } from "../lib/webgpu-pass-broker";

function fakeEncoder(events: string[]) {
  let passIndex = 0;
  return {
    beginComputePass(descriptor?: GPUComputePassDescriptor) {
      const index = ++passIndex;
      events.push(`begin:${descriptor?.label ?? index}`);
      return { end: () => events.push(`end:${index}`) };
    },
    clearBuffer() { events.push("clear"); },
    copyBufferToBuffer() { events.push("copy"); },
    finish() { events.push("finish"); return {}; },
  } as unknown as GPUCommandEncoder;
}

test("PassBroker reuses compute until an explicit fence", () => {
  const events: string[] = [];
  const broker = new PassBroker(fakeEncoder(events));
  const first = broker.compute({ label: "first" });
  assert.equal(broker.compute({ label: "second" }), first);
  assert.equal(broker.computePassCount, 1);
  assert.equal(broker.hasOpenComputePass, true);
  broker.fence("publication visibility");
  assert.equal(broker.lastFenceReason, "publication visibility");
  broker.fence("idempotent");
  assert.equal(broker.hasOpenComputePass, false);
  broker.compute({ label: "third" });
  broker.finish();
  assert.equal(broker.computePassCount, 2);
  assert.deepEqual(events, ["begin:first", "end:1", "begin:third", "end:2", "finish"]);
});

test("PassBroker copy, clear, and indirect update close compute first", () => {
  const events: string[] = [];
  const broker = new PassBroker(fakeEncoder(events));
  const buffer = {} as GPUBuffer;
  broker.compute();
  broker.clearBuffer(buffer);
  broker.compute();
  broker.copyBufferToBuffer(buffer, 0, buffer, 0, 4);
  broker.compute();
  broker.updateIndirectBuffer(buffer, 0, buffer, 0, 12);
  assert.deepEqual(events, [
    "begin:1", "end:1", "clear",
    "begin:2", "end:2", "copy",
    "begin:3", "end:3", "copy",
  ]);
});

test("raw command encoder access is an explicit pass boundary", () => {
  const events: string[] = [];
  const encoder = fakeEncoder(events);
  const broker = new PassBroker(encoder);
  broker.compute({ label: "module" });
  assert.equal(broker.commandEncoder(), encoder);
  assert.equal(broker.hasOpenComputePass, false);
  assert.equal(broker.lastFenceReason, "raw command encoder access");
  assert.deepEqual(events, ["begin:module", "end:1"]);
});

test("PassBroker cutover has no raw-encoder adapter or proxy facade", () => {
  const brokerSource = readFileSync(new URL("../lib/webgpu-pass-broker.ts", import.meta.url), "utf8");
  const ownerPageSource = readFileSync(new URL("../lib/webgpu-octree-owner-pages.ts", import.meta.url), "utf8");
  const powerDescriptorSource = readFileSync(
    new URL("../lib/webgpu-octree-power-descriptor.ts", import.meta.url), "utf8");
  const powerTopologySource = readFileSync(
    new URL("../lib/webgpu-octree-power-topology.ts", import.meta.url), "utf8");
  const powerFacesSource = readFileSync(
    new URL("../lib/webgpu-octree-power-faces.ts", import.meta.url), "utf8");
  const powerSolidFacesSource = readFileSync(
    new URL("../lib/webgpu-octree-power-solid-faces.ts", import.meta.url), "utf8");
  const fineTopologySource = readFileSync(
    new URL("../lib/webgpu-octree-fine-levelset-topology.ts", import.meta.url), "utf8");
  const fineTransportSource = readFileSync(
    new URL("../lib/webgpu-octree-fine-levelset-transport.ts", import.meta.url), "utf8");
  const velocityPrepassSource = readFileSync(
    new URL("../lib/webgpu-octree-power-velocity-prepass.ts", import.meta.url), "utf8");
  const mgpcgSource = readFileSync(
    new URL("../lib/webgpu-octree-mgpcg.ts", import.meta.url), "utf8");
  const spgridSource = readFileSync(
    new URL("../lib/webgpu-octree-spgrid-vcycle.ts", import.meta.url), "utf8");
  const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const advanceSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");

  for (const legacyName of ["PassBrokerSource", "withPassBroker", "commandEncoderFacade", "PASS_BROKER_OWNER",
    "legacy encoder boundary"]) {
    assert.doesNotMatch(brokerSource, new RegExp(legacyName), `${legacyName} must stay deleted`);
  }
  assert.match(brokerSource, /constructor\(private readonly encoder: GPUCommandEncoder\)/,
    "raw encoder access must go through the fencing commandEncoder boundary");
  assert.match(ownerPageSource, /encode\(broker: PassBroker\): void/);
  assert.match(ownerPageSource, /encodeAnalyticBootstrap\(broker: PassBroker\): void/);
  assert.doesNotMatch(ownerPageSource, /GPUCommandEncoder \| PassBroker|PassBrokerSource|withPassBroker/);
  for (const [name, source] of [
    ["power descriptors", powerDescriptorSource],
    ["power topology", powerTopologySource],
    ["power faces", powerFacesSource],
  ] as const) {
    assert.match(source, /encode(?:RowDirectory)?\(broker: PassBroker/,
      `${name} must consume the caller's shared broker`);
    assert.doesNotMatch(source, /beginComputePass/,
      `${name} must not own local compute passes after cutover`);
  }
  for (const [name, source] of [
    ["power solid faces", powerSolidFacesSource],
  ] as const) {
    assert.match(source, /encode[A-Za-z]*\(broker: PassBroker/,
      `${name} must consume the caller's pressure-spine broker`);
    assert.doesNotMatch(source, /beginComputePass|new PassBroker|broker\.fence|commandEncoder\(\)/,
      `${name} must not own or routinely terminate the caller's pressure-spine pass`);
  }
  assert.match(fineTopologySource, /encode\(broker: PassBroker,/,
    "fine topology publication must consume the caller's shared broker");
  assert.match(fineTopologySource, /encodeFromAllInterfaceLeaves\(broker: PassBroker,/,
    "fine interface seeding must consume the caller's shared broker");
  assert.match(fineTopologySource, /encodeFinalizePublication\(broker: PassBroker,/,
    "fine publication finalization must consume the caller's shared broker");
  assert.doesNotMatch(fineTopologySource, /beginComputePass|encode(?:FromAllInterfaceLeaves|FinalizePublication)?\(encoder: GPUCommandEncoder/,
    "fine topology must not retain standalone compute-pass ownership or raw encoder entry points");
  assert.match(fineTransportSource, /encode\(initialBroker: PassBroker,/,
    "fine transport must consume the caller's shared broker");
  assert.doesNotMatch(fineTransportSource, /beginComputePass|new PassBroker|GPUCommandEncoder/,
    "fine transport must not retain standalone pass ownership or a raw encoder escape");
  assert.doesNotMatch(fineTransportSource,
    /broker\.fence\("fine transport publication complete"\)/,
    "fine transport leaves pass ownership with its caller after publication");
  assert.match(velocityPrepassSource, /encodeRowDescriptors\(broker: PassBroker,/);
  assert.match(velocityPrepassSource, /encodeFromPositions\(broker: PassBroker,/);
  assert.doesNotMatch(velocityPrepassSource, /beginComputePass|new PassBroker|GPUCommandEncoder/,
    "the direct Stage-B producer must not recreate a broker beneath fine transport");
  for (const [name, source] of [["MGPCG", mgpcgSource], ["SPGrid V-cycle", spgridSource]] as const) {
    assert.match(source, /PassBroker/, `${name} must use broker-owned encoding`);
    assert.doesNotMatch(source, /GPUCommandEncoder|beginComputePass|copyBufferToBuffer|sharedPass/,
      `${name} must not retain raw encoder, raw copy, or optional-pass compatibility paths`);
  }
  assert.match(octreeSource,
    /descriptor\.encode\(broker,[\s\S]*topology\.encode\(broker,[\s\S]*faces\.encode\(broker,/,
    "one broker must span descriptor, topology, and generalized-face publication");
  assert.doesNotMatch(octreeSource, /faces\.encodeRowDirectory/,
    "the retired duplicate face-directory prepass must stay deleted");
  assert.match(octreeSource,
    /operator\.encodeAssemblyFromControl\(broker,[\s\S]*splitProductionPhase\(undefined, "powerOperatorRhsAssembly"\);[\s\S]*operator\.encodeLeafRowPublication\(broker,/,
    "normal recurring assembly and leaf-row publication must share the pressure-spine broker");
  assert.doesNotMatch(octreeSource,
    /splitProductionPhase\(undefined, "powerOperatorRhsAssembly", true\)/,
    "intrusive phase attribution must not impose a routine production fence");
  assert.doesNotMatch(octreeSource, /this\.faceMirror|this\.faceTransport|this\.solidFaces/,
    "the deleted Cartesian face lane must not remain in the pressure spine");
  assert.doesNotMatch(advanceSource, /commandEncoderFacade|new PassBroker\(commandEncoder\)/,
    "the production advance must not route raw helpers through a compatibility proxy");
});

test("remaining octree production stages have no standalone compute-pass ownership", () => {
  const modules = [
    "webgpu-octree-power-face-seed",
    "webgpu-octree-coarse-levelset",
    "webgpu-octree-power-coarse-levelset",
    "webgpu-octree-power-velocity-prepass",
    "webgpu-octree-power-operator",
    "webgpu-octree-solid-vertex-sdf",
    "webgpu-octree-power-solid-faces",
    "webgpu-octree-fine-to-coarse-levelset",
    "webgpu-octree-fine-levelset-volume",
    "webgpu-octree-fine-levelset-summary",
    "webgpu-octree-fine-levelset-redistance",
    "webgpu-octree-analytic-bootstrap",
    "webgpu-octree-voxel-inspection",
    "webgpu-octree-technique-overlay",
  ];
  for (const module of modules) {
    const source = readFileSync(new URL(`../lib/${module}.ts`, import.meta.url), "utf8");
    assert.match(source, /PassBroker/, `${module} must use broker-owned encoding`);
    assert.doesNotMatch(source, /beginComputePass/,
      `${module} must not retain standalone compute-pass ownership`);
    if (module === "webgpu-octree-power-operator") {
      assert.doesNotMatch(source, /GPUCommandEncoder|new PassBroker|broker\.fence/,
        "the power operator must preserve its caller's pass across assembly, row publication, and projection");
      assert.match(source,
        /encodeAssemblyFromControl\(\s*broker: PassBroker,[\s\S]*encodeLeafRowPublication\(\s*broker: PassBroker,[\s\S]*encodeProjectionFromControl\(\s*broker: PassBroker,/,
        "every recurring power-operator entry point must consume the pressure spine broker");
    }
  }
});
