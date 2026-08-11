import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  PassBroker,
  formatPassBrokerBoundaryAudit,
  formatPassBrokerLabelAudit,
  passBrokerBoundaryAuditSnapshot,
  passBrokerBoundaryAuditTotals,
  passBrokerLabelIsolationPrefixes,
  passBrokerLabelIsolationRequested,
  resetPassBrokerBoundaryAuditTotals,
  xctraceSafeComputeLabel,
} from "../lib/webgpu-pass-broker";

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

test("compute labels remain attributable through Dawn's ASCII-only Metal path", () => {
  assert.equal(xctraceSafeComputeLabel("SPGrid V-cycle · A→B × 4"),
    "SPGrid V-cycle - AtoB x 4");
  const events: string[] = [];
  const broker = new PassBroker(fakeEncoder(events), {
    isolateLabels: true,
    isolateLabelPrefixes: ["SPGrid V-cycle ·"],
  });
  broker.compute({ label: "SPGrid V-cycle · one-pass symmetric correction" });
  broker.compute({ label: "ordinary" });
  broker.finish();
  assert.deepEqual(events, [
    "begin:SPGrid V-cycle - one-pass symmetric correction", "end:1",
    "begin:ordinary", "end:2", "finish",
  ]);
});

test("label isolation gives every distinct compute label its own pass", () => {
  const events: string[] = [];
  const broker = new PassBroker(fakeEncoder(events), { isolateLabels: true });
  const first = broker.compute({ label: "count rows" });
  // A different label must not be silently swallowed into the open pass: that
  // swallowing is what makes a per-pass timestamp report attribute one label's
  // ms to a whole group of stages.
  const second = broker.compute({ label: "scatter families" });
  assert.notEqual(second, first);
  // Repeating a label keeps the pass, so a stage that reopens its own pass to
  // add dispatches still reports as one bucket.
  assert.equal(broker.compute({ label: "scatter families" }), second);
  // An unlabelled request never forces a split; there is no bucket to protect.
  assert.equal(broker.compute(), second);
  broker.finish();
  assert.equal(broker.computePassCount, 2);
  assert.deepEqual(events, [
    "begin:count rows", "end:1", "begin:scatter families", "end:2", "finish",
  ]);
});

test("label isolation can protect one micro-stage prefix without splitting unrelated stages", () => {
  const events: string[] = [];
  const broker = new PassBroker(fakeEncoder(events), {
    isolateLabels: true,
    isolateLabelPrefixes: ["Fine JFA -"],
  });
  const ordinary = broker.compute({ label: "ordinary A" });
  assert.equal(broker.compute({ label: "ordinary B" }), ordinary,
    "unrelated labels retain the production grouping");
  const firstFine = broker.compute({ label: "Fine JFA - seed" });
  assert.notEqual(firstFine, ordinary, "entering the selected prefix closes ordinary work");
  const secondFine = broker.compute({ label: "Fine JFA - flood stride 16" });
  assert.notEqual(secondFine, firstFine, "selected micro-stages remain individually attributable");
  const after = broker.compute({ label: "ordinary C" });
  assert.notEqual(after, secondFine, "leaving the selected prefix closes the target bucket");
  assert.equal(broker.compute({ label: "ordinary D" }), after);
  broker.finish();
  assert.deepEqual(events, [
    "begin:ordinary A", "end:1", "begin:Fine JFA - seed", "end:2",
    "begin:Fine JFA - flood stride 16", "end:3", "begin:ordinary C", "end:4", "finish",
  ]);
  assert.deepEqual(passBrokerLabelIsolationPrefixes({
    FLUID_GPU_ISOLATE_PASS_LABEL_PREFIXES: " Fine JFA - , SPGrid accurate A2 - ",
  }), ["Fine JFA -", "SPGrid accurate A2 -"]);
});

test("label isolation is off unless asked for, and then only changes pass boundaries", () => {
  assert.equal(passBrokerLabelIsolationRequested({}), false);
  // A browser has no `process`; the broker must not throw there.
  assert.equal(passBrokerLabelIsolationRequested(undefined), false);
  assert.equal(passBrokerLabelIsolationRequested({ FLUID_GPU_ISOLATE_PASS_LABELS: "0" }), false);
  assert.equal(passBrokerLabelIsolationRequested({ FLUID_GPU_ISOLATE_PASS_LABELS: "1" }), true);
  // Default construction must reproduce the production command stream exactly:
  // one pass across differing labels, and the later labels dropped.
  const events: string[] = [];
  const broker = new PassBroker(fakeEncoder(events), { isolateLabels: false });
  const first = broker.compute({ label: "count rows" });
  assert.equal(broker.compute({ label: "scatter families" }), first);
  broker.finish();
  assert.equal(broker.computePassCount, 1);
  assert.deepEqual(events, ["begin:count rows", "end:1", "finish"]);
});

/**
 * Regression pin for a 3.55 ms misdiagnosis on the mini dam lane, 2026-07-28.
 *
 * A capture scoped to `--isolate-label-prefix="Fine JFA -"` reported
 * `Structured boundary worksets - count row classes` at 3.553 ms/advance with
 * 0.6% occupancy. That kernel is a 24-workgroup classify over ~1,483 rows. The
 * number was real GPU time, but it belonged to the whole SPGrid candidate
 * hierarchy rebuild, which `encodeFromStructuredControl` leaves sharing the
 * pass that the count kernel opened. A fully isolated capture the same day
 * measured the count kernel at 0.009 ms and the rebuild stages at ~3.39 ms.
 *
 * Metal names an encoder once, so the later labels can never reach a trace.
 * What the broker CAN do is say which labels it dropped and into what -- the
 * fact a report needs to stop printing a composite bucket as a stage.
 */
test("the broker records which labels a shared pass swallowed", () => {
  const events: string[] = [];
  const broker = new PassBroker(fakeEncoder(events), {
    isolateLabels: true, isolateLabelPrefixes: ["Fine JFA -"],
  });
  // The five workset publication dispatches, unfenced, exactly as
  // `encodeFromStructuredControl` encodes them.
  const publication = [
    "Structured boundary worksets - count row classes",
    "Structured boundary worksets - count family classes",
    "Structured boundary worksets - scan blocks",
    "Structured boundary worksets - scatter rows",
    "Structured boundary worksets - scatter families",
  ];
  const first = broker.compute({ label: publication[0] });
  for (const label of publication.slice(1)) {
    assert.equal(broker.compute({ label }), first,
      "an out-of-scope label must keep the production grouping");
  }
  // ...and then the caller keeps encoding into the same open pass.
  broker.compute({ label: "SPGrid V-cycle - candidate build level sets" });
  broker.compute();
  assert.equal(broker.computePassCount, 1, "one Metal encoder holds the whole group");

  assert.equal(broker.attributionIsExact(publication[0]), false,
    "the label that opened the pass must not be presented as an exact stage");
  assert.deepEqual(broker.absorbedBy(publication[0]), [
    ...publication.slice(1),
    "SPGrid V-cycle - candidate build level sets",
    "(unlabelled)",
  ], "every stage charged to that encoder must be nameable");
  assert.equal(broker.absorbedLabelCount, 6);

  // A stage that really did own its encoder stays exact.
  const fine = broker.compute({ label: "Fine JFA - seed closest points" });
  assert.notEqual(fine, first);
  broker.fence("fine seed complete");
  assert.equal(broker.attributionIsExact("Fine JFA - seed closest points"), true);

  // Repeating one label is not absorption: the bucket is still that stage.
  const repeated = broker.compute({ label: "repeat" });
  assert.equal(broker.compute({ label: "repeat" }), repeated);
  assert.equal(broker.attributionIsExact("repeat"), true);

  const audit = formatPassBrokerLabelAudit(broker.labelAttributionAudit);
  assert.ok(audit[0]?.includes("composite buckets"), audit.join("\n"));
  assert.ok(audit.some((line) => line.includes(publication[0]) && line.includes("+ 6")),
    audit.join("\n"));
});

test("the audit keys on the label Metal actually recorded, including none", () => {
  const events: string[] = [];
  const broker = new PassBroker(fakeEncoder(events), { isolateLabels: false });
  // A pass opened without a label still owns everything encoded into it, and a
  // report that finds `(unlabelled)` in the trace must be able to look it up.
  broker.compute();
  broker.compute({ label: "swallowed by an anonymous pass" });
  assert.deepEqual([...broker.labelAttributionAudit.keys()], ["(unlabelled)"]);
  assert.deepEqual(broker.absorbedBy("(unlabelled)"), ["swallowed by an anonymous pass"]);
  // The audit is keyed by the xctrace-safe form, because that is the only
  // spelling a Metal trace can ever show.
  broker.fence("next");
  broker.compute({ label: "SPGrid V-cycle · correction" });
  broker.compute({ label: "tail" });
  assert.deepEqual(broker.absorbedBy("SPGrid V-cycle · correction"), ["tail"]);
  assert.deepEqual(broker.absorbedBy("SPGrid V-cycle - correction"), ["tail"]);
  assert.equal(formatPassBrokerLabelAudit(new Map()).length, 0);
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
  assert.deepEqual(broker.boundaryAudit.get("clear buffer"), {
    requests: 1, passClosures: 1, copyCommands: 0, clearCommands: 1, commandBytes: 0,
  });
  assert.deepEqual(broker.boundaryAudit.get("copy buffer"), {
    requests: 1, passClosures: 1, copyCommands: 1, clearCommands: 0, commandBytes: 4,
  });
  assert.deepEqual(broker.boundaryAudit.get("stage indirect args"), {
    requests: 1, passClosures: 1, copyCommands: 1, clearCommands: 0, commandBytes: 12,
  });
  assert.deepEqual(formatPassBrokerBoundaryAudit(broker.boundaryAudit), [
    "3 compute-pass boundaries by cause:",
    "  stage indirect args: 1 closures / 1 requests, 1 copies, 0 clears, 12 bytes",
    "  copy buffer: 1 closures / 1 requests, 1 copies, 0 clears, 4 bytes",
    "  clear buffer: 1 closures / 1 requests, 0 copies, 1 clears, 0 bytes",
  ]);
});

test("boundary audit distinguishes idempotent requests from serial-spine closures", () => {
  const events: string[] = [];
  const broker = new PassBroker(fakeEncoder(events));
  broker.fence("publication");
  broker.compute(); broker.fence("publication"); broker.fence("publication");
  assert.deepEqual(broker.boundaryAudit.get("publication"), {
    requests: 3, passClosures: 1, copyCommands: 0, clearCommands: 0, commandBytes: 0,
  });
});

/**
 * A broker owns one command encoder and is dropped at `finish()`. Roughly
 * thirty call sites construct one per encoder, so `boundaryAudit` on any single
 * instance describes a fragment of a frame and is garbage before a report
 * exists. The process-wide census is what a benchmark can actually read.
 */
test("boundary causes aggregate across every broker instance in the process", () => {
  resetPassBrokerBoundaryAuditTotals();
  const events: string[] = [];
  const buffer = { size: 64 } as GPUBuffer;
  for (let encoder = 0; encoder < 3; encoder += 1) {
    const broker = new PassBroker(fakeEncoder(events), { isolateLabels: false });
    broker.compute({ label: "stage" });
    broker.updateIndirectBuffer(buffer, 0, buffer, 0, 12);
    broker.compute({ label: "next stage" });
    broker.fence("publication visibility");
    // Idempotent: nothing is open, so this is a caller that could be batched
    // away for free rather than a pass that has to be merged.
    broker.fence("publication visibility");
    broker.finish();
    assert.equal(broker.boundaryAudit.get("publication visibility")?.passClosures, 1,
      "a per-encoder audit must keep describing only its own encoder");
  }
  const totals = passBrokerBoundaryAuditTotals();
  assert.deepEqual(totals.get("stage indirect args"), {
    requests: 3, passClosures: 3, copyCommands: 3, clearCommands: 0, commandBytes: 36,
  });
  assert.deepEqual(totals.get("publication visibility"), {
    requests: 6, passClosures: 3, copyCommands: 0, clearCommands: 0, commandBytes: 0,
  });

  const snapshot = passBrokerBoundaryAuditSnapshot();
  assert.equal(snapshot.brokers, 3);
  assert.equal(snapshot.passClosures, 6);
  assert.equal(snapshot.requests, 12);
  assert.equal(snapshot.copyCommands, 3);
  assert.equal(snapshot.commandBytes, 36);
  assert.equal(snapshot.labelIsolated, false);
  assert.ok(snapshot.resets >= 1, "an unreset census still carries construction");
  assert.deepEqual(Object.keys(snapshot.byReason), [
    "stage indirect args", "publication visibility", "finish command encoder",
  ], "the snapshot is ranked exactly as the formatter prints");

  // A snapshot is a copy: encoding that continues afterwards must not mutate a
  // report a harness is already holding.
  const held = totals.get("publication visibility");
  const later = new PassBroker(fakeEncoder(events), { isolateLabels: false });
  later.compute();
  later.fence("publication visibility");
  assert.equal(held?.passClosures, 3);
  assert.equal(snapshot.byReason["publication visibility"]?.passClosures, 3);
  assert.equal(passBrokerBoundaryAuditTotals().get("publication visibility")?.passClosures, 4);
});

test("the process census records label isolation and the merges already happening", () => {
  resetPassBrokerBoundaryAuditTotals();
  assert.equal(passBrokerBoundaryAuditTotals().size, 0, "a reset starts a new window");
  const events: string[] = [];
  // Production grouping: the second label is dropped into the first label's
  // encoder, which is a merge that has already happened.
  const shared = new PassBroker(fakeEncoder(events), { isolateLabels: false });
  shared.compute({ label: "owner" });
  shared.compute({ label: "swallowed" });
  shared.finish();
  assert.equal(passBrokerBoundaryAuditSnapshot().absorbedLabelPairs, 1);
  assert.equal(passBrokerBoundaryAuditSnapshot().compositeEncoderLabels, 1);
  assert.equal(passBrokerBoundaryAuditSnapshot().labelIsolated, false);

  // With isolation on the broker manufactures a boundary per label change. A
  // table that did not say so would report a diagnostic stream as production.
  const isolated = new PassBroker(fakeEncoder(events), {
    isolateLabels: true, isolateLabelPrefixes: ["SPGrid V-cycle ·"],
  });
  isolated.compute({ label: "SPGrid V-cycle · smooth" });
  isolated.compute({ label: "SPGrid V-cycle · restrict" });
  isolated.finish();
  const snapshot = passBrokerBoundaryAuditSnapshot();
  assert.equal(snapshot.labelIsolated, true);
  assert.equal(snapshot.labelIsolatedBrokers, 1);
  assert.equal(snapshot.brokers, 2);
  assert.deepEqual(snapshot.labelIsolationPrefixes, ["SPGrid V-cycle -"]);
  assert.equal(snapshot.byReason["pass label isolation"]?.passClosures, 1);
  resetPassBrokerBoundaryAuditTotals();
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
  const structuredVelocitySource = readFileSync(
    new URL("../lib/webgpu-octree-structured-velocity-gpu.ts", import.meta.url), "utf8");
  const structuredBoundarySource = readFileSync(
    new URL("../lib/webgpu-octree-structured-boundary.ts", import.meta.url), "utf8");
  const structuredDynamicsSource = readFileSync(
    new URL("../lib/webgpu-octree-structured-dynamics.ts", import.meta.url), "utf8");
  const fineTopologySource = readFileSync(
    new URL("../lib/webgpu-octree-fine-levelset-topology.ts", import.meta.url), "utf8");
  const fineTransportSource = readFileSync(
    new URL("../lib/webgpu-octree-fine-levelset-transport.ts", import.meta.url), "utf8");
  const spgridSource = readFileSync(
    new URL("../lib/webgpu-octree-spgrid-vcycle.ts", import.meta.url), "utf8");
  const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const advanceSource = readFileSync(new URL("../lib/webgpu-octree-eulerian.ts", import.meta.url), "utf8");

  for (const legacyName of ["PassBrokerSource", "withPassBroker", "commandEncoderFacade", "PASS_BROKER_OWNER",
    "legacy encoder boundary"]) {
    assert.doesNotMatch(brokerSource, new RegExp(legacyName), `${legacyName} must stay deleted`);
  }
  assert.match(brokerSource, /constructor\(private readonly encoder: GPUCommandEncoder[,)]/,
    "raw encoder access must go through the fencing commandEncoder boundary");
  assert.match(ownerPageSource, /encodeInactiveCandidate\(broker: PassBroker\): void/);
  assert.match(ownerPageSource, /encodeReadyCommit\(broker: PassBroker\): void/);
  assert.match(ownerPageSource, /encodeAnalyticBootstrap\(broker: PassBroker\): void/);
  assert.doesNotMatch(ownerPageSource, /GPUCommandEncoder \| PassBroker|PassBrokerSource|withPassBroker/);
  for (const [name, source] of [
    ["power descriptors", powerDescriptorSource],
    ["power topology", powerTopologySource],
    ["direct structured velocity", structuredVelocitySource],
  ] as const) {
    assert.match(source, /encode(?:RowDirectory)?\(broker: PassBroker/,
      `${name} must consume the caller's shared broker`);
    assert.doesNotMatch(source, /beginComputePass/,
      `${name} must not own local compute passes after cutover`);
  }
  for (const [name, source] of [
    ["structured boundary coefficients", structuredBoundarySource],
    ["structured velocity dynamics", structuredDynamicsSource],
  ] as const) {
    assert.match(source, /encode[A-Za-z]*\(broker: PassBroker/,
      `${name} must consume the caller's pressure-spine broker`);
    assert.doesNotMatch(source, /beginComputePass|new PassBroker|commandEncoder\(\)/,
      `${name} must not own a pass or construct a broker beneath the pressure spine`);
  }
  assert.match(fineTopologySource, /encode\(broker: PassBroker,/,
    "fine topology publication must consume the caller's shared broker");
  assert.match(fineTopologySource, /encodeFromAllInterfaceLeaves\(broker: PassBroker,/,
    "fine interface seeding must consume the caller's shared broker");
  assert.match(fineTopologySource, /encodeFinalizePublication\(broker: PassBroker,/,
    "fine publication finalization must consume the caller's shared broker");
  assert.doesNotMatch(fineTopologySource, /beginComputePass|encode(?:FromAllInterfaceLeaves|FinalizePublication)?\(encoder: GPUCommandEncoder/,
    "fine topology must not retain standalone compute-pass ownership or raw encoder entry points");
  assert.match(fineTransportSource, /encode\(broker: PassBroker,/,
    "fine transport must consume the caller's shared broker");
  assert.doesNotMatch(fineTransportSource, /beginComputePass|new PassBroker|GPUCommandEncoder/,
    "fine transport must not retain standalone pass ownership or a raw encoder escape");
  assert.doesNotMatch(fineTransportSource,
    /broker\.fence\("fine transport publication complete"\)/,
    "fine transport leaves pass ownership with its caller after publication");
  assert.match(spgridSource, /PassBroker/, "SPGrid V-cycle must use broker-owned encoding");
  assert.doesNotMatch(spgridSource, /GPUCommandEncoder|beginComputePass|new PassBroker/,
    "SPGrid V-cycle must not retain raw encoder or optional-pass ownership paths");
  assert.doesNotMatch(octreeSource,
    /this\.powerFaces|this\.powerOperator|this\.powerVelocity|this\.globalFineFaceExtension/,
    "the retired generalized-face graph must not remain in the pressure spine");
  assert.doesNotMatch(advanceSource, /commandEncoderFacade|new PassBroker\(commandEncoder\)/,
    "the production advance must not route raw helpers through a compatibility proxy");
});
