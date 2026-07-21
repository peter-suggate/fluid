import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_SURFACE_STATE,
  OCTREE_UNIFIED_SURFACE_RESIDENT_FRACTION,
  WebGPUOctreeSurfacePages,
  octreeSurfaceDensePublicationShader,
  octreeSurfacePageShader,
  planOctreeSurfaceBacktraceSegments,
  planOctreeSurfacePages,
  requiredOctreeSurfaceHaloCells,
  selectOctreeSurfaceCandidates,
  traceOctreeSurfaceBacktrace,
} from "../lib/webgpu-octree-surface-pages";

test("unified direct-page reserve covers the dam-break cold core/halo census", () => {
  const rowCapacity = 41_728;
  const observedColdCandidates = 15_464;
  const plan = planOctreeSurfacePages(rowCapacity, [240, 72, 64], {
    maximumResidentFraction: OCTREE_UNIFIED_SURFACE_RESIDENT_FRACTION,
  });
  assert.equal(OCTREE_UNIFIED_SURFACE_RESIDENT_FRACTION, 0.40);
  assert.equal(plan.pageCapacity, 16_692);
  assert.ok(plan.pageCapacity > observedColdCandidates);
  assert.ok(plan.pageCapacity < rowCapacity / 2,
    "the reserve must stay proportional to the narrow compact publication");
});

test("dense publication binding is dropped idempotently after bootstrap", () => {
  const pages = Object.create(WebGPUOctreeSurfacePages.prototype) as unknown as {
    densePublicationBindGroup?: GPUBindGroup;
  };
  pages.densePublicationBindGroup = {} as GPUBindGroup;
  const release = WebGPUOctreeSurfacePages.prototype.releaseDensePublicationBinding;
  assert.equal(release.call(pages as never), true);
  assert.equal(pages.densePublicationBindGroup, undefined);
  assert.equal(release.call(pages as never), false);
});

test("surface page arena scales with compact leaves instead of box volume", () => {
  const shallow = planOctreeSurfacePages(100, [64, 64, 64], { maximumPages: 10 });
  const deep = planOctreeSurfacePages(100, [64, 64, 4096], { maximumPages: 10 });
  assert.equal(shallow.pageCapacity, 10);
  assert.equal(shallow.hashCapacity, 256, "the direct topology sampler indexes every live leaf");
  assert.equal(shallow.airHashCapacity, 128, "the exact-key air halo remains proportional to resident pages");
  assert.equal(shallow.airHaloCells, 4);
  assert.equal(shallow.maximumSegments, 8);
  assert.equal(shallow.pageResolution, 2);
  assert.equal(shallow.samplesPerPage, 8);
  assert.equal(shallow.allocatedWords, 1_002);
  assert.equal(shallow.arenaBytes, 4_008);
  assert.equal(shallow.allocatedBytes, 4_148);
  assert.equal(deep.allocatedBytes, shallow.allocatedBytes);
  assert.equal(shallow.denseEquivalentBytes, 64 * 64 * 64 * 12);
  assert.equal(deep.denseEquivalentBytes, 64 * 64 * 4096 * 12);
});

test("surface planning degrades to a byte ceiling and accounts exactly", () => {
  const onePage = planOctreeSurfacePages(8, [8, 8, 8], { maximumPages: 1 });
  const degraded = planOctreeSurfacePages(8, [8, 8, 8], { maximumPages: 8, maximumArenaBytes: onePage.arenaBytes });
  assert.equal(degraded.pageCapacity, 1);
  assert.equal(degraded.degraded, true);
  assert.equal(degraded.allocatedBytes, onePage.allocatedBytes);
  assert.throws(() => planOctreeSurfacePages(8, [8, 8, 8], { maximumArenaBytes: 4 }), /cannot hold one page/);
  assert.throws(() => planOctreeSurfacePages(8, [8, 8, 8], { airHaloCells: 0 }), /air halo/);
  assert.throws(() => planOctreeSurfacePages(8, [8, 8, 8], { airHaloCells: 9 }), /air halo/);
  assert.equal(planOctreeSurfacePages(8, [8, 8, 8], { airHaloCells: 3 }).airHaloCells, 3);
  const highQuality = planOctreeSurfacePages(8, [8, 8, 8], { maximumPages: 1, pageResolution: (4 as const) });
  assert.equal(highQuality.pageResolution, 4);
  assert.equal(highQuality.samplesPerPage, 64);
  assert.throws(() => planOctreeSurfacePages(8, [8, 8, 8], { pageResolution: 3 as 2 | 4 }), /2 or 4/);
  assert.equal(planOctreeSurfacePages(8, [8, 8, 8], { maximumSegments: 16 }).maximumSegments, 16);
  assert.throws(() => planOctreeSurfacePages(8, [8, 8, 8], { maximumSegments: 0 }), /maximum segments/);
  assert.throws(() => planOctreeSurfacePages(8, [8, 8, 8], { maximumSegments: 17 }), /maximum segments/);
});

test("bounded multisegment planning follows local page displacement", () => {
  assert.equal(planOctreeSurfaceBacktraceSegments(0, 1, 0.25), 1);
  assert.equal(planOctreeSurfaceBacktraceSegments(1, 1, 0.25), 4);
  assert.equal(planOctreeSurfaceBacktraceSegments(100, 1, 0.25, 8), 8);
  assert.equal(planOctreeSurfaceBacktraceSegments(1, Number.NaN, 0.25), 1);
  assert.throws(() => planOctreeSurfaceBacktraceSegments(1, 1, 0.25, 17), /maximum segments/);
});

test("multisegment oracle translates a plane and resamples across page boundaries", () => {
  const translated = traceOctreeSurfaceBacktrace([1, 0, 0], () => [2, 0, 0], 0.5, 0.25);
  assert.deepEqual(translated, { departure: [0, 0, 0], segments: 4, velocitySamples: 4 });
  const oldPlane = (x: number) => x - 0.25;
  assert.equal(oldPlane(translated.departure[0]), -0.25, "phi is sampled once at the complete departure point");

  const crossed: number[] = [];
  const adaptive = traceOctreeSurfaceBacktrace([1.25, 0, 0], ([x]) => {
    crossed.push(x);
    return [x >= 1 ? 2 : 1, 0, 0];
  }, 1, 0.5, 8);
  assert.equal(adaptive.segments, 4);
  assert.deepEqual(crossed, [1.25, 0.75, 0.5, 0.25], "velocity is re-queried after crossing sparse page ownership");
  assert.deepEqual(adaptive.departure, [0, 0, 0]);
});

test("multisegment resampling improves rotating and deforming trajectories", () => {
  const rotation = ([x, y]: readonly number[]) => [-y, x, 0] as const;
  const start = [1, 0, 0] as const, dt = 0.5;
  const exactRotation = [Math.cos(dt), -Math.sin(dt), 0] as const;
  const one = traceOctreeSurfaceBacktrace(start, rotation, dt, 1, 1).departure;
  const segmented = traceOctreeSurfaceBacktrace(start, rotation, dt, 0.03125, 16).departure;
  const error = (a: readonly number[], b: readonly number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  assert.ok(error(segmented, exactRotation) < error(one, exactRotation));

  const deformation = ([x, y, z]: readonly number[]) => [x, -0.5 * y, 0.25 * z] as const;
  const deformStart = [1, 1, 1] as const;
  const exactDeformation = [Math.exp(-dt), Math.exp(0.5 * dt), Math.exp(-0.25 * dt)] as const;
  const deformOne = traceOctreeSurfaceBacktrace(deformStart, deformation, dt, 1, 1).departure;
  const deformSegmented = traceOctreeSurfaceBacktrace(deformStart, deformation, dt, 0.03125, 16).departure;
  assert.ok(error(deformSegmented, exactDeformation) < error(deformOne, exactDeformation));
});

test("2-cubed and 4-cubed pages choose identical trajectories at equal physical spacing", () => {
  const velocity = ([x, y]: readonly number[]) => [1 + 0.1 * y, 0.1 * x, 0] as const;
  const page2 = planOctreeSurfacePages(4, [8, 8, 8], { pageResolution: 2, maximumPages: 1 });
  const page4 = planOctreeSurfacePages(4, [8, 8, 8], { pageResolution: 4, maximumPages: 1 });
  const two = traceOctreeSurfaceBacktrace([0.75, 0.5, 0], velocity, 0.5, 0.25, page2.maximumSegments);
  const four = traceOctreeSurfaceBacktrace([0.75, 0.5, 0], velocity, 0.5, 0.25, page4.maximumSegments);
  assert.deepEqual(four, two);
});

test("swept halo selects finest neighbors around interface leaves", () => {
  assert.equal(requiredOctreeSurfaceHaloCells(3, 0.5, 1), 6);
  const selected = selectOctreeSurfaceCandidates([
    { row: 0, origin: [0, 0, 0], size: 1, finest: true, phiMin: -1, phiMax: 1 },
    { row: 1, origin: [3, 0, 0], size: 1, finest: true, phiMin: 1, phiMax: 2 },
    { row: 2, origin: [8, 0, 0], size: 1, finest: true, phiMin: 1, phiMax: 2 },
    { row: 3, origin: [1, 0, 0], size: 2, finest: false, phiMin: -1, phiMax: 1 },
  ], 2);
  assert.deepEqual(selected, [
    { row: 0, flags: OCTREE_SURFACE_STATE.core },
    { row: 1, flags: OCTREE_SURFACE_STATE.halo },
  ]);
});

test("WGSL exposes guarded lifecycle, hierarchy, transport and redistance", () => {
  assert.match(octreeSurfacePageShader, /fn hierarchicalPhi/);
  assert.match(octreeSurfacePageShader, /fn transportPhi/);
  assert.match(octreeSurfacePageShader, /fn traceSparseDeparture/);
  assert.match(octreeSurfacePageShader, /velocity=leaves\[row\]\.motion\.xyz/);
  assert.match(octreeSurfacePageShader, /atomicAdd\(&arena\[11\],1u\)/);
  assert.doesNotMatch(octreeSurfacePageShader, /atomicOr\(&arena\[3\],DEPARTURE_OUTSIDE_BAND\)/,
    "one unsupported departure must not invalidate every resident surface page");
  assert.match(octreeSurfacePageShader, /hierarchicalPhi\(row,trace\.departure,params\.offsets1\.z\)/);
  assert.match(octreeSurfacePageShader, /fn redistanceAToB/);
  assert.match(octreeSurfacePageShader, /fn redistanceBToA/);
  assert.match(octreeSurfacePageShader, /atomicLoad\(&arena\[3\]\)!=0u/);
  assert.match(octreeSurfacePageShader, /prepareDispatch/);
  assert.match(octreeSurfacePageShader, /abs\(x\)\+abs\(y\)\+abs\(z\)>radius/);
  assert.match(octreeSurfacePageShader, /let encodedRow=0xffffffffu-row/);
  assert.match(octreeSurfacePageShader, /@group\(0\) @binding\(5\) var<storage,read> previousLeaves/);
  assert.match(octreeSurfacePageShader, /fn previousPageRow/);
  assert.match(octreeSurfacePageShader, /previousLeaves\[oldRow\]\.pad/);
  assert.match(octreeSurfacePageShader, /claimPreviousPage\(oldRow,row,slot\)/,
    "row-attached pages must migrate by spatial leaf identity when topology compaction reorders rows");
  assert.match(octreeSurfacePageShader, /leaves\[row\]\.pad=slot/,
    "publication must carry the spatial page slot in the existing leaf record");
  const lifecycle = WebGPUOctreeSurfacePages.prototype.encodeLifecycle.toString();
  assert.doesNotMatch(lifecycle, /dispatchWorkgroupsIndirect\(this\.candidateSource\.countAndDispatch/,
    "candidate control is a storage binding and must not alias INDIRECT usage in the same pass");
  assert.match(lifecycle, /candidateWorkgroups=Math\.ceil\(this\.plan\.leafCapacity\/64\)/,
    "page candidate kernels use a bounded direct GPU launch and guard against the published count");
  const trace = octreeSurfacePageShader.match(/fn traceSparseDeparture[\s\S]*?return Backtrace\(departure,segments,true\);/)?.[0] ?? "";
  assert.equal(trace.match(/departure-=velocity\*segmentDt/g)?.length, 1,
    "each trajectory segment must apply its displacement exactly once");
  assert.match(octreeSurfacePageShader, /slot==INVALID\|\|slot>=params\.shape\.y\|\|!leafContains\(leaf,p\)/,
    "out-of-owner samples must use the affine leaf continuation instead of clamping to a resident page edge");
});

const modulePath = process.env.WEBGPU_NODE_MODULE;

function bytes(data: ArrayBufferView): Uint8Array<ArrayBuffer> {
  const result = new Uint8Array(data.byteLength);
  result.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  return result;
}

test("Dawn allocates, transports and redistances leaf-attached pages", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU surface-page checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]); const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice(); const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => errors.push((event as { error: { message: string } }).error.message));
  const owned: GPUBuffer[] = [];
  const make = (data: ArrayBufferView, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage });
    device.queue.writeBuffer(buffer, 0, bytes(data)); owned.push(buffer); return buffer;
  };
  const leafData = new ArrayBuffer(2 * 48), leafU32 = new Uint32Array(leafData), leafF32 = new Float32Array(leafData);
  // packed origin, size, flags, pad; phi, gradient; velocity, speed.
  leafU32.set([0, 1, 0, 0], 0); leafF32.set([-0.25, 1, 0, 0, 0.1, 0, 0, 0.1], 4);
  leafU32.set([1, 1, 0, 0], 12); leafF32.set([0.75, 1, 0, 0, 0.1, 0, 0, 0.1], 16);
  const leaves = make(new Uint8Array(leafData), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const candidates = make(new Uint32Array([0, OCTREE_SURFACE_STATE.core, 1, OCTREE_SURFACE_STATE.halo]), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const candidateControl = make(new Uint32Array([2, 1, 1, 1]), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT);
  const publishedPhi = device.createTexture({
    size: [2, 1, 1], dimension: "3d", format: "r32float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const pages = new WebGPUOctreeSurfacePages(device, {
    leaves, candidates: { candidates, countAndDispatch: candidateControl },
  }, 2, [128, 128, 128], [1, 1, 1], { maximumPages: 2, maximumResidentFraction: 1 }, {
    texture: publishedPhi, dimensions: [2, 1, 1],
  });
  try {
    const shaderModule = device.createShaderModule({ code: octreeSurfacePageShader });
    assert.deepEqual((await shaderModule.getCompilationInfo()).messages.filter((message) => message.type === "error"), []);
    assert.deepEqual((await device.createShaderModule({ code: octreeSurfaceDensePublicationShader }).getCompilationInfo()).messages.filter((message) => message.type === "error"), []);
    const encoder = device.createCommandEncoder();
    pages.encodeLifecycle(encoder); pages.encodeTransport(encoder, 0.1); pages.encodeRedistance(encoder, 4); pages.encodeVolumeCorrection(encoder);
    assert.equal(pages.encodeDensePublication(encoder), true);
    const readback = device.createBuffer({ size: pages.plan.arenaBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const phiReadback = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(pages.arena, 0, readback, 0, pages.plan.arenaBytes);
    encoder.copyTextureToBuffer({ texture: publishedPhi }, { buffer: phiReadback, bytesPerRow: 256, rowsPerImage: 1 }, [2, 1, 1]);
    device.queue.submit([encoder.finish()]); await Promise.all([readback.mapAsync(GPUMapMode.READ), phiReadback.mapAsync(GPUMapMode.READ)]);
    const copy = readback.getMappedRange().slice(0); readback.unmap(); readback.destroy();
    const published = new Float32Array(phiReadback.getMappedRange().slice(0), 0, 2); phiReadback.unmap(); phiReadback.destroy();
    const words = new Uint32Array(copy), floats = new Float32Array(copy);
    assert.equal(words[3], 0, "surface arena must not overflow");
    assert.equal(words[6], 2, "both core and swept-halo pages are active");
    assert.notEqual(words[pages.plan.pageTableOffsetWords], 0xffff_ffff);
    assert.notEqual(words[pages.plan.pageTableOffsetWords + 1], 0xffff_ffff);
    const airRecords = Array.from({ length: pages.plan.airHashCapacity }, (_, slot) => [
      words[pages.plan.airHashOffsetWords + 2 * slot],
      words[pages.plan.airHashOffsetWords + 2 * slot + 1],
    ] as const);
    assert.ok(airRecords.some(([key, encodedRow]) => key === 2 && encodedRow === 0xffff_ffff),
      `the positive-x air cell keeps its exact packed key and incident core row: ${JSON.stringify(airRecords.filter(([key, row]) => key !== 0 || row !== 0))}`);
    const samples = floats.slice(pages.plan.phiAOffsetWords, pages.plan.phiAOffsetWords + 2 * pages.plan.samplesPerPage);
    assert.ok(Array.from(samples).every(Number.isFinite));
    assert.ok(Array.from(published).every(Number.isFinite));
    assert.notDeepEqual(Array.from(published), [0, 0], "active pages must publish evolved phi to the topology texture");
    const diagnostics = await pages.readDiagnostics();
    assert.equal(diagnostics.activePages, 2);
    assert.equal(diagnostics.adapterCandidateRows, 2);
    assert.equal(diagnostics.adapterDispatchX, 1);
    assert.equal(diagnostics.candidatePages, 2, "page-arena candidate telemetry must not be overwritten by adapter dispatch words");
    assert.equal(diagnostics.overflow, false);
    assert.equal(diagnostics.departureOutsideResidentBand, 0);
    assert.ok(Number.isFinite(diagnostics.correctionShiftCells));
    assert.ok(Math.abs(diagnostics.correctionShiftCells) <= 1.5);

    const escapedLeafData = new ArrayBuffer(48), escapedLeafU32 = new Uint32Array(escapedLeafData), escapedLeafF32 = new Float32Array(escapedLeafData);
    escapedLeafU32.set([0, 1, 0, 0], 0);
    escapedLeafF32.set([-0.25, 1, 0, 0, 4, 0, 0, 4], 4);
    const escapedLeaves = make(new Uint8Array(escapedLeafData), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const escapedCandidates = make(new Uint32Array([0, OCTREE_SURFACE_STATE.core]), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const escapedControl = make(new Uint32Array([1, 1, 1, 1]), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT);
    const escapedPages = new WebGPUOctreeSurfacePages(device, {
      leaves: escapedLeaves, candidates: { candidates: escapedCandidates, countAndDispatch: escapedControl },
    }, 1, [2, 1, 1], [1, 1, 1], { maximumPages: 1, maximumResidentFraction: 1, maximumSegments: 8 });
    const escapedEncoder = device.createCommandEncoder();
    escapedPages.encodeLifecycle(escapedEncoder);
    escapedPages.encodeTransport(escapedEncoder, 1); // initializes the newly resident page
    escapedPages.encodeTransport(escapedEncoder, 1); // every departure exits its only resident leaf
    device.queue.submit([escapedEncoder.finish()]);
    const escapedDiagnostics = await escapedPages.readDiagnostics();
    assert.equal(escapedDiagnostics.adapterCandidateRows, 1);
    assert.equal(escapedDiagnostics.adapterDispatchX, 1);
    assert.equal(escapedDiagnostics.departureOutsideResidentBand, 8);
    assert.equal(escapedDiagnostics.overflowCode & (1 << 6), 1 << 6, "resident-band departure closes the page authority gate");
    escapedPages.destroy();

    const overflowPages = new WebGPUOctreeSurfacePages(device, {
      leaves, candidates: { candidates, countAndDispatch: candidateControl },
    }, 2, [128, 128, 128], [1, 1, 1], { maximumPages: 1, maximumResidentFraction: 1 });
    const overflowEncoder = device.createCommandEncoder();
    overflowPages.encodeLifecycle(overflowEncoder); overflowPages.encodeTransport(overflowEncoder, 0.1);
    const overflowReadback = device.createBuffer({ size: overflowPages.plan.arenaBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    overflowEncoder.copyBufferToBuffer(overflowPages.arena, 0, overflowReadback, 0, overflowPages.plan.arenaBytes);
    device.queue.submit([overflowEncoder.finish()]); await overflowReadback.mapAsync(GPUMapMode.READ);
    const overflowCopy = overflowReadback.getMappedRange().slice(0); overflowReadback.unmap(); overflowReadback.destroy();
    const overflowWords = new Uint32Array(overflowCopy), overflowFloats = new Float32Array(overflowCopy);
    assert.notEqual(overflowWords[3], 0, "insufficient residency must publish overflow");
    assert.ok(Array.from(overflowFloats.slice(overflowPages.plan.phiAOffsetWords, overflowPages.plan.phiAOffsetWords + overflowPages.plan.samplesPerPage)).every((value) => value === 0), "overflow must suppress authoritative phi writes");
    overflowPages.destroy();
    await device.queue.onSubmittedWorkDone();
    assert.deepEqual(errors, []);
  } finally {
    await device.queue.onSubmittedWorkDone();
    pages.destroy(); publishedPhi.destroy(); for (const buffer of owned) buffer.destroy(); device.destroy();
  }
});
