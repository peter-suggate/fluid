#!/usr/bin/env node
/** Isolated production water extraction, on one device in one process.
 *
 * Default (Uniform Geometric): the dense extraction on the Geometric
 * presentation (the solver's 4h vertex base, its ownership, and the detail
 * field while the solver names one), at zero detail (Requested, no region),
 * partial detail (a Fine region) and Full, on a 64^3 and a 128^3 scene.
 * Compute only: classify + prepare is one timed pass, polygonise the next;
 * no raster pass is encoded. An arm pairs a classify scan with a polygonise
 * (see ARMS). Scans:
 *   legacy    the full-lattice scan through the sampler before the 4h base
 *             existed (built here from the shipped shader text; identity
 *             placement only)
 *   before    the listed-window scan with every cube loading its own eight
 *             corners through the owner-aware sampler (the shipped entry's
 *             predecessor, built here)
 *   shipped   a window's vertices and cells formed once in workgroup memory,
 *             an all-h window in straight-line form, a listed window with no
 *             surface stopping after its vertices
 * Polygonisers, by the shipped shader's overrides:
 *   sample    every corner through latticeValue, every normal sample through
 *             umVertexValue, a crossing per triangle corner: the reference
 *   before    uniformNormalGather + shareCubeVertices
 *   kept      + uniformCubeMemo + uniformNormalLoops: the shipped polygonise
 *   floor     no nodal normal at all (the contour normal): what the pass
 *             costs besides the normal; a diagnostic, its mesh is not compared
 *   separable the normal's Gaussian weight as a product of per-axis factors
 *             (built here; it moves every normal by ulps, so it is measured
 *             and not shipped)
 * A pass's time depends on the pass in front of it (the worklist's order),
 * so the polygonisers are compared behind one scan (`before`), and the
 * `shipped` arm is the shipped scan with the shipped polygonise behind it.
 * Every other arm's mesh is compared with `sample`'s: position bits per
 * triangle, normal bits per vertex, the largest normal angle. Pass times are
 * quantised by the timestamp period, so each arm reports the median and the
 * mean of its samples.
 *
 * --render: the fresh-versus-retained mesh presentation timing at 640x360
 * (raster passes; ration it on this machine).
 *
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/benchmark-water-extraction-dawn.ts
 *   [--scenes=id,id] [--render] [--method=adaptive-volume] [--out=PATH] [--wait]
 */
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { createSparseCM12LongDamBreakScene, getSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import type { FluidRefinementRegion, SceneDescription } from "../lib/core/model";
import { resolveMethodValues, type GPUSolverInstance } from "../lib/core/method-contract";
import { createGlobalFineLevelSetConsumerSource } from "../lib/core/octree-consumer-sampling";
import { GPUPassTimestampRecorder } from "../lib/core/performance-trace";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { RasterWaterPipeline, SURFACE_EXTRACTION_VERTEX_CACHE, surfaceExtractionShader } from "../lib/core/webgpu-water-pipeline";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import { uniformDetailExtent } from "../lib/methods/uniform/uniform-detail-fields";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
const flag = (name: string) => process.argv.find(arg => arg.startsWith(`--${name}=`))?.split("=").slice(1).join("=");
const method = flag("method") === "adaptive-volume" ? adaptiveMassMethod : uniformVolumeMethod;
const render = process.argv.includes("--render") || method !== uniformVolumeMethod;
const outputPath = flag("out") ?? `artifacts/water-extraction-${method.id}${render ? "" : "-classify"}.json`;

function packUniform(device: GPUDevice, uniform: GPUBuffer, scene: SceneDescription, dimensions: readonly number[], gridKind: number, viewport: readonly [number, number]) {
  const packed = new Float32Array(100), span = scene.container.width_m;
  packed.set([viewport[0], viewport[1], 0, 0]); packed.set([.6 * span, .65 * span, 1.3 * span, 0], 4);
  packed.set([0, .38 * scene.container.height_m, 0, 0], 8);
  packed.set([span, scene.container.height_m, scene.container.depth_m, scene.container.height_m * scene.container.fillFraction], 12);
  packed.set([0, scene.voxelDomain.finestCellSize_m, 0, 0], 16);
  packed.set([dimensions[0]!, dimensions[1]!, dimensions[2]!, gridKind], 20); packed.set([0, .5, 0, 0], 24);
  device.queue.writeBuffer(uniform, 0, packed);
}

/** The extraction shader as it sampled before the 4h base: every canonical
 * vertex from the h vertex field, one eight-tap loop in the regular sample,
 * the incident-tile walk everywhere, the lattice from that field's extent. */
/** The shipped shader with umVertexValue as the sampler has it: no cube hook. */
function unhookedExtractionShader(): string {
  assert.equal(surfaceExtractionShader.split(SURFACE_EXTRACTION_VERTEX_CACHE).length, 2, "vertex cache anchor");
  return surfaceExtractionShader.replace(SURFACE_EXTRACTION_VERTEX_CACHE, "");
}
function legacyExtractionShader(): string {
  const swap = (source: string, from: string, to: string) => { assert.equal(source.split(from).length, 2, `legacy shader anchor: ${from}`); return source.replace(from, to); };
  const taps = (load: string) => `for(var k=0u;k<umPresentationLoopBound();k++){
   let corner=umCorner(k,2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));
   values[k]=${load}(origin+corner*owner.width)*w.x*w.y*w.z;
  }`;
  let code = swap(unhookedExtractionShader(), "fn umDimensions()->vec3u{return 4u*(textureDimensions(coarseVertexPhi)-vec3u(1u));}", "fn umDimensions()->vec3u{return textureDimensions(denseVertexPhi)-vec3u(1u);}");
  code = swap(code, "fn umLoadCoarseVertex(p:vec3u)->f32{return textureLoad(coarseVertexPhi,vec3i(p>>vec3u(2u)),0).x;}", "fn umLoadCoarseVertex(p:vec3u)->f32{return udrStoredVertex(denseVertexPhi,p);}");
  code = swap(code, "if(all((p&vec3u(3u))==vec3u(0u))){return umLoadCoarseVertex(p);}", "");
  code = swap(code, `if(owner.width==1u){${taps("umLoadFineVertex")}}else{${taps("umLoadCoarseVertex")}}`, taps("umLoadVertex"));
  code = swap(code, "select(umPresentationLoopBound(),0u,uniform&&widest==4u)", "umPresentationLoopBound()");
  code = swap(code, "if(stored&&!(uniform&&widest==4u))", "if(stored)");
  return code;
}

/** The extraction shader with no nodal normal: every vertex keeps its cube's
 * contour normal. What polygonise costs besides the normal. */
function floorExtractionShader(): string {
  const anchor = "fn uniformPhiNormal(lattice:vec3f, fallback:vec3f) -> vec3f {";
  assert.equal(surfaceExtractionShader.split(anchor).length, 2, "floor shader anchor");
  return surfaceExtractionShader.replace(anchor, `${anchor}\n  if(u.gridInfo.x>=0.0){return fallback;}`);
}

/** The unhooked shader with its listed-window entry replaced: every cube
 * loads its own corners through the owner-aware sampler. */
function beforeWindowScanShader(): string {
  const source = unhookedExtractionShader();
  const start = source.indexOf("@compute @workgroup_size(4, 4, 4)\nfn extractWindowsMain("), end = source.indexOf("\n}\n", start);
  assert.ok(start > 0 && end > start, "window scan anchor");
  return `${source.slice(0, start)}@compute @workgroup_size(4, 4, 4)
fn extractWindowsMain(@builtin(workgroup_id) group: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_id) lane: vec3u) {
  udrInit();
  // The production launch: the workgroups share the list out.
  let count = min(atomicLoad(&surfaceWindows.count), arrayLength(&surfaceWindows.windows));
  let each = (count + groups.x - 1u) / groups.x;
  for (var slot = group.x * each; slot < min((group.x + 1u) * each, count); slot += 1u) {
    let window = surfaceWindows.windows[slot];
    let base = vec3i(4u * vec3u(window & 1023u, (window >> 10u) & 1023u, window >> 20u) + lane);
    classifyCube(base);
  }
}
${source.slice(end + 3)}`;
}

/** The normal with a separable Gaussian weight: nine exp per crossing in
 * place of 27, in every loop. Not bit-identical to the shipped normal. */
function separableExtractionShader(): string {
  const weight = "let weight=exp(-0.5*dot(delta,delta)/(.85*.85));", sums = "  var weightSum=0.0;var phiSum=0.0;\n";
  assert.ok(surfaceExtractionShader.split(weight).length > 2, "separable shader anchor: weight");
  assert.equal(surfaceExtractionShader.split(sums).length, 2, "separable shader anchor: sums");
  return surfaceExtractionShader.replaceAll(weight, "let weight=axisWeight[ox+1].x*axisWeight[oy+1].y*axisWeight[oz+1].z;").replace(sums, `  var axisWeight:array<vec3f,3>;
  for(var o=0;o<3;o+=1){let d=vec3f(clamp(center+vec3i(o-1),vec3i(0),dimensions))-x;axisWeight[o]=exp(-0.5*d*d/(.85*.85));}
${sums}`);
}

/** An arm pairs a classify scan with a polygoniser. gather: the normal's
 * samples classified once per cube. share: each crossing evaluated once per
 * cube. memo: a cube's nodal values formed once. loops: the normal's one loop
 * per cube class. */
type Scan = "legacy" | "before" | "shipped";
type Module = "legacy" | "unhooked" | "floor" | "separable" | "production";
interface Arm { readonly id: string; readonly module: Module; readonly scan: Scan; readonly gather: boolean; readonly share: boolean; readonly memo: boolean; readonly loops: boolean }
const BEFORE = { module: "unhooked", scan: "before", gather: true, share: true, memo: false, loops: false } as const;
const KEPT = { module: "production", scan: "before", gather: true, share: true, memo: true, loops: true } as const;
const ARMS: readonly Arm[] = [
  { id: "legacy", module: "legacy", scan: "legacy", gather: false, share: false, memo: false, loops: false },
  { id: "sample", module: "unhooked", scan: "before", gather: false, share: false, memo: false, loops: false },
  { id: "before", ...BEFORE },
  { id: "kept", ...KEPT },
  { id: "floor", ...KEPT, module: "floor", gather: false },
  { id: "separable", ...KEPT, module: "separable" },
  { id: "shipped", ...KEPT, scan: "shipped" },
];
const SCANS: readonly Scan[] = ["legacy", "before", "shipped"];
const COMPARED = ARMS.filter(arm => arm.id !== "sample" && arm.module !== "floor");
const quantile = (values: number[], q: number) => { const s = [...values].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))]! : NaN; };
const median = (values: number[]) => quantile(values, .5);
/** The mean of the middle 80%: sub-period resolution from quantised samples. */
const mean = (values: number[]) => { const s = [...values].sort((a, b) => a - b), cut = Math.floor(.1 * s.length), kept = s.slice(cut, s.length - cut); return kept.length ? kept.reduce((a, b) => a + b, 0) / kept.length : NaN; };
/** Position words first, then normal words, of a 24-word triangle row. */
const TRIANGLE_KEY = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23];
/** Triangles as rows of 24 words, sorted by position: emission order is a race. */
function sortedTriangles(vertices: Float32Array): Uint32Array[] {
  const words = new Uint32Array(vertices.buffer, vertices.byteOffset, vertices.length), rows: Uint32Array[] = [];
  for (let i = 0; i + 24 <= words.length; i += 24) rows.push(words.subarray(i, i + 24));
  return rows.sort((a, b) => { for (const i of TRIANGLE_KEY) if (a[i] !== b[i]) return a[i]! < b[i]! ? -1 : 1; return 0; });
}
/** Triangles whose position bits differ, vertices whose normal bits differ,
 * and the largest angle between corresponding normals. */
function compareMeshes(a: Uint32Array[], b: Uint32Array[]) {
  let positions = Math.abs(a.length - b.length), normals = 0, maxAngle_deg = 0;
  const f = new Float32Array(1), w = new Uint32Array(f.buffer), value = (word: number) => { w[0] = word; return f[0]!; };
  for (let r = 0; r < Math.min(a.length, b.length); r++) {
    const x = a[r]!, y = b[r]!;
    if (TRIANGLE_KEY.slice(0, 12).some(i => x[i] !== y[i])) { positions++; continue; }
    for (const v of [4, 12, 20]) {
      if (x[v] === y[v] && x[v + 1] === y[v + 1] && x[v + 2] === y[v + 2] && x[v + 3] === y[v + 3]) continue;
      normals++;
      const dot = value(x[v]!) * value(y[v]!) + value(x[v + 1]!) * value(y[v + 1]!) + value(x[v + 2]!) * value(y[v + 2]!);
      maxAngle_deg = Math.max(maxAngle_deg, Math.acos(Math.min(1, Math.max(-1, dot))) * 180 / Math.PI);
    }
  }
  return { positions, normals, maxAngle_deg };
}

async function classifyMatrix(device: GPUDevice, errors: string[]) {
  const sceneIds = (flag("scenes") ?? "minimal-power-dam-break-64,high-resolution-dam-break").split(",");
  const rows: Record<string, unknown>[] = [];
  for (const sceneId of sceneIds) {
    const scene = structuredClone(sceneDocument(getSceneDefinition(sceneId))); scene.fluid.refinementRegions = [];
    const solver = await WebGPUUniformReferenceSolver.createAsync(device, scene, "balanced", undefined, uniformGeometricSolverOptions({ detailPolicy: "requested" }, scene), () => {});
    const uniform = device.createBuffer({ size: 400, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bodies = device.createBuffer({ size: 768, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const column = device.createTexture({ size: [1, 1], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING });
    const water = new RasterWaterPipeline(device, "rgba8unorm", uniform, bodies);
    try {
      await water.initialize(); await water.prepareSurfaceCountForQA();
      const [nx, ny, nz] = uniformDetailExtent(solver.volumeTexture);
      packUniform(device, uniform, scene, [nx, ny, nz], 1, [640, 360]);
      water.setVolume(solver.surfaceFieldTexture ?? solver.volumeTexture, solver.columnBaseTexture ?? column, solver.denseLevelSetVolumeSource);
      const modules: Record<Module, GPUShaderModule> = {
        legacy: device.createShaderModule({ label: "Legacy extraction", code: legacyExtractionShader() }),
        unhooked: device.createShaderModule({ label: "Extraction without the cube hook", code: unhookedExtractionShader() }),
        floor: device.createShaderModule({ label: "Extraction without the nodal normal", code: floorExtractionShader() }),
        separable: device.createShaderModule({ label: "Extraction with a separable normal weight", code: separableExtractionShader() }),
        production: device.createShaderModule({ label: "Extraction arms", code: surfaceExtractionShader }),
      };
      const beforeScanModule = device.createShaderModule({ label: "Per-cube window scan", code: beforeWindowScanShader() });
      // Each scan's classifier and its count-only twin.
      const classifiers = new Map<string, GPUComputePipeline>();
      const pipeline = (label: string, module: GPUShaderModule, entryPoint: string, constants: Record<string, number>) =>
        device.createComputePipelineAsync({ label, layout: water.denseExtractionLayoutForQA, compute: { module, entryPoint, constants } });
      for (const scan of SCANS) for (const count of [false, true]) classifiers.set(`${scan}${count ? ":count" : ""}`, await pipeline(`Classify (${scan}${count ? ", count" : ""})`,
        scan === "legacy" ? modules.legacy : scan === "shipped" ? modules.production : beforeScanModule, scan === "legacy" ? "extractMain" : "extractWindowsMain", count ? { countOnly: 1 } : {}));
      const polygonisers = new Map<string, GPUComputePipeline>();
      for (const arm of ARMS) polygonisers.set(arm.id, await pipeline(`Polygonise (${arm.id})`, modules[arm.module], "polygoniseMain",
        { uniformNormalGather: arm.gather ? 1 : 0, shareCubeVertices: arm.share ? 1 : 0, uniformCubeMemo: arm.memo ? 1 : 0, uniformNormalLoops: arm.loops ? 1 : 0 }));
      let frame = 0;
      const step = async (n: number) => { for (let i = 0; i < n; i++) { frame++; assert.ok(solver.advanceTo(frame / 30), `advance ${frame}`); await solver.awaitFrameCompletion(); } };
      const encodeArm = (encoder: GPUCommandEncoder, arm: Arm, countOnly: boolean) => {
        const source = solver.denseLevelSetVolumeSource!;
        // The legacy arm reads the h vertex field at every occupancy; the
        // others bind what the solver names (nothing at zero capacity).
        const old = arm.module === "legacy";
        water.setDenseLevelSetVolumeSource(old ? { ...source, detailVertexPhi: source.vertexPhi } : source);
        water.encodeDenseSurfaceExtractionForQA(encoder, nx, ny, nz, old ? "full" : "windows", countOnly,
          classifiers.get(`${arm.scan}${countOnly ? ":count" : ""}`), polygonisers.get(arm.id));
      };
      const measure = async (label: string) => {
        const source = solver.denseLevelSetVolumeSource!; assert.ok(source.coarseVertexPhi && source.mixedOwnership);
        // Every scan's uncapped vertex count.
        const counts: Record<string, number> = {};
        for (const scan of SCANS) {
          const encoder = device.createCommandEncoder(); encodeArm(encoder, ARMS.find(arm => arm.scan === scan)!, true); device.queue.submit([encoder.finish()]);
          counts[scan] = (await water.readDenseSurfaceExtractionForQA()).vertexCount;
        }
        // A disagreement is reported in the row (and in each arm's mesh comparison), not thrown: the timings still stand.
        assert.ok(counts.before! > 0, `${sceneId} ${label}: no surface: ${JSON.stringify(counts)}`);
        // Every arm's mesh against the sampled normal's: positions and normals, bit for bit.
        const meshes: Record<string, Uint32Array[]> = {};
        for (const arm of [ARMS[1]!, ...COMPARED]) {
          const encoder = device.createCommandEncoder(); encodeArm(encoder, arm, false); device.queue.submit([encoder.finish()]);
          meshes[arm.id] = sortedTriangles((await water.readDenseSurfaceExtractionForQA()).vertices);
        }
        const mesh = Object.fromEntries(COMPARED.map(arm => [arm.id, compareMeshes(meshes[arm.id]!, meshes.sample!)]));
        const classify: Record<string, number[]> = {}, polygonise: Record<string, number[]> = {};
        const SUBMITS = 11, REPEATS = 4;
        for (let submit = 0; submit < SUBMITS; submit++) {
          const recorder = new GPUPassTimestampRecorder(device, 4 * ARMS.length * REPEATS + 8);
          const raw = device.createCommandEncoder(), encoder = recorder.instrument(raw), order: Arm[] = [];
          for (let repeat = 0; repeat < REPEATS; repeat++) for (let a = 0; a < ARMS.length; a++) {
            const arm = ARMS[(a + repeat + submit) % ARMS.length]!; order.push(arm); encodeArm(encoder, arm, false);
          }
          recorder.resolve(raw); device.queue.submit([raw.finish()]); await device.queue.onSubmittedWorkDone();
          const trace = await recorder.read(); assert.ok(trace, "pass timestamps");
          assert.equal(trace.passes.length, 2 * order.length);
          if (submit === 0) continue;
          order.forEach((arm, i) => {
            const c = trace.passes[2 * i]!, p = trace.passes[2 * i + 1]!;
            if (c.sampled) (classify[arm.scan] ??= []).push(c.duration_ms);
            if (p.sampled) (polygonise[arm.id] ??= []).push(p.duration_ms);
          });
        }
        const capacity = (solver as unknown as { mixedFrame: { ownership: { capacity: { fineTiles: number; tiles: number } } } }).mixedFrame.ownership.capacity;
        const stats = await solver.readStats();
        const vertices = counts.before!;
        const row = { scene: sceneId, dimensions: [nx, ny, nz], phase: label, frame, fineCapacity: capacity.fineTiles, tiles: capacity.tiles, vertices, triangles: meshes.sample!.length, meshAgainstSample: mesh,
          scanVertexCounts: counts, scansAgree: SCANS.every(scan => counts[scan] === counts.before),
          classify_ms: Object.fromEntries(SCANS.map(scan => [scan, { median: median(classify[scan] ?? []), mean: mean(classify[scan] ?? []), min: Math.min(...(classify[scan] ?? [NaN])), samples: classify[scan]?.length ?? 0 }])),
          polygonise_ms: Object.fromEntries(ARMS.map(arm => [arm.id, { median: median(polygonise[arm.id] ?? []), mean: mean(polygonise[arm.id] ?? []), p25: quantile(polygonise[arm.id] ?? [], .25), min: Math.min(...(polygonise[arm.id] ?? [NaN])), ns_per_vertex: 1e6 * median(polygonise[arm.id] ?? []) / vertices, mean_ns_per_vertex: 1e6 * mean(polygonise[arm.id] ?? []) / vertices, samples: polygonise[arm.id]?.length ?? 0 }])),
          extraction_ms: Object.fromEntries(ARMS.map(arm => [arm.id, median(classify[arm.scan] ?? []) + median(polygonise[arm.id] ?? [])])),
          simulatedTime_s: stats.simulatedTime_s };
        rows.push(row); console.log(JSON.stringify(row));
      };
      const phase = async (label: string, frames: number) => {
        try { await step(frames); await measure(label); }
        catch (error) { const row = { scene: sceneId, phase: label, frame, error: error instanceof Error ? error.message : String(error) }; rows.push(row); console.log(JSON.stringify(row)); throw error; }
      };
      try {
        await phase("zero", 8);
        const region: FluidRefinementRegion = { id: "fine", rule: "minimum-cell-size", minimumCellSize_cells: 1, maximumCellSize_cells: 1,
          min_m: { x: 0, y: 0, z: -scene.container.depth_m / 2 }, max_m: { x: scene.container.width_m / 2, y: scene.container.height_m / 2, z: scene.container.depth_m / 2 } };
        const drawn = structuredClone(scene); drawn.fluid.refinementRegions = [region]; solver.applySceneUniforms(drawn);
        await solver.pipelinesPrepared();
        await phase("partial", 4);
        solver.applyRuntimeValues({ detailPolicy: "full" });
        // Full's pipelines are compiled on first need.
        await solver.pipelinesPrepared();
        await phase("full", 4);
      } catch { /* the row carries the error; the next scene still runs */ }
    } finally { water.destroy(); uniform.destroy(); bodies.destroy(); column.destroy(); solver.destroy(); }
  }
  assert.deepEqual(errors, []);
  return { method: method.id, scope: "dense water extraction, compute only: classify+prepare pass and polygonise pass, GPU pass timestamps, arms interleaved in one encoder", rows };
}

async function renderSamples(device: GPUDevice, errors: string[]) {
  const scene = createSparseCM12LongDamBreakScene();
  const solver: GPUSolverInstance = await method.createSolverAsync!(device, scene, "balanced", resolveMethodValues(method, "balanced", {}), undefined, () => {});
  let water: RasterWaterPipeline | undefined;
  try {
    await solver.waitForSimulationReady?.();
    const uniform = device.createBuffer({ size: 400, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bodies = device.createBuffer({ size: 768, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const column = device.createTexture({ size: [1, 1], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING });
    const width = 640, height = 360;
    const output = device.createTexture({ size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    water = new RasterWaterPipeline(device, "rgba8unorm", uniform, bodies);
    await water.initialize();
    water.setSceneOptics({ optics: scene.fluid.optics, directional: scene.lighting?.directional, grade: scene.lighting?.grade, terrain: scene.terrain, container: { width_m: scene.container.width_m, depth_m: scene.container.depth_m } });
    // The Geometric presentation: the solver's nodal source, as the renderer binds it.
    water.setVolume(solver.surfaceFieldTexture ?? solver.volumeTexture, solver.columnBaseTexture ?? column, solver.denseLevelSetVolumeSource);
    water.setFluidDomain(solver.fluidDomain);
    const source = solver.globalFineLevelSetSource;
    if (source) water.setGlobalFineLevelSet(createGlobalFineLevelSetConsumerSource(source));
    water.setCoarseLevelSet(solver.coarseLevelSetSource);
    water.ensureSize(width, height);
    packUniform(device, uniform, scene, [solver.info.nx, solver.info.ny, solver.info.nz], solver.info.gridKind === "octree" ? 3 : 1, [width, height]);
    const samples = [];
    for (const fresh of [true, false]) {
      for (let i = 0; i < 10; i++) {
        if (fresh) water.invalidateSurface();
        const recorder: GPUPassTimestampRecorder | undefined = i === 9 ? new GPUPassTimestampRecorder(device, 128) : undefined;
        const begin = performance.now(), raw = device.createCommandEncoder(), encoder = recorder?.instrument(raw) ?? raw;
        const result: ReturnType<RasterWaterPipeline["encode"]> = water.encode(encoder, output, solver.info.nx, solver.info.ny, solver.info.nz, false, solver.info.maximumNeighborDelta ?? 0, 0, undefined, undefined, false, "clear", i === 0, undefined, true);
        assert.ok(result); assert.equal(result.surfaceUpdated, fresh);
        recorder?.resolve(raw);
        device.queue.submit([raw.finish()]);
        const submitted = performance.now();
        await device.queue.onSubmittedWorkDone();
        const complete = performance.now();
        await water.completeSurfaceDiagnostics();
        const trace = await recorder?.read(); recorder?.destroy();
        if (i >= 2) samples.push({ fresh, instrumented: Boolean(recorder), cpu_ms: submitted - begin, completion_ms: complete - submitted, total_ms: complete - begin, trace });
      }
    }
    assert.deepEqual(errors, []);
    const result = { method: method.id, scene: scene.sceneId, dimensions: [solver.info.nx, solver.info.ny, solver.info.nz], scope: "production water pipeline only, 640x360, paused t=0; no SVO lighting or browser", diagnostics: water.surfaceRenderDiagnostics, samples };
    assert.ok(result.diagnostics && result.diagnostics.vertexCount > 0, "the benchmark must render a nonempty mesh");
    return result;
  } finally { water?.destroy(); solver.destroy(); }
}

let device: GPUDevice | undefined;
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits), requiredFeatures: ["timestamp-query"] });
  assert.ok(device);
  const errors: string[] = [];
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  const result = render ? await renderSamples(device, errors) : await classifyMatrix(device, errors);
  await mkdir(dirname(outputPath), { recursive: true }); await writeFile(outputPath, JSON.stringify(result, null, 2));
  if (render) console.log(JSON.stringify(result));
} finally { device?.destroy(); }
