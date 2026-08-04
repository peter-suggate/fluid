import assert from "node:assert/strict";
import test, { after } from "node:test";
import { pathToFileURL } from "node:url";

import { encodeSvoBrickOccupancy } from "../lib/svo-brick-occupancy";
import { FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE } from "../lib/webgpu-device-limits";
import { SVO_GBUFFER_RENDER_TARGET_CONTRACT } from "../lib/webgpu-svo-gbuffer-targets";
import {
  createSvoBrickRasterCullWGSL,
  svoBrickRasterCullBindGroupLayoutEntries,
  svoBrickRasterCoverageBindGroupLayoutEntries,
  svoBrickRasterCoverageCandidateBytes,
  svoBrickRasterCoverageCountBytes,
  svoBrickRasterDrawBindGroupLayoutEntries,
  svoBrickRasterInstanceBytes,
  svoBrickRasterPublicationInstanceOffsetBytes,
  svoBrickRasterSortStateBytes,
  SVO_BRICK_RASTER_BOX_CORNERS,
  SVO_BRICK_RASTER_CONTRACT,
} from "../lib/webgpu-svo-brick-raster";
import { createSvoDrySceneFragmentWGSL, sparseVoxelDrySceneBindGroupLayoutEntries,
  SVO_DRY_SPLIT_GEOMETRY_FORMAT, SVO_DRY_SPLIT_IDENTITY_FORMAT,
  SVO_DRY_TRAVERSAL_MODES, SVO_SCENE_PRIMITIVE_RASTER_CONTRACT,
  type SvoDryOptimizationExperiments } from "../lib/webgpu-svo-dry-scene";

const cullShader = createSvoBrickRasterCullWGSL({ reversedZNear_m: 0.01 });
// Layout builders read the WebGPU stage flags, which only exist once a device
// module has installed its globals.
if (typeof globalThis.GPUShaderStage === "undefined") {
  Object.defineProperty(globalThis, "GPUShaderStage", {
    configurable: true, writable: true, value: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
  });
}

test("brick instances are sized, keyed and bound so the sorted list reaches the vertex stage", () => {
  assert.equal(SVO_BRICK_RASTER_CONTRACT.instanceStrideBytes, 32);
  assert.equal(SVO_BRICK_RASTER_CONTRACT.verticesPerInstance, 36);
  assert.equal(svoBrickRasterInstanceBytes(1024), 1024 * 32);
  assert.equal(svoBrickRasterCoverageCountBytes(1500, 1500), 1500 * 1500 * 4);
  assert.equal(svoBrickRasterCoverageCandidateBytes(1500, 1500),
    1500 * 1500 * 4 * SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel);
  assert.equal(svoBrickRasterSortStateBytes(), (8 + SVO_BRICK_RASTER_CONTRACT.sortBuckets) * 4);
  assert.equal(svoBrickRasterPublicationInstanceOffsetBytes(), 4_352);
  assert.match(cullShader, /_instanceAlignment:array<u32,56>/,
    "WGSL and host must agree on the aligned start of the instance runtime array");
  // Sort key and node index share one word; the key must not eat a real node.
  assert.equal(SVO_BRICK_RASTER_CONTRACT.nodeIndexMask + 1, 2 ** SVO_BRICK_RASTER_CONTRACT.sortKeyShift);
  assert.ok(SVO_BRICK_RASTER_CONTRACT.sortBuckets
    <= 2 ** (32 - SVO_BRICK_RASTER_CONTRACT.sortKeyShift), "sort key must fit above the node index");
  assert.equal(SVO_BRICK_RASTER_CONTRACT.sortBuckets % SVO_BRICK_RASTER_CONTRACT.scanWorkgroupSize, 0,
    "the single-workgroup scan divides the histogram evenly across its lanes");

  // Read-write storage is illegal in the vertex stage, so the sorted list is a
  // second, read-only binding rather than a second use of the cull group.
  const cull = svoBrickRasterCullBindGroupLayoutEntries();
  assert.equal(cull.length, 5);
  assert.equal(cull.filter((entry) => entry.buffer?.type === "storage" || entry.buffer?.type === "read-only-storage").length, 3);
  assert.ok(cull.every((entry) => entry.visibility === GPUShaderStage.COMPUTE));
  const draw = svoBrickRasterDrawBindGroupLayoutEntries();
  assert.deepEqual(draw, [{
    binding: SVO_BRICK_RASTER_CONTRACT.instanceDrawBinding,
    visibility: GPUShaderStage.VERTEX,
    buffer: { type: "read-only-storage" },
  }]);
  assert.deepEqual(svoBrickRasterCoverageBindGroupLayoutEntries(), [
    { binding: SVO_BRICK_RASTER_CONTRACT.instanceDrawBinding,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" } },
    {
      binding: SVO_BRICK_RASTER_CONTRACT.coverageCountBinding,
      visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    {
      binding: SVO_BRICK_RASTER_CONTRACT.coverageCandidateBinding,
      visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
  ]);
  assert.equal(SVO_BRICK_RASTER_CONTRACT.colorAttachmentBytesPerSample,
    FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE);
  assert.equal(SVO_BRICK_RASTER_CONTRACT.colorAttachmentBytesPerSample, 16 + 8 + 16 + 8);
});

test("the box table draws outward-wound faces so back-face culling keeps the far side", () => {
  assert.equal(SVO_BRICK_RASTER_BOX_CORNERS.length, SVO_BRICK_RASTER_CONTRACT.verticesPerInstance);
  const corner = (index: number): readonly [number, number, number] =>
    [index & 1, (index >> 1) & 1, (index >> 2) & 1];
  const centre = [0.5, 0.5, 0.5];
  for (let triangle = 0; triangle < 12; triangle += 1) {
    const [a, b, c] = [0, 1, 2].map((vertex) => corner(SVO_BRICK_RASTER_BOX_CORNERS[triangle * 3 + vertex]));
    const edge0 = [0, 1, 2].map((axis) => b[axis] - a[axis]);
    const edge1 = [0, 1, 2].map((axis) => c[axis] - a[axis]);
    const normal = [
      edge0[1] * edge1[2] - edge0[2] * edge1[1],
      edge0[2] * edge1[0] - edge0[0] * edge1[2],
      edge0[0] * edge1[1] - edge0[1] * edge1[0],
    ];
    assert.ok(normal.some((component) => component !== 0), `triangle ${triangle} is degenerate`);
    const outward = [0, 1, 2].reduce((sum, axis) => sum + normal[axis] * (a[axis] - centre[axis]), 0);
    assert.ok(outward > 0, `triangle ${triangle} winds inward`);
  }
  // Back faces are what survive: the camera may sit inside a brick.
  assert.equal(SVO_BRICK_RASTER_CONTRACT.cullMode, "back");
});

test("emission drops empty bricks, rasterizes the published occupied sub-AABB, and keys on view depth", () => {
  assert.match(cullShader, /if\(occupancy\.ready!=0u&&occupancy\.occupied==0u\)\{atomicAdd\(&svoBrickRaster\.sort\.empty,1u\);return;\}/);
  assert.match(cullShader, /if\(occupancy\.ready!=0u\)\{proxy=svoBrickOccupiedBounds\(occupancy,bounds\[0\],cellSize\);\}/);
  assert.match(cullShader, /if\(!svoBrickFrustumVisible\(camera,proxy\)\)\{atomicAdd\(&svoBrickRaster\.sort\.culled,1u\);return;\}/);
  assert.match(cullShader, /nodeIndex\|\(key<<SVO_BRICK_SORT_KEY_SHIFT\)/);
  // The key is the box's exact minimum view depth, not a centre distance.
  assert.match(cullShader, /dot\(center-camera\.origin,camera\.forward\)-dot\(abs\(camera\.forward\),halfExtent\)/);
  assert.match(cullShader, /if\(!svoBrickTopologyPublished\(\)\)\{return;\}/,
    "an unpublished topology must emit nothing rather than read stale nodes");
  assert.doesNotMatch(cullShader, /materialOwners|payload\[/,
    "occupancy comes from the published node word, never from a per-frame payload scan");
});

test("the raster-primary fragment reuses the production leaf tracer and writes depth-tested planes", () => {
  const shader = createSvoDrySceneFragmentWGSL(1, "raster-primary", "off", "split", 0, false, true, true);
  assert.ok(SVO_DRY_TRAVERSAL_MODES.includes("raster-primary"));
  assert.match(shader, /@fragment fn svoBrickRasterFragment/);
  assert.match(shader, /@vertex fn svoBrickRasterVertex/);
  assert.match(shader, /@fragment fn dryRasterPrimaryBackgroundMain/);
  // The payload DDA must be the production one, not a second copy.
  assert.match(shader, /let payload=traceLeafPayload\(ro,rd,leaf\);/);
  const fragment = shader.slice(shader.indexOf("@fragment fn svoBrickRasterFragment"));
  assert.doesNotMatch(fragment, /textureStore/,
    "every primary plane is a depth-tested colour attachment, so nothing escapes the depth test");
  assert.doesNotMatch(fragment, /traceStatic|nearestBody|traceGlass/,
    "the brick fragment carries no octree stack, rigid loop or pane loop");
  // Constant clip-space z with w = view depth is the reversed-Z infinite-far projection.
  assert.match(shader, /position=vec4f\(dot\(relative,right\)\/\(aspect\*cameraTanHalfFov\(\)\),dot\(relative,up\)\/cameraTanHalfFov\(\),DRY_REVERSED_Z_NEAR_M,viewDepth\);/);
  assert.match(shader, /struct DryRasterPrimaryOut\{[^}]*@location\(2\) geometry:vec4f,[^}]*@location\(3\) opaqueIdentity:vec2u,/);
  // Background and terrain only; the primary trace never runs full-screen here.
  assert.match(shader, /let terrain=traceTerrain\(camera\[0\],rd\);/);

  // Inline variants of the same module (the reduced cone prepass) legitimately
  // keep traversing, so they must omit the raster entries rather than fail.
  const inline = createSvoDrySceneFragmentWGSL(0.5, "raster-primary", "off", "inline");
  assert.doesNotMatch(inline, /svoBrickRasterFragment/);
  assert.throws(() => createSvoDrySceneFragmentWGSL(1, "raster-primary", "off", "split"),
    /requires raster glass discovery/);
});

test("conservative coverage moves expensive leaf tracing into one exact per-pixel resolve", () => {
  const shader = createSvoDrySceneFragmentWGSL(1, "raster-primary", "off", "split", 0, false, true, true);
  const coverageStart = shader.indexOf(`@fragment fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.coverage}`);
  const resolveHelperStart = shader.indexOf("fn dryBrickCoveragePixel", coverageStart);
  const resolveStart = shader.indexOf(`@fragment fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.resolve}`);
  const overflowStart = shader.indexOf(`@fragment fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.overflowResolve}`);
  const directStart = shader.indexOf("struct DryBrickRasterOut", overflowStart);
  assert.ok(coverageStart > 0 && resolveStart > coverageStart && overflowStart > resolveStart && directStart > overflowStart);
  const coverage = shader.slice(coverageStart, resolveHelperStart);
  const resolve = shader.slice(resolveHelperStart, overflowStart);
  const overflow = shader.slice(overflowStart, directStart);
  assert.match(coverage, /atomicAdd\(&svoBrickCoverageCounts\[pixel\],1u\)/);
  assert.doesNotMatch(coverage, /traceLeafPayload|frag_depth/,
    "the high-overdraw raster stage must only publish conservative coverage");
  assert.match(coverage, /svoBrickCoverageCandidates\[address\]=input\.instanceIndex;\}\s*discard;/,
    "ordinary conservative fragments stop after their cheap candidate write");
  assert.match(resolve, /return dryBrickCoverageResolve\(input\.position\.xy\)/);
  assert.doesNotMatch(resolve, /discard|traceStaticSolidScene/,
    "the normal exact resolve is one fragment per pixel and carries no canonical traversal stack");
  assert.match(shader, /let payload=traceLeafPayload\(ro,rd,leaf\)/,
    "candidate resolution must reuse the production exact leaf tracer");
  assert.match(resolve, /if\(payload\.t<DRY_MISS\).*break;/s,
    "a hit can stop because every tight proxy is contained by a disjoint SVO leaf cell");
  assert.match(overflow, /atomicLoad\(&svoBrickCoverageCounts\[pixel\]\)<=[0-9]+u\)\{discard;\}/);
  assert.match(overflow, /let payload=traceLeafPayload\(ro,rd,leaf\)/,
    "overflow reuses the direct brick arithmetic, so capacity changes cost but never parity");
  assert.doesNotMatch(overflow, /traceStaticSolidScene/,
    "canonical brick-boundary tie arithmetic must not leak into the raster control comparison");
});

test("raster-primary resolves live scene primitive overlap with exact per-primitive depth", () => {
  const shader = createSvoDrySceneFragmentWGSL(1, "raster-primary", "off", "split", 0, false, true, true);
  const { entryPoints } = SVO_SCENE_PRIMITIVE_RASTER_CONTRACT;
  assert.equal(SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.verticesPerProxy, 36);
  assert.match(shader, new RegExp(`@vertex fn ${entryPoints.vertex}`));
  assert.match(shader, new RegExp(`@fragment fn ${entryPoints.fragment}`));
  // Two entry points share one proxy body now: the exact historical set, and
  // the same set behind the near-field band's membership word. The proxy
  // derivation asserted below is the shared body, so the slice starts there
  // rather than at the entry point that used to contain it.
  assert.match(shader, new RegExp(`@vertex fn ${entryPoints.bandVertex}`),
    "the banded proxy is a separate entry point, so the direct control keeps drawing every record");

  const vertexStart = shader.indexOf("fn dryScenePrimitiveProxyVertex(");
  const fragmentStart = shader.indexOf(`@fragment fn ${entryPoints.fragment}`);
  const fragment = shader.slice(fragmentStart, shader.indexOf("\n}", fragmentStart) + 2);
  assert.ok(vertexStart > 0 && fragmentStart > vertexStart);
  assert.match(shader.slice(vertexStart, fragmentStart), /dryPrimitive\(primitiveIndex\)/,
    "each instance derives conservative coverage from its own analytic record");
  // The conservative local extent is the kind table's now, not a copy of it in
  // this shader: it was one of six transcriptions of the same formula, and a
  // bound that is too small here silently clips a silhouette rather than
  // failing. Every kind still has to appear in the dispatch, which is what
  // these assert — against the shared helper's own spelling.
  assert.match(shader.slice(vertexStart, fragmentStart), /svoPrimitiveLocalExtent_m\(svoPrimitiveKind\(record\)/,
    "the vertex stage reads the shared per-kind extent rather than its own");
  for (const kind of [
    "SVO_KIND_SPHERE", "SVO_KIND_BOX", "SVO_KIND_ELLIPSOID", "SVO_KIND_CAPSULE",
    "SVO_KIND_CYLINDER", "SVO_KIND_TORUS", "SVO_KIND_CONE", "SVO_KIND_SMOOTH_UNION_CLUSTER",
  ]) {
    assert.match(shader, new RegExp(`kind == ${kind}`), `${kind} has no arm in the shared local-extent dispatch`);
  }
  assert.match(fragment, /let exact=primitiveHit\(record,ro,rd,span\.x,span\.y\);/,
    "the voxel owner is not authoritative at a projected primitive boundary");
  // A marched kind's 48-iteration ceiling is only sufficient because the caller
  // hands in a bounded interval. This pass used to hand in the whole ray, which
  // left the bounding sphere as the only bracket.
  assert.match(fragment, /let span=dryScenePrimitiveMarchSpan\(record,ro,rd\);/,
    "the march is bracketed by the primitive's own box, not by the whole ray");
  const marchSpan = shader.slice(shader.indexOf("fn dryScenePrimitiveMarchSpan"));
  assert.match(marchSpan.slice(0, marchSpan.indexOf("\n}")),
    /svoPrimitiveLocalExtent_m\(svoPrimitiveKind\(record\)/,
    "the bracket reads the shared per-kind extent, so it cannot bound tighter than the proxy");
  assert.match(fragment, /return drySceneRasterOut\(dryRasterPrimarySurface\(exact,ro,rd,camera\[1\],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE\)\);/,
    "the exact hit publishes all four primary planes, its producing-pass tag and analytic frag_depth together");
  // The indirection above exists so `scenePrimitiveHsrProbe` can drop the depth
  // write. Production must still carry it: without an analytic frag_depth the
  // stored depth is the proxy's rather than the surface's, and every later
  // primary producer tests against the wrong occluder.
  const sceneRasterOut = shader.slice(shader.indexOf("struct DrySceneRasterOut"));
  assert.match(sceneRasterOut.slice(0, sceneRasterOut.indexOf("}")), /@builtin\(frag_depth\) hardwareDepth:f32/,
    "the shipped scene-primitive fragment publishes analytic depth");
  assert.doesNotMatch(fragment, /materialOwners|traceLeafPayload/);

  const primitiveBinding = sparseVoxelDrySceneBindGroupLayoutEntries().find((entry) => entry.binding === 4);
  assert.ok((primitiveBinding!.visibility & GPUShaderStage.VERTEX) !== 0,
    "the conservative proxy vertex can read the primitive arena");
  const paramsBinding = sparseVoxelDrySceneBindGroupLayoutEntries().find((entry) => entry.binding === 9);
  assert.ok((paramsBinding!.visibility & GPUShaderStage.VERTEX) !== 0,
    "the conservative proxy vertex can reject instances beyond the live primitive count");
});

test("the direct brick arm and its depth experiments stay controls off the production path", () => {
  const build = (experiments: SvoDryOptimizationExperiments): string =>
    createSvoDrySceneFragmentWGSL(1, "raster-primary", "off", "split", 0, false, true, true, false, experiments);
  // From the output struct, so the slice covers the frag_depth declaration the
  // brick fragment returns as well as the body that produces it.
  const brickFragment = (shader: string): string => shader.slice(shader.indexOf("struct DryBrickRasterOut"));

  // The historical direct control keeps both. Production uses the conservative
  // coverage/resolve entries above, but retaining this exact arm makes the
  // recorded baseline and its two probes re-measurable.
  const production = brickFragment(build({}));
  assert.match(production, /@builtin\(frag_depth\) hardwareDepth:f32/);
  assert.match(production, /discard/);

  // Dropping frag_depth alone still picks the same winner — brick proxies are
  // disjoint along a ray and an empty brick still discards — so only the stored
  // depth value changes. That is the arm with a sound redesign behind it.
  const noFragmentDepth = brickFragment(build({ rasterPrimaryNoFragmentDepth: true }));
  assert.doesNotMatch(noFragmentDepth, /@builtin\(frag_depth\)/);
  assert.match(noFragmentDepth, /discard/);

  // The upper-bound arm additionally drops discard, which corrupts the image,
  // so it must never be reachable as a rendering mode.
  const probe = brickFragment(build({ rasterPrimaryHsrProbe: true }));
  assert.doesNotMatch(probe, /@builtin\(frag_depth\)/);
  assert.doesNotMatch(probe, /discard/);

  for (const experiments of [{ rasterPrimaryDirect: true }, { rasterPrimaryNoFragmentDepth: true }, { rasterPrimaryHsrProbe: true }]) {
    assert.throws(() => createSvoDrySceneFragmentWGSL(1, "canonical-parametric", "off", "split",
      0, false, false, false, false, experiments), /only apply to raster-primary/);
  }
});

// ---------------------------------------------------------------------------
// GPU: emission, scan and scatter over a synthetic published topology.
// ---------------------------------------------------------------------------
const modulePath = process.env.WEBGPU_NODE_MODULE;
let sharedGpu: GPU | undefined;
let sharedDevice: Promise<GPUDevice> | undefined;

function device(): Promise<GPUDevice> {
  sharedDevice ??= (async () => {
    const { create, globals } = await import(pathToFileURL(modulePath!).href) as {
      create(options: string[]): GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, globals);
    sharedGpu = create(["backend=metal"]);
    const adapter = await sharedGpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter, "no Metal adapter");
    return adapter.requestDevice({ requiredLimits: {
      maxStorageBuffersPerShaderStage: 10,
      maxColorAttachmentBytesPerSample: SVO_BRICK_RASTER_CONTRACT.colorAttachmentBytesPerSample,
    } });
  })();
  return sharedDevice;
}
after(async () => { (await sharedDevice)?.destroy(); sharedGpu = undefined; });

test("the raster-primary exact live-scene primitive entries compile on WebGPU",
  { skip: modulePath ? false : "set WEBGPU_NODE_MODULE" }, async () => {
    const gpuDevice = await device();
    const layout = gpuDevice.createBindGroupLayout({ entries: sparseVoxelDrySceneBindGroupLayoutEntries() });
    for (const [scale, threshold] of [[1, 0], [0.5, 1]] as const) {
      const shaderModule = gpuDevice.createShaderModule({
        code: createSvoDrySceneFragmentWGSL(scale, "raster-primary", "off", "split", threshold, false, true, true),
      });
      const info = await shaderModule.getCompilationInfo();
      assert.deepEqual(info.messages.filter((message) => message.type === "error")
        .map((message) => `${message.lineNum}:${message.linePos} ${message.message}`), []);
      await gpuDevice.createRenderPipelineAsync({
        layout: gpuDevice.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module: shaderModule, entryPoint: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.vertex },
        fragment: { module: shaderModule, entryPoint: SVO_SCENE_PRIMITIVE_RASTER_CONTRACT.entryPoints.fragment, targets: [
          { format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.packedSurfaceFormat },
          { format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.identityMediaFormat },
          { format: SVO_DRY_SPLIT_GEOMETRY_FORMAT },
          { format: SVO_DRY_SPLIT_IDENTITY_FORMAT },
        ] },
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: {
          format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.hardwareDepthFormat,
          depthWriteEnabled: true,
          depthCompare: SVO_GBUFFER_RENDER_TARGET_CONTRACT.depthCompare,
        },
      });
    }
  });

/** Interleave three level-bounded coordinates into the canonical low word. */
function mortonLow(coordinate: readonly [number, number, number], level: number): number {
  let low = 0;
  for (let bit = 0; bit < level; bit += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      low |= ((coordinate[axis] >>> bit) & 1) << (bit * 3 + axis);
    }
  }
  return low >>> 0;
}

test("instance emission reconstructs brick bounds from Morton and orders them front to back",
  { skip: modulePath ? false : "set WEBGPU_NODE_MODULE" }, async () => {
    const gpuDevice = await device();
    const brickSize = 8;
    const maximumDepth = 2;
    const level = 2;
    const cellSize = 1;
    // Four resident leaves in a row along +X. One is published empty, one has
    // no occupancy word at all, and two carry occupied sub-boxes.
    const leaves: { coordinate: [number, number, number]; flags: number }[] = [
      {
        coordinate: [0, 0, 0],
        flags: encodeSvoBrickOccupancy({ ready: true, occupied: true, macroMask: 1, minInclusive: [0, 0, 0], maxInclusive: [3, 3, 3] }),
      },
      { coordinate: [1, 0, 0], flags: encodeSvoBrickOccupancy({ ready: true, occupied: false, macroMask: 0, minInclusive: [0, 0, 0], maxInclusive: [0, 0, 0] }) },
      {
        coordinate: [2, 0, 0],
        flags: encodeSvoBrickOccupancy({ ready: true, occupied: true, macroMask: 0xff, minInclusive: [0, 0, 0], maxInclusive: [7, 7, 7] }),
      },
      { coordinate: [3, 0, 0], flags: 0 },
    ];
    const nodes = new Uint32Array(leaves.length * 8);
    const leafRecords = new Uint32Array(leaves.length * 4);
    leaves.forEach((leaf, index) => {
      nodes[index * 8] = mortonLow(leaf.coordinate, level);
      nodes[index * 8 + 2] = level;
      nodes[index * 8 + 7] = leaf.flags;
      leafRecords[index * 4] = index;
      leafRecords[index * 4 + 1] = index * brickSize ** 3;
    });
    const control = new Uint32Array(32);
    control[0] = leaves.length;
    control[1] = leaves.length;
    control[16] = nodes.length;
    const publication = Uint32Array.of(1, 0xffff_ffff, 1, 1);

    // 400-byte view uniform: viewport, cameraPosition, cameraTarget, ... The
    // camera looks along +X so view depth increases with the brick index.
    const uniforms = new Float32Array(100);
    uniforms.set([256, 256, 0, -1], 0);
    uniforms.set([-40, 16, 16, 0], 4);
    uniforms.set([0, 16, 16, 0], 8);
    const mapping = new ArrayBuffer(48);
    new Float32Array(mapping, 0, 3).set([0, 0, 0]);
    new Uint32Array(mapping, 12, 1)[0] = brickSize;
    new Float32Array(mapping, 16, 3).set([cellSize, cellSize, cellSize]);
    new Uint32Array(mapping, 28, 1)[0] = maximumDepth;
    new Uint32Array(mapping, 32, 4).set([leaves.length, leaves.length, 256, 0]);

    const storage = (data: Uint32Array | Float32Array, usage: number): GPUBuffer => {
      const buffer = gpuDevice.createBuffer({ size: Math.max(16, data.byteLength), usage });
      gpuDevice.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
      return buffer;
    };
    const READ = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const uniformBuffer = storage(uniforms, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const mappingBuffer = gpuDevice.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    gpuDevice.queue.writeBuffer(mappingBuffer, 0, mapping);
    const structureWords = new Uint32Array(128 + nodes.length + leafRecords.length);
    structureWords.set(control, 0); structureWords.set(publication, 64);
    structureWords.set(nodes, 128); structureWords.set(leafRecords, 128 + nodes.length);
    const structureBuffer = storage(structureWords, READ);
    const instanceBytes = svoBrickRasterInstanceBytes(leaves.length);
    const candidates = gpuDevice.createBuffer({ size: instanceBytes, usage: GPUBufferUsage.STORAGE });
    const rasterPublication = gpuDevice.createBuffer({
      size: svoBrickRasterPublicationInstanceOffsetBytes() + instanceBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    gpuDevice.queue.writeBuffer(rasterPublication, 0, Uint32Array.of(SVO_BRICK_RASTER_CONTRACT.verticesPerInstance));

    const layout = gpuDevice.createBindGroupLayout({ entries: svoBrickRasterCullBindGroupLayoutEntries() });
    const module = gpuDevice.createShaderModule({ code: cullShader });
    const info = await module.getCompilationInfo();
    assert.deepEqual(info.messages.filter((message) => message.type === "error").map((message) => message.message), []);
    const pipelineLayout = gpuDevice.createPipelineLayout({ bindGroupLayouts: [layout] });
    const { bindings, entryPoints } = SVO_BRICK_RASTER_CONTRACT;
    const bindGroup = gpuDevice.createBindGroup({
      layout,
      entries: [
        { binding: bindings.uniforms, resource: { buffer: uniformBuffer } },
        { binding: bindings.mapping, resource: { buffer: mappingBuffer } },
        { binding: bindings.structure, resource: { buffer: structureBuffer } },
        { binding: bindings.candidates, resource: { buffer: candidates } },
        { binding: bindings.rasterPublication, resource: { buffer: rasterPublication } },
      ],
    });
    const encoder = gpuDevice.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, bindGroup);
    for (const [entryPoint, groups] of [[entryPoints.emit, 1], [entryPoints.scan, 1], [entryPoints.scatter, 1]] as const) {
      pass.setPipeline(gpuDevice.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } }));
      pass.dispatchWorkgroups(groups);
    }
    pass.end();
    const stateRead = gpuDevice.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const instanceRead = gpuDevice.createBuffer({ size: instanceBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(rasterPublication, 0, stateRead, 0, 32);
    encoder.copyBufferToBuffer(rasterPublication, svoBrickRasterPublicationInstanceOffsetBytes(), instanceRead, 0, instanceBytes);
    gpuDevice.queue.submit([encoder.finish()]);

    await stateRead.mapAsync(GPUMapMode.READ);
    const state = new Uint32Array(stateRead.getMappedRange().slice(0));
    stateRead.unmap();
    assert.equal(state[0], SVO_BRICK_RASTER_CONTRACT.verticesPerInstance, "indirect vertex count survives the frame clear");
    assert.equal(state[1], 3, "one published-empty brick is never drawn");
    assert.equal(state[4], 3, "candidate count matches the draw count");
    assert.equal(state[5], 0, "every brick is inside this frustum");
    assert.equal(state[6], 1, "exactly one brick reported empty");
    assert.equal(state[7], leaves.length, "every leaf was visited");

    await instanceRead.mapAsync(GPUMapMode.READ);
    const raw = instanceRead.getMappedRange().slice(0);
    instanceRead.unmap();
    const floats = new Float32Array(raw);
    const words = new Uint32Array(raw);
    const record = (slot: number) => ({
      minimum: [...floats.subarray(slot * 8, slot * 8 + 3)],
      voxelOffset: words[slot * 8 + 3],
      maximum: [...floats.subarray(slot * 8 + 4, slot * 8 + 7)],
      nodeIndex: words[slot * 8 + 7] & SVO_BRICK_RASTER_CONTRACT.nodeIndexMask,
      key: words[slot * 8 + 7] >>> SVO_BRICK_RASTER_CONTRACT.sortKeyShift,
    });
    const sorted = [0, 1, 2].map(record);
    assert.deepEqual(sorted.map(({ nodeIndex }) => nodeIndex), [0, 2, 3], "front to back along +X, empty brick omitted");
    assert.deepEqual(sorted.map(({ voxelOffset }) => voxelOffset), [0, 1024, 1536]);
    assert.ok(sorted[0].key <= sorted[1].key && sorted[1].key <= sorted[2].key, "sort keys are non-decreasing");
    // Node 0 covers cells [0,8) in world units; its occupied sub-box is [0,4).
    assert.deepEqual(sorted[0].minimum, [0, 0, 0]);
    assert.deepEqual(sorted[0].maximum, [4, 4, 4]);
    // Node 2 is fully occupied, and node 3 has no occupancy word: both keep the whole brick.
    assert.deepEqual(sorted[1].minimum, [16, 0, 0]);
    assert.deepEqual(sorted[1].maximum, [24, 8, 8]);
    assert.deepEqual(sorted[2].minimum, [24, 0, 0]);
    assert.deepEqual(sorted[2].maximum, [32, 8, 8]);
  });

test("the sky lighting entry resolves every value it reads under either glass discovery mode", () => {
  // The sky entry earns its keep by skipping the G-buffer identity load, and the
  // glass key hides in that same plane whenever raster glass discovery is off.
  // Getting that wrong compiles under the production configuration and fails
  // only on the fallback, so both are checked here rather than on a device.
  for (const rasterGlassDiscovery of [true, false]) {
    const shader = createSvoDrySceneFragmentWGSL(0.5, "canonical-parametric", "off", "split", 0, false,
      rasterGlassDiscovery, false, false, {});
    const start = shader.indexOf("@fragment fn drySkyLightingMain");
    assert.ok(start > 0, "the split path must expose a sky-only lighting entry");
    const body = shader.slice(start, shader.indexOf("\n}", start));
    assert.match(body, /var opaque=missHit\(\);/, "the sky entry must not read primary geometry back");
    assert.doesNotMatch(body, /drySplitGeometryAt/, "sky pixels have no surface to decode");
    for (const value of ["packedOpaqueMaterial", "glassKey", "coordinate"]) {
      const declaration = new RegExp(`let ${value}=`);
      const readAnywhereElse = new RegExp(`[^a-zA-Z]${value}[^a-zA-Z=]`).test(body.replace(declaration, ""));
      if (!readAnywhereElse) continue;
      assert.match(body, declaration,
        `${value} is read by the sky entry with rasterGlassDiscovery=${rasterGlassDiscovery} but never declared in it`);
    }
  }
});
