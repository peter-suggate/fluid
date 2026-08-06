import test from "node:test";
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

import {
  SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII,
  SPARSE_BRICK_SCENE_GEOMETRY_FORMATS,
  resolveSparseBrickPayloadLayout,
  sparseBrickLaneStrideBytes,
  sparseBrickLaneVoxelsPerWord,
  sparseBrickSceneDistanceAt,
  sparseBrickSceneFractionAt,
  sparseBrickSceneGeometryCodecWGSL,
} from "../lib/sparse-brick-octree";
import { sparseSceneProxyVoxelizationShaderFor } from "../lib/webgpu-sparse-scene-proxies";
import { liveSvoDerivedBuildWGSLFor, liveSvoRadianceFeedbackWGSLFor } from "../lib/webgpu-svo-live-derived-builder";
import { octreeLiveSceneSceneGeometryFormat } from "../lib/webgpu-octree-sparse-bricks";

const voxels = 1 << 20;

test("each scene geometry format costs the bytes a voxel it claims", () => {
  const widths = { "f32x2": 8, "f16-unorm8": 4 } as const;
  for (const format of SPARSE_BRICK_SCENE_GEOMETRY_FORMATS) {
    const layout = resolveSparseBrickPayloadLayout("dry", voxels, format);
    assert.equal(layout.sceneGeometryFormat, format);
    assert.equal(layout.lanes.sceneGeometry.strideBytes, widths[format], format);
    assert.equal(layout.lanes.sceneGeometry.bytes, voxels * widths[format], format);
    // The owner lane is untouched by this axis; the whole dry voxel is the two.
    assert.equal(layout.bytesPerVoxel, widths[format] + 4, format);
  }
});

test("the channel-format invariant sizes a lane rather than assuming f32", () => {
  assert.equal(sparseBrickLaneStrideBytes(["f32", "f32", "f32", "f32"]), 16);
  assert.equal(sparseBrickLaneStrideBytes(["f32", "f32"]), 8);
  assert.equal(sparseBrickLaneStrideBytes(["u32"]), 4);
  // 24 declared bits straddle a word, so the lane pays for four bytes and the
  // remaining eight are spare rather than a third channel.
  assert.equal(sparseBrickLaneStrideBytes(["f16", "unorm8"]), 4);
  assert.equal(sparseBrickLaneStrideBytes(["snorm8", "unorm8"]), 2);
  assert.equal(sparseBrickLaneStrideBytes(["unorm8"]), 1);
  assert.equal(sparseBrickLaneVoxelsPerWord(8), 1);
  assert.equal(sparseBrickLaneVoxelsPerWord(4), 1);
  assert.equal(sparseBrickLaneVoxelsPerWord(2), 2);
  assert.equal(sparseBrickLaneVoxelsPerWord(1), 4);
});

test("only a dry world may narrow the scene geometry lane", () => {
  // `full` mins the scene distance against the solver's own metres, so a
  // band-relative value there would silently compare unlike quantities.
  assert.throws(() => resolveSparseBrickPayloadLayout("full", voxels, "f16-unorm8"), RangeError);
  assert.doesNotThrow(() => resolveSparseBrickPayloadLayout("full", voxels, "f32x2"));
  assert.throws(() => resolveSparseBrickPayloadLayout("dry", voxels, "snorm16" as never), RangeError);
});

test("the shipped arm is the default and every arm stays reachable from the command line", () => {
  // f16 is the narrowest arm that survives: a 2 B snorm8 arm saturated 4.5 M of
  // the hero garden's 5.3 M voxels and flipped 3 299 normals to the fallback,
  // measured on device, and has been removed. See
  // `octreeLiveSceneSceneGeometryFormat`. This assertion is the thing that would
  // catch the default drifting off the shipped arm.
  assert.equal(octreeLiveSceneSceneGeometryFormat({}), "f16-unorm8");
  assert.equal(octreeLiveSceneSceneGeometryFormat({ FLUID_SVO_SCENE_GEOMETRY: "" }), "f16-unorm8");
  for (const format of SPARSE_BRICK_SCENE_GEOMETRY_FORMATS) {
    assert.equal(octreeLiveSceneSceneGeometryFormat({ FLUID_SVO_SCENE_GEOMETRY: format }), format);
  }
  assert.throws(() => octreeLiveSceneSceneGeometryFormat({ FLUID_SVO_SCENE_GEOMETRY: "f16" }), RangeError);
});

test("f32x2 emits the text that shipped, to the character", () => {
  assert.equal(sparseSceneProxyVoxelizationShaderFor("dry"), sparseSceneProxyVoxelizationShaderFor("dry", "f32x2"));
  assert.equal(liveSvoDerivedBuildWGSLFor("dry"), liveSvoDerivedBuildWGSLFor("dry", undefined, undefined, "f32x2"));
  assert.equal(sparseBrickSceneGeometryCodecWGSL("f32x2"), "");
  // The full profile can never carry a codec: it is refused at the layout and
  // must also be refused at the shader, or a `full` world would compile helpers
  // that read a lane laid out four f32 channels wide.
  for (const format of SPARSE_BRICK_SCENE_GEOMETRY_FORMATS) {
    assert.equal(sparseSceneProxyVoxelizationShaderFor("full", format), sparseSceneProxyVoxelizationShaderFor("full"));
    assert.equal(
      liveSvoDerivedBuildWGSLFor("full", undefined, undefined, format),
      liveSvoDerivedBuildWGSLFor("full"));
  }
});

test("producer and consumer share one definition of the narrowed encoding", () => {
  // The failure this rules out does not fail to compile: a writer and a reader
  // that disagree about a shift render a wrong scene rather than no scene.
  for (const format of ["f16-unorm8"] as const) {
    const codec = sparseBrickSceneGeometryCodecWGSL(format);
    assert.ok(codec.includes("fn sceneGeometryWord("), format);
    for (const source of [
      sparseSceneProxyVoxelizationShaderFor("dry", format),
      liveSvoDerivedBuildWGSLFor("dry", undefined, undefined, format),
      liveSvoRadianceFeedbackWGSLFor("dry", undefined, format),
    ]) {
      assert.ok(source.includes(codec), `${format} shader is missing the shared codec`);
      assert.ok(source.includes("sceneGeometryWord("), `${format} shader does not address through the codec`);
    }
    // No arm may keep addressing the lane as two f32 words alongside the codec.
    assert.ok(!sparseSceneProxyVoxelizationShaderFor("dry", format).includes("output * 2u"), format);
    assert.ok(!liveSvoDerivedBuildWGSLFor("dry", undefined, undefined, format)
      .includes("params.laneOffsets.z+voxel*2u"), format);
  }
});

test("a payload word shared by two invocations is written atomically", () => {
  // Voxels sharing a word are two invocations of one dispatch. A plain
  // read-modify-write of the shared word drops one of them, and nothing about that
  // failure is visible except missing geometry. This used to be a property of a
  // two-voxels-a-word *geometry* arm; that arm is gone, and the sharer is now the
  // occupancy mask, which puts 32 voxels in a word at every geometry width.
  const shared = sparseSceneProxyVoxelizationShaderFor("dry", "f16-unorm8", "occupancy");
  assert.match(shared, /var<storage, read_write> payload: array<atomic<u32>>;/);
  assert.match(shared, /atomicStore\(&payload\[materialOffset\],/);
  // The dense arms must not pay for atomics they cannot race on.
  for (const format of ["f32x2", "f16-unorm8"] as const) {
    const exclusive = sparseSceneProxyVoxelizationShaderFor("dry", format);
    assert.match(exclusive, /var<storage, read_write> payload: array<u32>;/);
    assert.ok(!exclusive.includes("atomicAnd(&payload"), format);
    assert.ok(!exclusive.includes("atomicStore(&payload"), format);
  }
});

test("positive coverage survives the 8-bit fraction", () => {
  // `material != 0` and `fraction > 0` are one predicate (see the voxeliser's
  // own `select(NO_MATERIAL_OWNER, ..., primitiveFraction > 0.0)`), and the
  // opacity pyramid stores `solid > 0.` as a hard bit. Round-to-nearest alone
  // would retire every voxel under 1/510 of coverage.
  for (const format of ["f16-unorm8"] as const) {
    const codec = sparseBrickSceneGeometryCodecWGSL(format);
    assert.match(codec, /select\(0u,max\(1u,u32\(round\(clamped\*255\.0\)\)\),clamped>0\.0\)/, format);
  }
});

test("the emitted codec round-trips on Dawn", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for Dawn validation",
}, async () => {
  // The failure this catches is a writer and a reader that disagree about a
  // shift. Nothing above can see it — the shader compiles, the frame renders,
  // and only the geometry is wrong — so the two halves are exercised here
  // against each other on the device that will run them.
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();

  const cellRadius = 0.5 * Math.sqrt(3) * 0.00625;
  const band = SPARSE_BRICK_SCENE_DISTANCE_BAND_RADII * cellRadius;
  // Interleaved even/odd pairs: every case is written to both halves of a word
  // so a parity mistake cannot cancel out.
  const cases: { distance: number; fraction: number }[] = [
    { distance: 0, fraction: 0 },
    { distance: 0, fraction: 1 },
    { distance: -0.5 * band, fraction: 0.5 },
    { distance: 0.5 * band, fraction: 0.25 },
    { distance: 4 * band, fraction: 0 },            // saturates
    { distance: -4 * band, fraction: 1 },           // saturates
    { distance: 1e20, fraction: 0 },                // the never-written sentinel
    { distance: 0.13 * band, fraction: 1 / 512 },   // rounds to zero without the floor
  ];

  for (const format of ["f16-unorm8"] as const) {
    const pairs = cases.length / 2;
    const code = /* wgsl */ `
${sparseBrickSceneGeometryCodecWGSL(format)}
struct Case{distance:f32,fraction:f32}
@group(0) @binding(0) var<storage,read> cases:array<Case>;
@group(0) @binding(1) var<storage,read_write> lane:array<u32>;
@group(0) @binding(2) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(1) fn roundTrip(@builtin(global_invocation_id) gid:vec3u){
  let pair=gid.x;
  // Two words a pair is enough for either format: one voxel a word, or two.
  let base=pair*2u;
  lane[base]=0u; lane[base+1u]=0u;
  for(var half=0u;half<2u;half+=1u){
    let c=cases[pair*2u+half];
    let word=sceneGeometryWord(base,half);
    lane[word]=(lane[word]&~sceneGeometryMask(half))
      |(packSceneGeometry(c.distance,c.fraction,${cellRadius})<<sceneGeometryShift(half));
  }
  for(var half=0u;half<2u;half+=1u){
    let word=lane[sceneGeometryWord(base,half)];
    result[(pair*2u+half)*2u]=sceneDistanceOf(word,half);
    result[(pair*2u+half)*2u+1u]=sceneFractionOf(word,half);
  }
}`;
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ label: `${format} round trip`, code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    assert.equal(errors.length, 0, errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n"));
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "roundTrip" } });

    const caseData = new Float32Array(cases.flatMap((c) => [c.distance, c.fraction]));
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const caseBuffer = device.createBuffer({ size: caseData.byteLength, usage: storage });
    device.queue.writeBuffer(caseBuffer, 0, caseData);
    const laneBuffer = device.createBuffer({ size: pairs * 2 * 4, usage: storage });
    const resultBuffer = device.createBuffer({ size: cases.length * 2 * 4, usage: storage });
    const staging = device.createBuffer({ size: cases.length * 2 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const laneStaging = device.createBuffer({ size: pairs * 2 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: caseBuffer } },
        { binding: 1, resource: { buffer: laneBuffer } },
        { binding: 2, resource: { buffer: resultBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder({ label: `${format} round trip` });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(pairs);
    pass.end();
    encoder.copyBufferToBuffer(resultBuffer, 0, staging, 0, cases.length * 2 * 4);
    encoder.copyBufferToBuffer(laneBuffer, 0, laneStaging, 0, pairs * 2 * 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const error = await device.popErrorScope();
    assert.equal(error, null, error?.message);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    await laneStaging.mapAsync(GPUMapMode.READ);
    const laneWords = new Uint32Array(laneStaging.getMappedRange().slice(0));
    laneStaging.unmap();

    for (const [index, expected] of cases.entries()) {
      const distance = out[index * 2];
      const fraction = out[index * 2 + 1];
      // Coverage: 8 bits everywhere, with positive coverage floored at 1/255 so
      // `fraction > 0` survives as the occupancy predicate it also is.
      assert.ok(Math.abs(fraction - expected.fraction) <= 1 / 255 + 1e-6,
        `${format} case ${index} fraction ${fraction} vs ${expected.fraction}`);
      assert.equal(fraction > 0, expected.fraction > 0, `${format} case ${index} occupancy flipped`);
      // Metres at half precision, with the sentinel clamped short of infinity.
      const clamped = Math.max(-1024, Math.min(1024, expected.distance));
      assert.ok(Number.isFinite(distance), `${format} case ${index} distance is not finite`);
      assert.ok(Math.abs(distance - clamped) <= Math.abs(clamped) * 2 ** -10 + 1e-9,
        `${format} case ${index} distance ${distance} vs ${clamped}`);
    }
    // The CPU decoders read the very words the device wrote. They are a second
    // definition of the encoding — the ground oracle in the dry render smoke
    // needs one off the device — so they are held against the shader's own
    // unpack here rather than trusted to stay in step by inspection.
    for (const pair of [...Array(pairs).keys()]) {
      const words = laneWords.subarray(pair * 2, pair * 2 + 2);
      for (const half of [0, 1]) {
        const index = pair * 2 + half;
        assert.equal(sparseBrickSceneFractionAt(words, format, half), out[index * 2 + 1],
          `${format} case ${index} CPU fraction disagrees with the shader`);
        assert.equal(sparseBrickSceneDistanceAt(words, format, half), out[index * 2],
          `${format} case ${index} CPU distance disagrees with the shader`);
      }
    }
    caseBuffer.destroy(); laneBuffer.destroy(); resultBuffer.destroy(); staging.destroy(); laneStaging.destroy();
  }
  device.destroy();
});
