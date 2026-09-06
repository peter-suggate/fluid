import { benchmarkInterleavedPrimary, type PrimaryComputeJob } from "./benchmark-svo-interleaved-compute";
import { sharedFrontierShader } from "./svo-shared-frontier-probe";
import assert from "node:assert/strict";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

/** Benchmark-only capture of public WebGPU descriptors. The default probe calls
 * the production primary entry. Opt-in traversal experiments alter its captured
 * WGSL only; all variants consume every G-buffer output, including depth.
 */
export function capturePrimaryComputeInputs(device: GPUDevice) {
  const shaders = new WeakMap<GPUShaderModule, string>();
  const pipelines = new WeakMap<GPURenderPipeline, GPURenderPipelineDescriptor>();
  const groups = new WeakMap<GPUBindGroup, GPUBindGroupDescriptor>();
  const layouts = new WeakMap<GPUBindGroupLayout, GPUBindGroupLayoutDescriptor>();
  const textures = new WeakMap<GPUTextureView, GPUTexture>();
  const createTexture = device.createTexture.bind(device);
  device.createTexture = (d) => {
    const texture = createTexture(d);
    const createView = texture.createView.bind(texture);
    texture.createView = (vd) => { const view = createView(vd); textures.set(view, texture); return view; };
    return texture;
  };
  const shader = device.createShaderModule.bind(device);
  const pipeline = device.createRenderPipelineAsync.bind(device);
  const group = device.createBindGroup.bind(device);
  const layout = device.createBindGroupLayout.bind(device);
  device.createShaderModule = (d) => { const result = shader(d); shaders.set(result, d.code); return result; };
  device.createRenderPipelineAsync = async (d) => { const result = await pipeline(d); pipelines.set(result, d); return result; };
  device.createBindGroup = (d) => { const result = group(d); groups.set(result, d); return result; };
  device.createBindGroupLayout = (d) => { const result = layout(d); layouts.set(result, d); return result; };

  return async function probe(
    encodeFrame: (encoder: GPUCommandEncoder) => void,
    width: number, height: number, warmups: number, cycles: number, outPath: string,
  ) {
    let selectedPipeline: GPURenderPipeline | undefined;
    let selectedPass: GPURenderPassDescriptor | undefined;
    const selectedGroups = new Map<number, GPUBindGroup>();
    const scratch = device.createCommandEncoder();
    encodeFrame(new Proxy(scratch, {
      get(target, property) {
        if (property === "beginRenderPass") return (descriptor: GPURenderPassDescriptor) => {
          const pass = target.beginRenderPass(descriptor);
          if (descriptor.label !== "Sparse voxel primary visibility") return pass;
          selectedPass = descriptor;
          return new Proxy(pass, {
            get(p, key) {
              if (key === "setPipeline") return (value: GPURenderPipeline) => { selectedPipeline = value; p.setPipeline(value); };
              if (key === "setBindGroup") return (index: number, value: GPUBindGroup) => {
                selectedGroups.set(index, value); p.setBindGroup(index, value);
              };
              const value = Reflect.get(p, key, p);
              return typeof value === "function" ? value.bind(p) : value;
            },
          });
        };
        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    }));
    scratch.finish();
    assert.ok(selectedPipeline);
    assert.ok(selectedPass);
    // Snapshot the *configured* raster variant warmed immediately before this
    // probe. The general frame benchmark captures its reference G-buffer only
    // after switching back to scale 1.
    const attachments = Array.from(selectedPass.colorAttachments);
    const snapshots = [
      { name: "surface", view: attachments[0]!.view, bytes: 16, aspect: "all" as const },
      { name: "identity", view: attachments[1]!.view, bytes: 8, aspect: "all" as const },
      { name: "depth", view: selectedPass.depthStencilAttachment!.view, bytes: 4, aspect: "depth-only" as const },
    ];
    for (const snapshot of snapshots) {
      const texture = textures.get(snapshot.view as GPUTextureView);
      assert.ok(texture, `missing captured ${snapshot.name} texture`);
      const row = Math.ceil(width * snapshot.bytes / 256) * 256;
      const buffer = device.createBuffer({ size: row * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const copy = device.createCommandEncoder();
      copy.copyTextureToBuffer({ texture, aspect: snapshot.aspect }, { buffer, bytesPerRow: row }, [width, height]);
      device.queue.submit([copy.finish()]);
      await buffer.mapAsync(GPUMapMode.READ);
      const mapped = new Uint8Array(buffer.getMappedRange());
      const compact = new Uint8Array(width * snapshot.bytes * height);
      for (let y = 0; y < height; y++) compact.set(mapped.subarray(y * row, y * row + width * snapshot.bytes), y * width * snapshot.bytes);
      const file = `${outPath}-raster-${snapshot.name}.bin`;
      mkdirSync(dirname(file), { recursive: true }); writeFileSync(file, compact);
      buffer.unmap(); buffer.destroy();
    }
    const descriptor = pipelines.get(selectedPipeline)!;
    assert.equal(descriptor.fragment?.entryPoint, "dryVisibilityMain");
    let original = shaders.get(descriptor.fragment!.module)!;
    const sharedPaired = process.env.FLUID_SVO_DRY_FRAME_PRIMARY_SHARED_PAIRED === "1";
    const sharedSweep = sharedPaired || process.env.FLUID_SVO_DRY_FRAME_PRIMARY_SHARED_SWEEP === "1";
    const stackSweep = process.env.FLUID_SVO_DRY_FRAME_PRIMARY_STACK_SWEEP === "1";
    const diagnostic = process.env.FLUID_SVO_DRY_FRAME_PRIMARY_DIAGNOSTIC === "1";
    if (diagnostic) {
      const replaceRequired = (before: string, after: string) => {
        assert.ok(original.includes(before), `diagnostic shader seam missing: ${before}`);
        original = original.replaceAll(before, after);
      };
      original += `
var<private> probeSeed:u32; var<private> probeWinner:u32;
var<private> probeStack:u32; var<private> probeNodes:u32; var<private> probeLeaves:u32;
`;
      replaceRequired("let texel=textureLoad(dryPrimaryEntrySeedRead,coordinate,0);",
        "let texel=textureLoad(dryPrimaryEntrySeedRead,coordinate,0); probeSeed=texel.y;");
      replaceRequired("if(payloadHit.t<seeded.t){voxel=payloadHit;",
        "if(payloadHit.t<seeded.t){probeWinner=leaf.nodeIndex+1u;voxel=payloadHit;");
      replaceRequired("(*continuation).stackSize += 1u;",
        "(*continuation).stackSize += 1u; probeStack=max(probeStack,(*continuation).stackSize);");
      replaceRequired("let leaf=dryTraversalCursorNextPrimary(ray,mapping,&continuation);",
        "let leaf=dryTraversalCursorNextPrimary(ray,mapping,&continuation);probeNodes+=leaf.visits;probeLeaves+=select(0u,1u,leaf.status==SVO_STATUS_HIT);");
    }
    mkdirSync(dirname(outPath), { recursive: true });
    writeFileSync(`${outPath}-primary.wgsl`, original);
    assert.ok(original.includes("@fragment fn dryVisibilityMain("));
    const groupCount = Math.max(...selectedGroups.keys()) + 1;
    const computeLayouts: GPUBindGroupLayout[] = [];
    const computeGroups: GPUBindGroup[] = [];
    for (let i = 0; i < groupCount; i++) {
      const gd = groups.get(selectedGroups.get(i)!)!;
      assert.ok(gd, `missing primary group ${i}`);
      const ld = layouts.get(gd.layout)!;
      const cloned = device.createBindGroupLayout({ ...ld,
        entries: Array.from(ld.entries, (entry) => ({ ...entry, visibility: entry.visibility | GPUShaderStage.COMPUTE })),
      });
      computeLayouts.push(cloned);
      computeGroups.push(device.createBindGroup({ ...gd, layout: cloned }));
    }
    // Offline fixed-view scheduling oracle; all rays still execute every sample.
    const scheduleMap = process.env.FLUID_SVO_DRY_FRAME_PRIMARY_SCHEDULE_MAP;
    assert.ok(Number(sharedSweep) + Number(diagnostic) + Number(stackSweep) + Number(Boolean(scheduleMap)) <= 1, "choose one compute diagnostic/scheduling/stack experiment");
    const tilesX = Math.ceil(width / 8), tilesY = Math.ceil(height / 8);
    const raster = Array.from({ length: tilesX * tilesY }, (_, i) => i);
    const morton = (tile: number) => {
      const x = tile % tilesX, y = Math.floor(tile / tilesX);
      let key = 0;
      for (let bit = 0; bit < 15; bit++) key += ((x >>> bit) & 1) * 2 ** (2 * bit) + ((y >>> bit) & 1) * 2 ** (2 * bit + 1);
      return key;
    };
    const orders = new Map<string, number[]>();
    if (scheduleMap) {
      const report = JSON.parse(readFileSync(scheduleMap, "utf8"));
      const raw = readFileSync(report.primaryWorkMap.rawPath);
      assert.equal(raw.byteLength, width * height * 16, "schedule work-map resolution mismatch");
      const words = new Uint32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
      const scores = raster.map((tile) => {
        let score = 0;
        for (let half = 0; half < 2; half++) {
          let longest = 0;
          for (let dy = half * 4; dy < half * 4 + 4; dy++) for (let dx = 0; dx < 8; dx++) {
            const x = tile % tilesX * 8 + dx, y = Math.floor(tile / tilesX) * 8 + dy;
            if (x < width && y < height) {
              const p = (y * width + x) * 4;
              longest = Math.max(longest, (words[p] & 65535) + words[p + 2]);
            }
          }
          score += longest;
        }
        return score;
      });
      orders.set("raster", raster);
      orders.set("morton", [...raster].sort((a, b) => morton(a) - morton(b)));
      const reverse = (i: number) => { let r = 0; for (let b = 0; b < 16; b++) r = r * 2 + ((i >>> b) & 1); return r; };
      orders.set("spread", [...raster].sort((a, b) => reverse(a) - reverse(b)));
      orders.set("cost-first-oracle", [...raster].sort((a, b) => scores[b] - scores[a] || morton(a) - morton(b)));
      orders.set("raster-repeat", raster);
      for (const order of orders.values()) assert.equal(new Set(order).size, raster.length);
    }
    const orderBuffer = scheduleMap ? device.createBuffer({ size: raster.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }) : undefined;
    const bytes = width * height * 48;
    assert.ok(bytes <= device.limits.maxStorageBufferBindingSize);
    const output = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const outputLayout = device.createBindGroupLayout({ entries: [...(orderBuffer ? [{ binding: 30, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } }] : []), { binding: 31, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
    computeLayouts.push(outputLayout);
    computeGroups.push(device.createBindGroup({ layout: outputLayout, entries: [...(orderBuffer ? [{ binding: 30, resource: { buffer: orderBuffer } }] : []), { binding: 31, resource: { buffer: output } }] }));
    const query = device.createQuerySet({ type: "timestamp", count: 2 });
    const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const timestamps = device.createBuffer({ size: cycles * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const pixels = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const results: Array<{ workgroup: number[]; schedule: string; rawPath: string; median_ms: number; samples_ms: number[] }> = [];
    try {
      const cases = sharedPaired ? ["native", "shared-2-cooperative"].map((schedule) => ({ x: 8, y: 4, schedule })) : sharedSweep ? (process.env.FLUID_SVO_DRY_FRAME_PRIMARY_SHARED_COOPERATIVE === "1" ? ["native", "shared-2-cooperative", "shared-3-cooperative", "native-repeat"] : ["native", "shared-2", "shared-3", "shared-4", "shared-3-forced-fallback", "native-repeat"]).map((schedule) => ({ x: 8, y: 4, schedule })) : stackSweep ? ["stack-32", "stack-24", "stack-20", "stack-32-repeat"].map((schedule) => ({ x: 8, y: 8, schedule })) : diagnostic ? [{ x: 8, y: 8, schedule: "diagnostic" }] : scheduleMap ? [...orders.keys()].map((schedule) => ({ x: 8, y: 8, schedule }))
        : [[8, 4], [8, 8], [16, 8]].map(([x, y]) => ({ x, y, schedule: "native" }));
      const pairedJobs: PrimaryComputeJob[] = [];
      const pipelineCache = new Map<string, GPUComputePipeline>();
      for (const { x, y, schedule } of cases) {
        if (orderBuffer) device.queue.writeBuffer(orderBuffer, 0, new Uint32Array(orders.get(schedule)!));
        const capacity = stackSweep ? Number(schedule.split("-")[1]) : 32;
        const sharedDepth = schedule.startsWith("shared-") ? Number(schedule.split("-")[1]) : 0;
        let variant = stackSweep ? original.replaceAll("array<SvoStackEntry, 32>", `array<SvoStackEntry, ${capacity}>`)
          .replace("const SVO_STACK_CAPACITY: u32 = 32u;", `const SVO_STACK_CAPACITY: u32 = ${capacity}u;`) : original;
        if (sharedDepth) variant = sharedFrontierShader(variant, width, height, x, y, sharedDepth, schedule.includes("forced-fallback") ? 0 : 64, schedule.includes("cooperative"));
        const code = variant.replace("@fragment fn dryVisibilityMain(", "fn dryVisibilityMain(") + `
struct PrimaryComputeResult { surface:vec4u, identity:vec4u, depth:vec4f }
@group(${groupCount}) @binding(31) var<storage,read_write> primaryComputeResult:array<PrimaryComputeResult>;
${orderBuffer ? `@group(${groupCount}) @binding(30) var<storage,read> primaryTileOrder:array<u32>;` : ""}
@compute @workgroup_size(${x},${y}) fn primaryComputeProbe(@builtin(global_invocation_id) globalId:vec3u, @builtin(workgroup_id) groupId:vec3u, @builtin(local_invocation_id) localId:vec3u){
  ${orderBuffer ? `let tile=primaryTileOrder[groupId.y*${tilesX}u+groupId.x];
  let id=vec3u(vec2u(tile%${tilesX}u,tile/${tilesX}u)*vec2u(8u)+localId.xy,0u);` : "let id=globalId;"}
  ${sharedDepth ? `${schedule.includes("cooperative") ? "probeBuildFrontierParallel(groupId,localId);" : "if(localId.x==0u && localId.y==0u){probeBuildFrontier(groupId);}"}
  workgroupBarrier();probeSharedEnabled=true;` : ""}
  if(id.x>=${width}u||id.y>=${height}u){return;}
  let position=vec2f(id.xy)+vec2f(0.5);
  let uv=vec2f(position.x/${width}.0,1.0-position.y/${height}.0);
  var hit=dryVisibilityMain(VertexOut(vec4f(position,0.0,1.0),uv));
  ${sharedDepth ? `if(probeSharedRetry){probeSharedEnabled=false;hit=dryVisibilityMain(VertexOut(vec4f(position,0.0,1.0),uv));}` : ""}
  // Match the raster target's reverse-Z greater test against a zero clear:
  // a miss writes none of its metadata into the cleared attachments.
  var result=PrimaryComputeResult(vec4u(0u),vec4u(0u),vec4f(0.0));
  if(hit.hardwareDepth>0.0){result=PrimaryComputeResult(hit.packedSurface,hit.identityMedia,vec4f(hit.hardwareDepth,0.0,0.0,0.0));}
  ${diagnostic ? `result.depth=vec4f(result.depth.x,bitcast<f32>(probeSeed),bitcast<f32>(probeWinner),bitcast<f32>(min(probeStack,255u)|(min(probeNodes,4095u)<<8u)|(min(probeLeaves,4095u)<<20u)));` : ""}
  ${sharedDepth ? `result.depth.y=bitcast<f32>(probeFrontierCount);result.depth.z=bitcast<f32>(select(0u,1u,probeSharedUsed));result.depth.w=bitcast<f32>(select(0u,1u,probeSharedRetry)|(select(0u,1u,probeFrontierValid==0u)<<1u));` : ""}
  primaryComputeResult[id.y*${width}u+id.x]=result;
}`;
        let cp = pipelineCache.get(`${x}x${y}-${capacity}-${sharedDepth}-${schedule.includes("forced-fallback")}-${schedule.includes("cooperative")}`);
        if (!cp) {
          const module = device.createShaderModule({ label: `Production primary compute ${x}x${y}`, code });
          const compilation = await module.getCompilationInfo();
          assert.deepEqual(compilation.messages.filter((m) => m.type === "error"), []);
          cp = await device.createComputePipelineAsync({ label: `Production primary compute ${x}x${y}`,
            layout: device.createPipelineLayout({ bindGroupLayouts: computeLayouts }), compute: { module, entryPoint: "primaryComputeProbe" } });
          pipelineCache.set(`${x}x${y}-${capacity}-${sharedDepth}-${schedule.includes("forced-fallback")}-${schedule.includes("cooperative")}`, cp);
        }
        if (sharedPaired) { pairedJobs.push({ x, y, schedule, pipeline: cp }); continue; }
        for (let sample = -warmups; sample < cycles; sample++) {
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
          pass.setPipeline(cp);
          computeGroups.forEach((bg, i) => pass.setBindGroup(i, bg));
          pass.dispatchWorkgroups(Math.ceil(width / x), Math.ceil(height / y));
          pass.end();
          if (sample >= 0) {
            encoder.resolveQuerySet(query, 0, 2, resolve, 0);
            encoder.copyBufferToBuffer(resolve, 0, timestamps, sample * 16, 16);
          }
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
        }
        await timestamps.mapAsync(GPUMapMode.READ);
        const ticks = new BigUint64Array(timestamps.getMappedRange());
        const samples_ms = Array.from({ length: cycles }, (_, i) => Number(ticks[2 * i + 1] - ticks[2 * i]) / 1e6);
        timestamps.unmap();
        const sorted = [...samples_ms].sort((a, b) => a - b);
        const file = `${outPath}-${x}x${y}-${schedule}.bin`;
        results.push({ workgroup: [x, y], schedule, rawPath: file, median_ms: sorted[Math.floor(cycles / 2)], samples_ms });
        console.error(`Primary compute ${x}x${y} ${schedule}: ${sorted[Math.floor(cycles / 2)].toFixed(3)} ms`);
        const copy = device.createCommandEncoder();
        copy.copyBufferToBuffer(output, 0, pixels, 0, bytes);
        device.queue.submit([copy.finish()]);
        await pixels.mapAsync(GPUMapMode.READ);
        mkdirSync(dirname(file), { recursive: true });
        writeFileSync(file, new Uint8Array(pixels.getMappedRange()));
        pixels.unmap();
      }
      if (sharedPaired) results.push(...await benchmarkInterleavedPrimary(device, pairedJobs, computeGroups, width, height, output, bytes, warmups, cycles, outPath));
      const surface = readFileSync(`${outPath}-raster-surface.bin`);
      const identity = readFileSync(`${outPath}-raster-identity.bin`);
      const depth = readFileSync(`${outPath}-raster-depth.bin`);
      const surfaceWords = new Uint32Array(surface.buffer, surface.byteOffset, surface.byteLength / 4);
      const identityWords = new Uint16Array(identity.buffer, identity.byteOffset, identity.byteLength / 2);
      const depthWords = new Uint32Array(depth.buffer, depth.byteOffset, depth.byteLength / 4);
      const depthFloats = new Float32Array(depth.buffer, depth.byteOffset, depth.byteLength / 4);
      const checked = results.map((result) => {
        const raw = readFileSync(result.rawPath);
        const words = new Uint32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
        const floats = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
        const diagnosticStats = { seededPixels: 0, voxelHitPixels: 0, seedMatchesWinner: 0,
          seedMissesWinner: 0, seedWithoutVoxelHit: 0, nodeVisits: 0, leafVisits: 0,
          stackHighWaterHistogram: Array<number>(33).fill(0) };
        const sharedStats = { usedPixels: 0, retryPixels: 0, tileOverflowPixels: 0, frontierHistogram: Array<number>(65).fill(0) };
        let differentFromFirst = 0;
        const firstRaw = sharedSweep ? readFileSync(results[0].rawPath) : undefined;
        const firstWords = firstRaw ? new Uint32Array(firstRaw.buffer, firstRaw.byteOffset, firstRaw.byteLength / 4) : undefined;
        let surfacePixels = 0, identityPixels = 0, depthPixels = 0, hitMaskPixels = 0, maximumDepthDelta = 0;
        for (let p = 0; p < width * height; p++) {
          if (firstWords) {
            let changed = false;
            for (let k = 0; k < 9; k++) changed ||= firstWords[p * 12 + k] !== words[p * 12 + k];
            differentFromFirst += Number(changed);
            sharedStats.frontierHistogram[words[p * 12 + 9]]++;
            sharedStats.usedPixels += words[p * 12 + 10];
            sharedStats.retryPixels += words[p * 12 + 11] & 1;
            sharedStats.tileOverflowPixels += (words[p * 12 + 11] >>> 1) & 1;
          }
          if (diagnostic) {
            const seed = words[p * 12 + 9], winner = words[p * 12 + 10], packed = words[p * 12 + 11];
            diagnosticStats.seededPixels += Number(seed !== 0);
            diagnosticStats.voxelHitPixels += Number(winner !== 0);
            diagnosticStats.seedMatchesWinner += Number(winner !== 0 && seed === winner);
            diagnosticStats.seedMissesWinner += Number(winner !== 0 && seed !== winner);
            diagnosticStats.seedWithoutVoxelHit += Number(seed !== 0 && winner === 0);
            diagnosticStats.nodeVisits += (packed >>> 8) & 4095;
            diagnosticStats.leafVisits += packed >>> 20;
            assert.ok((packed & 255) <= 32);
            diagnosticStats.stackHighWaterHistogram[packed & 255]++;
          }
          let surfaceChanged = false, identityChanged = false;
          for (let k = 0; k < 4; k++) {
            surfaceChanged ||= surfaceWords[p * 4 + k] !== words[p * 12 + k];
            // The attachment is rgba16uint, not two u32 identity components.
            identityChanged ||= identityWords[p * 4 + k] !== words[p * 12 + 4 + k];
          }
          surfacePixels += Number(surfaceChanged); identityPixels += Number(identityChanged);
          depthPixels += Number(depthWords[p] !== words[p * 12 + 8]);
          hitMaskPixels += Number((depthWords[p] === 0) !== (words[p * 12 + 8] === 0));
          maximumDepthDelta = Math.max(maximumDepthDelta, Math.abs(depthFloats[p] - floats[p * 12 + 8]));
        }
        return { ...result, sharedStats: sharedSweep ? sharedStats : undefined, pixelsDifferentFromFirst: sharedSweep ? differentFromFirst : undefined, diagnosticStats: diagnostic ? diagnosticStats : undefined, matchesFirstSchedule: (scheduleMap || stackSweep) ? raw.equals(readFileSync(results[0].rawPath)) : undefined, comparison: { pixels: width * height, surfacePixels, identityPixels, depthPixels, hitMaskPixels, maximumDepthDelta } };
      });
      return { interleaved: sharedPaired, instrumented: diagnostic, scope: `${sharedSweep ? "shared upper-tree prototype with production fine traversal" : "primary shader compute probe"}; 48-byte output per pixel; excludes depth bridge`, results: checked };
    } finally {
      orderBuffer?.destroy(); output.destroy(); query.destroy(); resolve.destroy(); timestamps.destroy(); pixels.destroy();
    }
  };
}
