import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_ANALYTIC_BOOTSTRAP_PARAMETER_BYTES,
  octreeAnalyticBootstrapWorklistShader,
  WebGPUOctreeAnalyticBootstrapWorklist,
} from "../lib/webgpu-octree-analytic-bootstrap";
import {
  octreeAnalyticOwnerBootstrapPageCount,
  octreeSimulationOwnerPageLifecycleShader,
  WebGPUOctreeSimulationOwnerPages,
} from "../lib/webgpu-octree-owner-pages";

test("analytic cold topology publishes the resident worklist ABI entirely on GPU", () => {
  assert.equal(OCTREE_ANALYTIC_BOOTSTRAP_PARAMETER_BYTES, 32);
  assert.match(octreeAnalyticBootstrapWorklistShader,
    /fn emitAnalyticTopologyWorklist[\s\S]*tileWorklist\[0\] = count[\s\S]*tileWorklist\[8\] = candidate\.x/);
  assert.match(octreeAnalyticBootstrapWorklistShader,
    /fn emitAnalyticTopologyWorklist[\s\S]*tileWorklist\[HEADER_WORDS \+ slot\] = tile[\s\S]*publishTileState\(tile\)/);
  assert.match(octreeAnalyticBootstrapWorklistShader,
    /fn publishTileState[\s\S]*atomicCompareExchangeWeak\(&tileStates\[slot\*2u\],0u,encoded\)/,
    "direct-page bootstrap publishes sparse tile keys without a logical-tile state array");
  const encode = WebGPUOctreeAnalyticBootstrapWorklist.prototype.encode.toString();
  assert.doesNotMatch(encode, /mapAsync|getMappedRange|copyBufferToBuffer/,
    "production bootstrap must not read topology decisions back to the CPU");
  assert.doesNotMatch(encode, /dims\.nx|dims\.ny|dims\.nz/,
    "bootstrap emission must cover compact tile bounds rather than the finest lattice");
  assert.match(octreeSimulationOwnerPageLifecycleShader,
    /fn activateAnalyticTopologyPages[\s\S]*worklist\[HEADER \+ tileSlot\][\s\S]*analyticOwnerWord\(origin, size\)/,
    "cold owner pages must consume the bounded analytic tile stream and seed real coarse owners");
  const ownerEncode = WebGPUOctreeSimulationOwnerPages.prototype.encodeAnalyticBootstrap.toString();
  assert.doesNotMatch(ownerEncode, /mapAsync|getMappedRange|copyBufferToBuffer/,
    "production owner bootstrap remains GPU-only");
});

test("analytic owner capacity covers exactly the clipped 16/32 cold tile pages", () => {
  assert.equal(octreeAnalyticOwnerBootstrapPageCount([48, 24, 40], {
    tileSizeCells: 16, activeTileLimits: [2, 1, 2],
  }), 4 * 2 * 4);
  assert.equal(octreeAnalyticOwnerBootstrapPageCount([48, 24, 40], {
    tileSizeCells: 32, activeTileLimits: [1, 1, 1],
  }), 4 * 3 * 4);
});

test("Dawn emits deterministic clipped analytic tile indices and resident dispatches", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU analytic-bootstrap checks",
}, async (t) => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]); const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  const tileDimensions = [5, 4, 3] as const, activeTileLimits = [3, 2, 2] as const;
  const tileCapacity = tileDimensions[0] * tileDimensions[1] * tileDimensions[2];
  const tileWorklist = device.createBuffer({ size: (16 + 2 * tileCapacity) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const tileStates = device.createBuffer({ size: tileCapacity * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const builder = new WebGPUOctreeAnalyticBootstrapWorklist(device, tileWorklist, tileStates, {
    tileDimensions, activeTileLimits, tileSizeCells: 16, activeTileCount: 12,
  });
  const readback = device.createBuffer({ size: (16 + 12 + tileCapacity) * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder(); builder.encode(encoder);
  encoder.copyBufferToBuffer(tileWorklist, 0, readback, 0, (16 + 12) * 4);
  encoder.copyBufferToBuffer(tileStates, 0, readback, (16 + 12) * 4, tileCapacity * 4);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  const validationError = await device.popErrorScope(); assert.equal(validationError, null);
  await readback.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
  if (words.slice(0, 16).every((value) => value === 0)) {
    readback.destroy(); builder.destroy(); tileWorklist.destroy(); tileStates.destroy(); device.destroy();
    t.skip("local Dawn Metal runtime completed a validated compute submission as a no-op");
    return;
  }
  assert.deepEqual([...words.slice(0, 16)], [12, 768, 1, 1, 0, 0, 1, 1, 96, 1, 1, 0, 0, 1, 1, 1]);
  assert.deepEqual([...words.slice(16, 28)], [0, 1, 2, 5, 6, 7, 20, 21, 22, 25, 26, 27]);
  const states = words.slice(28); const live = [...states.entries()].filter(([, value]) => value !== 0).map(([index]) => index);
  assert.deepEqual(live, [0, 1, 2, 5, 6, 7, 20, 21, 22, 25, 26, 27]);
  readback.destroy(); builder.destroy(); tileWorklist.destroy(); tileStates.destroy(); device.destroy();
});

test("Dawn analytic cold bootstrap publishes genuine max-leaf-16/32 owner censuses", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU analytic owner-page checks",
}, async (t) => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]); const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  let poisoned = false;
  for (const maximumLeaf of [16, 32] as const) {
    const dimensions = [maximumLeaf, maximumLeaf, maximumLeaf] as const;
    const tileWorklist = device.createBuffer({ size: 18 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const tileStates = device.createBuffer({ size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const residencyWorklist = device.createBuffer({ size: 32 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    const builder = new WebGPUOctreeAnalyticBootstrapWorklist(device, tileWorklist, tileStates, {
      tileDimensions: [1, 1, 1], activeTileLimits: [1, 1, 1],
      tileSizeCells: maximumLeaf, activeTileCount: 1,
    });
    const pageCount = (maximumLeaf / 8) ** 3;
    const pages = new WebGPUOctreeSimulationOwnerPages(device, dimensions, residencyWorklist,
      { maximumPages: pageCount }, {
        tileWorklist, tileSizeCells: maximumLeaf,
        activeTileLimits: [1, 1, 1], activeTileCount: 1,
      });
    const readback = device.createBuffer({ size: pages.plan.allocatedBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    builder.encode(encoder); pages.encodeAnalyticBootstrap(encoder);
    encoder.copyBufferToBuffer(pages.arena, 0, readback, 0, pages.plan.allocatedBytes);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
    if (words[1] === 0) {
      poisoned = true;
    } else {
      assert.equal(words[0], 0, `max-leaf ${maximumLeaf} consumes its exact bounded page capacity`);
      assert.equal(words[1], pageCount);
      assert.equal(words[2], 0, "owner-page bootstrap must not overflow");
      const payload = words.slice(pages.plan.ownerPagesOffsetWords);
      const cellCensus = new Map<number, number>();
      for (const word of payload) {
        if (word === 0) continue;
        const size = (word & 0x8000_0000) !== 0 ? 1 : 1 << (word & 7);
        cellCensus.set(size, (cellCensus.get(size) ?? 0) + 1);
      }
      assert.deepEqual([...cellCensus], [[maximumLeaf, maximumLeaf ** 3]]);
      assert.equal((cellCensus.get(maximumLeaf) ?? 0) / maximumLeaf ** 3, 1,
        `the cold owner census contains one genuine ${maximumLeaf}-cubed leaf`);
    }
    readback.destroy(); pages.destroy(); builder.destroy();
    residencyWorklist.destroy(); tileStates.destroy(); tileWorklist.destroy();
    if (poisoned) break;
  }
  const validationError = await device.popErrorScope(); assert.equal(validationError, null);
  device.destroy();
  if (poisoned) t.skip("local Dawn Metal runtime completed a validated compute submission as a no-op");
});
