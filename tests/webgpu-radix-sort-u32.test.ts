import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import test from "node:test";
import {
  buildSPGridTouchedDirectory,
  stableRadixSortU32,
  type SPGridTouchedCell,
} from "../lib/octree-spgrid-touched-directory";
import {
  RADIX_SORT_INPUT_VALID,
  WebGPURadixSortU32,
  radixSortU32WGSL,
  spgridTouchedRadixSortEnabled,
  unpackRadixSortU32Control,
} from "../lib/webgpu-radix-sort-u32";
import { PassBroker } from "../lib/webgpu-pass-broker";

test("touched-identity radix sort cutover gate defaults off", () => {
  assert.equal(spgridTouchedRadixSortEnabled({}), false);
  assert.equal(spgridTouchedRadixSortEnabled({ FLUID_SPGRID_TOUCHED_RADIX_SORT: "1" }), true);
  assert.equal(spgridTouchedRadixSortEnabled({ FLUID_SPGRID_TOUCHED_RADIX_SORT: "0" }), false);
});

test("every recurring sort launch is indirect; capacity sizes buffers, never launches", () => {
  const source = readFileSync(new URL("../lib/webgpu-radix-sort-u32.ts", import.meta.url), "utf8");
  const encode = source.slice(source.indexOf("encode(broker: PassBroker"), source.indexOf("destroy(): void"));
  assert.ok(encode.length > 0);
  assert.match(encode, /dispatchWorkgroupsIndirect\(this\.dispatch, 0\)/);
  assert.doesNotMatch(encode, /dispatchWorkgroups\(Math\.ceil/,
    "no launch may be shaped by allocation-time capacity");
  assert.match(encode, /dispatchWorkgroups\(1\)/,
    "fixed-shape stages stay control singletons");
  assert.doesNotMatch(encode, /new PassBroker|beginComputePass/,
    "the sort remains within the caller's command graph");
  assert.match(encode, /broker\.fence\("Prepare stable radix sort"\)/,
    "the GPU-authored indirect record must publish before its consumers");
});

test("the sort publication is fail-closed and epoch-stamped", () => {
  assert.match(radixSortU32WGSL, new RegExp(`header\\[0\\]==${RADIX_SORT_INPUT_VALID}u&&epoch!=0u&&count<=p\\.capacity\\.x`),
    "count past capacity or an unpublished producer must reject the generation");
  assert.match(radixSortU32WGSL, /control\.count=select\(0u,count,valid\)/,
    "an invalid publication zeroes the live count so every launch is indirect-zeroed");
  assert.match(radixSortU32WGSL, /control\.published=select\(0u,control\.epoch,control\.flags==0u\)/,
    "consumers verify published==epoch, never a host attempt stamp");
  assert.doesNotMatch(radixSortU32WGSL,
    /atomic(?:Load|Store|Add|Or|Min|Max|CompareExchange)|atomic<u32>/,
    "digit counts, bases, and run compaction stay deterministic with no scheduling order");
  assert.match(radixSortU32WGSL, /for\(var t=0u;t<lane;t\+=1u\)/,
    "the stable rank counts only earlier same-digit elements of the block");
  assert.match(radixSortU32WGSL, /workgroupUniformLoad\(&chunkBound\)/,
    "storage-derived loop bounds must be laundered before predicating a barrier");
});

test("sorted run boundaries reproduce the oracle directory's brick and page order", () => {
  // The cutover consumes exactly this form: duplicate touched brick identities
  // (one per live cell) sorted with `compactSortedRuns` semantics -- the run
  // values are the oracle's dense-radix brick order, and one more sort over the
  // runs' page identities is the oracle's page order.
  const dimensions = [64, 32, 48] as const;
  // xorshift32: an LCG's low bits cycle within a few hundred coordinate
  // triples and starve the unique-cell loop.
  let state = 0x2468ace0;
  const nextRandom = () => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    return (state >>>= 0);
  };
  const seen = new Set<number>();
  const cells: SPGridTouchedCell[] = [];
  for (let draw = 0; cells.length < 3000; draw += 1) {
    assert.ok(draw < 1_000_000, "cell generation must terminate");
    const x = nextRandom() % dimensions[0], y = nextRandom() % dimensions[1],
      z = nextRandom() % dimensions[2];
    const packed = x + dimensions[0] * (y + dimensions[1] * z);
    if (seen.has(packed)) continue;
    seen.add(packed);
    cells.push({ coordinate: [x, y, z], slot: cells.length });
  }
  const directory = buildSPGridTouchedDirectory(dimensions, cells);
  const brickDims = dimensions.map((value) => Math.ceil(value / 4));
  const identities = cells.map(({ coordinate: [x, y, z] }) =>
    (x >> 2) + brickDims[0]! * ((y >> 2) + brickDims[1]! * (z >> 2)));
  const sorted = stableRadixSortU32(identities);
  const uniqueBricks = sorted.filter((value, index) => index === 0 || sorted[index - 1] !== value);
  assert.deepEqual(uniqueBricks, directory.bricks.map((brick) => brick.dense));
  const pageDims = [Math.ceil(dimensions[0] / 8), Math.ceil(dimensions[1] / 8)];
  const pageIdentities = uniqueBricks.map((dense) => {
    const bx = dense % brickDims[0]!, by = Math.floor(dense / brickDims[0]!) % brickDims[1]!,
      bz = Math.floor(dense / (brickDims[0]! * brickDims[1]!));
    return Math.floor(bx / 2) + pageDims[0]! * (Math.floor(by / 2) + pageDims[1]! * bz);
  });
  const sortedPages = stableRadixSortU32(pageIdentities);
  assert.deepEqual(
    sortedPages.filter((value, index) => index === 0 || sortedPages[index - 1] !== value),
    directory.pages);
});

test("control unpacking validates its word count", () => {
  assert.throws(() => unpackRadixSortU32Control([0, 0, 0]), /eight words/);
  assert.deepEqual(unpackRadixSortU32Control([1, 2, 3, 4, 5, 0, 0, 0]),
    { flags: 1, count: 2, epoch: 3, published: 4, runCount: 5 });
});

test("Dawn sorts GPU-published identity counts to the CPU oracle's stable order", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU sort validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const compilation = await device.createShaderModule({ code: radixSortU32WGSL }).getCompilationInfo();
  assert.deepEqual(compilation.messages.filter((message) => message.type === "error"), []);
  const capacity = 65536;
  const header = device.createBuffer({ size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.pushErrorScope("validation");
  const sorter = new WebGPURadixSortU32(device, capacity, header);
  let epoch = 100;
  const runSort = async (values: readonly number[],
    overrides?: { count?: number; magic?: number; epoch?: number }) => {
    epoch += 1;
    const claimed = overrides?.count ?? values.length;
    const stamped = overrides?.epoch ?? epoch;
    device.queue.writeBuffer(header, 0, new Uint32Array([
      overrides?.magic ?? RADIX_SORT_INPUT_VALID, claimed, stamped, 0]));
    if (values.length) device.queue.writeBuffer(sorter.keys, 0, new Uint32Array(values));
    const keyWords = Math.max(values.length, 8), runWords = 2 * Math.max(values.length, 1);
    const readback = device.createBuffer({ size: 32 + (keyWords + runWords) * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const broker = new PassBroker(device.createCommandEncoder());
    sorter.encode(broker);
    broker.copyBufferToBuffer(sorter.control, 0, readback, 0, 32);
    broker.copyBufferToBuffer(sorter.keys, 0, readback, 32, keyWords * 4);
    broker.copyBufferToBuffer(sorter.runs, 0, readback, 32 + keyWords * 4, runWords * 4);
    device.queue.submit([broker.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const bytes = readback.getMappedRange().slice(0);
    readback.unmap(); readback.destroy();
    return { expectedEpoch: stamped,
      control: unpackRadixSortU32Control(new Uint32Array(bytes, 0, 8)),
      keys: Array.from(new Uint32Array(bytes, 32, keyWords)),
      runs: Array.from(new Uint32Array(bytes, 32 + keyWords * 4, runWords)) };
  };
  const oracleRuns = (sorted: readonly number[]) => sorted.flatMap((value, index) =>
    index === 0 || sorted[index - 1] !== value ? [value, index] : []);
  const assertOracle = async (values: readonly number[]) => {
    const { control, keys, runs, expectedEpoch } = await runSort(values);
    const expected = stableRadixSortU32(values);
    const expectedRuns = oracleRuns(expected);
    assert.deepEqual(
      { flags: control.flags, count: control.count, epoch: control.epoch,
        published: control.published, runCount: control.runCount },
      { flags: 0, count: values.length, epoch: expectedEpoch,
        published: expectedEpoch, runCount: expectedRuns.length / 2 });
    assert.deepEqual(keys.slice(0, values.length), expected);
    assert.deepEqual(runs.slice(0, expectedRuns.length), expectedRuns);
  };

  // Tens of thousands of full-range identities across every byte lane. LSD
  // correctness over multi-byte keys is itself the inter-pass stability proof.
  // xorshift32: an LCG's low bits are periodic under the small moduli below.
  let state = 0x12345678;
  const nextRandom = () => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    return (state >>>= 0);
  };
  await assertOracle(Array.from({ length: 40000 }, nextRandom));
  // Duplicate-heavy identities exercise long equal runs and run compaction.
  const identities = [0, 1, 0xff, 0x100, 0xffff_ffff, 0x8000_0000, 42];
  await assertOracle(Array.from({ length: 10000 },
    () => identities[nextRandom() % identities.length]!));
  // Workgroup-block boundaries, including the partial tail block.
  for (const count of [1, 2, 255, 256, 257, 4095, 4096, 4097]) {
    await assertOracle(Array.from({ length: count }, () => nextRandom() % 1024));
  }
  // A valid empty publication sorts nothing and still stamps its epoch.
  const empty = await runSort([]);
  assert.deepEqual([empty.control.flags, empty.control.count, empty.control.runCount,
    empty.control.published], [0, 0, 0, empty.expectedEpoch]);

  // Fail-closed: a count past capacity rejects the generation and leaves the
  // producer's bank-0 words untouched -- never a partial output.
  const sentinels = [7, 6, 5, 4, 3, 2, 1, 0];
  const overflowing = await runSort(sentinels, { count: capacity + 1 });
  assert.deepEqual([overflowing.control.flags, overflowing.control.count,
    overflowing.control.published, overflowing.control.runCount], [1, 0, 0, 0]);
  assert.deepEqual(overflowing.keys.slice(0, 8), sentinels);
  const wrongMagic = await runSort(sentinels, { magic: 0 });
  assert.deepEqual([wrongMagic.control.flags, wrongMagic.control.published], [1, 0]);
  const zeroEpoch = await runSort(sentinels, { epoch: 0 });
  assert.deepEqual([zeroEpoch.control.flags, zeroEpoch.control.published], [1, 0]);

  const error = await device.popErrorScope();
  assert.equal(error, null, error?.message);
  sorter.destroy();
  header.destroy();
  device.destroy();
});
