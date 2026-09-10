import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createSparseAdaptiveMassAtlas, sparseAtlasBrickKey, type SparseBrickResolution } from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { packSparseCM12ResidentTopologyArchetypesForQA,
  packSparseCM12ResidentTopologyTemplatesForQA } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { expandSparseCM12TemplateArchetypesGPU } from "../lib/methods/adaptive-mass/sparse-cm12-template-expansion-gpu";

type Point = readonly [number, number, number];
interface Fixture { name: string; dimensions: Point; bricks: readonly {
  q: Point; span: number; r: SparseBrickResolution; unclipped?: boolean;
}[] }
// Deliberately do not import a .test.ts: its registered CPU tests would run as
// a side effect of importing fixture helpers into this focused GPU gate.
const fixtures: readonly Fixture[] = [
  { name: "single full B8", dimensions: [8, 8, 8], bricks: [{ q: [0, 0, 0], span: 1, r: 2 }] },
  { name: "mixed adjacent B8", dimensions: [16, 8, 8], bricks: [{ q: [0, 0, 0], span: 1, r: 2 }, { q: [1, 0, 0], span: 1, r: 4 }] },
  { name: "clipped adjacent B8", dimensions: [13, 7, 5], bricks: [{ q: [0, 0, 0], span: 1, r: 4 }, { q: [1, 0, 0], span: 1, r: 8 }] },
  { name: "signed full frontier", dimensions: [8, 8, 8], bricks: [{ q: [-1, 0, 0], span: 1, r: 2, unclipped: true }, { q: [0, 0, 0], span: 1, r: 4 }] },
  { name: "immutable macro guard", dimensions: [24, 16, 16], bricks: [{ q: [0, 0, 0], span: 2, r: 2 }, { q: [2, 0, 0], span: 1, r: 4 }] },
  { name: "three accepted rungs", dimensions: [24, 8, 8], bricks: [{ q: [0, 0, 0], span: 1, r: 1 }, { q: [1, 0, 0], span: 1, r: 2 }, { q: [2, 0, 0], span: 1, r: 4 }] },
];

function prepare(fixture: Fixture) {
  const brickDimensions = fixture.dimensions.map(value => Math.ceil(value / 8)) as [number, number, number];
  const atlas = createSparseAdaptiveMassAtlas(fixture.dimensions, fixture.bricks.map(brick => ({
    key: sparseAtlasBrickKey(brick.q, { brickDimensions, signedCoordinates: true }),
    coordinate: brick.q, spanBricks: brick.span, resolution: brick.r, unclipped: brick.unclipped,
    density: Float64Array.from({ length: brick.r ** 3 }, (_, local) => (local % 7) / 7),
    gamma: Float64Array.from({ length: brick.r ** 3 }, (_, local) => .75 + local / 100),
  })), 0, 8, true);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const reference = packSparseCM12ResidentTopologyTemplatesForQA(atlas, grid);
  const compact = packSparseCM12ResidentTopologyArchetypesForQA(atlas, grid);
  assert.ok(compact.gpuExpansion, "fixture must exercise the production GPU expansion recipe");
  return { reference, compact };
}
const cache = new Map<string, ReturnType<typeof prepare>>();
function prepared(fixture: Fixture) {
  let result = cache.get(fixture.name);
  if (!result) { result = prepare(fixture); cache.set(fixture.name, result); }
  return result;
}

function wordLocation(reference: Uint32Array, word: number) {
  const [cells, rows, terms, incidence] = [reference[6]!, reference[7]!, reference[8]!, reference[9]!];
  if (word < cells) return `header word ${word}`;
  if (word < rows) return `cell ${Math.floor((word - cells) / 8)} component ${(word - cells) % 8}`;
  if (word < terms) return `row plane ${Math.floor((word - rows) / reference[3]!)} row ${(word - rows) % reference[3]!}`;
  if (word < incidence) return `term ${Math.floor((word - terms) / 2)} ${((word - terms) & 1) ? "coefficient" : "cell ID"}`;
  if (word < reference.length) return `incidence/catalog tail word ${word - incidence}`;
  return `foreign arena guard ${word - reference.length}`;
}
function exactWords(actual: Uint32Array, reference: Uint32Array, label: string) {
  assert.equal(actual.length, reference.length, `${label}: complete word count`);
  let first = -1;
  for (let word = 0; word < reference.length; word++) if (actual[word] !== reference[word]) { first = word; break; }
  assert.equal(first, -1, `${label}: ${wordLocation(reference, first)} at word ${first}: `
    + `actual 0x${actual[first]?.toString(16)}, expected 0x${reference[first]?.toString(16)}`);
}

for (const fixture of fixtures) test(`${fixture.name}: expansion CPU preparation matches legacy geometry and row terms`, () => {
  const { reference, compact } = prepared(fixture);
  exactWords(compact.words, reference.words, fixture.name);
  assert.equal(compact.cellCount, reference.cellCount);
  assert.equal(compact.rowCount, reference.rowCount);
  const w = reference.words, f = new Float32Array(w.buffer);
  assert.ok(w[2]! > 0 && w[3]! > 0 && w[4]! > 0);
  for (let cell = 0; cell < w[2]!; cell++) {
    const at = w[6]! + 8 * cell;
    assert.ok([0, 1, 2, 3, 4, 5, 6].every(component => Number.isFinite(f[at + component])));
    assert.ok([4, 5, 6].every(component => f[at + component]! > 0));
    assert.equal(f[at + 3], f[at + 4]! * f[at + 5]! * f[at + 6]!, "clipped cell volume uses all three extents");
    assert.ok((w[at + 7]! >>> 5) < fixture.bricks.length, "native metadata identifies an actual leaf");
  }
  for (let row = 0; row < w[3]!; row++) {
    const base = w[7]!, descriptor = w[base + row]!, first = descriptor & 0x7fffff, count = descriptor >>> 23;
    assert.ok(count > 0 && first + count <= w[4]!);
    assert.ok((w[base + w[3]! + row]! >>> 30) < 3, "face axis is physical x/y/z");
    assert.ok(f[base + 3 * w[3]! + row]! > 0, "face retains positive clipped area");
    for (let term = first; term < first + count; term++) {
      assert.ok(w[w[8]! + 2 * term]! < w[2]!, "term addresses a real native cell");
      assert.ok(Number.isFinite(f[w[8]! + 2 * term + 1]), "term coefficient is finite");
    }
  }
});

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const liveDawn = new Set<GPU>();
const guardWords = Uint32Array.from({ length: 256 }, (_, index) =>
  (0xc7000000 | ([0, 64, 128, 255][index % 4]! << 8) | index) >>> 0);

for (const fixture of fixtures) (dawnModule ? test : test.skip)(
  `${fixture.name}: GPU expansion is word-exact and preserves surrounding arena data`,
  { timeout: 120_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", `template-expansion-${fixture.name}`);
    let gpu: GPU | undefined, device: GPUDevice | undefined, output: GPUBuffer | undefined, readback: GPUBuffer | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href); Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); liveDawn.add(gpu!);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
      const { reference, compact } = prepared(fixture);
      const cellBase = compact.words[6]!, incidenceBase = compact.words[9]!;
      const totalWords = reference.words.length + guardWords.length;
      output = device.createBuffer({ label: "GPU-expanded SCMT with foreign arena guard", size: 4 * totalWords,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      readback = device.createBuffer({ label: "complete SCMT acceptance readback", size: output.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      // Expansion owns only native cells, rows and terms. Headers, incidence,
      // structural catalogues and later arena payloads (including solid state)
      // must survive. These bytes test isolation, not solid-boundary physics.
      device.queue.writeBuffer(output, 0, compact.words.buffer as ArrayBuffer, compact.words.byteOffset, 4 * cellBase);
      device.queue.writeBuffer(output, 4 * incidenceBase, compact.words.buffer as ArrayBuffer,
        compact.words.byteOffset + 4 * incidenceBase, 4 * (compact.words.length - incidenceBase));
      device.queue.writeBuffer(output, 4 * reference.words.length, guardWords);
      for (const poison of [0xa5a5a5a5, 0x5a5a5a5a]) {
        // Poisoning every owned output word proves the shader actually writes
        // zero-valued coordinates/coefficients as well as nonzero records.
        device.queue.writeBuffer(output, 4 * cellBase,
          new Uint32Array(incidenceBase - cellBase).fill(poison));
        await expandSparseCM12TemplateArchetypesGPU(device, output, compact.words, compact.gpuExpansion!);
        const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
        device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
        try {
          const actual: Uint32Array = new Uint32Array(readback.getMappedRange());
          exactWords(actual.subarray(0, reference.words.length), reference.words, `${fixture.name} poison ${poison}`);
          assert.deepEqual(actual.subarray(reference.words.length), guardWords, "expansion must not overwrite foreign arena/occupancy guards");
        } finally { readback.unmap(); }
      }
      await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []);
      console.log(JSON.stringify({ fixture: fixture.name, words: reference.words.length,
        cells: reference.cellCount, rows: reference.rowCount, terms: reference.words[4],
        archetypes: compact.gpuExpansion!.archetypeCount, checkedGuardWords: guardWords.length,
        repeatedExpansions: 2, mismatchedWords: 0 }));
    } finally {
      readback?.destroy(); output?.destroy(); device?.destroy(); if (gpu) liveDawn.delete(gpu);
      await releaseWebGPUExclusiveLock();
    }
  });
