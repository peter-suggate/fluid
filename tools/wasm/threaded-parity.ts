import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import type { FluidWasmModule } from "../../lib/physics-wasm/module";
import { NodeWebWorker } from "./node-web-worker.mjs";

const repository = resolve(import.meta.dirname, "../..");
const words = (values: readonly number[]) =>
  Array.from(new Uint32Array(Float32Array.from(values).buffer));

async function load(artifact: "scalar" | "simd" | "threaded") {
  const directory = resolve(repository, `public/wasm/fluid-wasm/${artifact}`);
  const fluidWasm = await import(pathToFileURL(resolve(directory, "fluid_wasm.js")).href) as FluidWasmModule;
  await fluidWasm.default({ module_or_path: await readFile(resolve(directory, "fluid_wasm_bg.wasm")) });
  return fluidWasm;
}

function largeDiagonalFixture() {
  const count = 1024;
  const diagonal = Float32Array.from({ length: count }, (_, i) => 1 + (i % 13) * 0.125);
  const rhs = Float32Array.from({ length: count }, (_, i) => ((i % 29) - 14) * 0.03125);
  return {
    diagonal, rhs, pressure: new Float32Array(count), member: new Uint8Array(count).fill(1),
    graph: JSON.stringify({ rowOffsets: Array.from({ length: count + 1 }, (_, i) => i),
      rowCells: Array.from({ length: count }, (_, i) => i), rowCoefficients: Array(count).fill(1),
      rowWeights: Array.from(diagonal), executionOrder: Array.from({ length: count }, (_, i) => i) }),
    options: JSON.stringify({ maximumIterations: 24, relativeTolerance: 1e-6 }),
  };
}

export async function runThreadedParity() {
  const fixture = largeDiagonalFixture();
  const outputs = new Map<string, { pressure: number[]; residual: number[] }>();
  globalThis.Worker = NodeWebWorker as unknown as typeof Worker;
  globalThis.self = { addEventListener() {}, removeEventListener() {} } as unknown as Window & typeof globalThis;
  for (const artifact of ["scalar", "simd", "threaded"] as const) {
    const fluidWasm = await load(artifact);
    if (artifact === "threaded") {
      if (!fluidWasm.initThreadPool) throw new Error("threaded Wasm has no initThreadPool");
      await fluidWasm.initThreadPool(4);
    }
    const result = JSON.parse(fluidWasm.run_pressure!(fixture.diagonal, fixture.rhs, fixture.pressure,
      fixture.member, fixture.graph, fixture.options)) as { pressure: number[]; residual: number[] };
    outputs.set(artifact, result);
  }
  const scalar = outputs.get("scalar")!;
  for (const artifact of ["simd", "threaded"] as const) {
    const result = outputs.get(artifact)!;
    assert.deepEqual(words(result.pressure), words(scalar.pressure), `${artifact} pressure differs from scalar`);
    assert.deepEqual(words(result.residual), words(scalar.residual), `${artifact} residual differs from scalar`);
  }
  return { cells: fixture.diagonal.length, reductionGroups: fixture.diagonal.length / 64, threads: 4 };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const result = await runThreadedParity();
  process.stdout.write(`threaded pressure parity: ${result.cells} cells, ${result.reductionGroups} groups, `
    + `${result.threads} initialized Rayon workers\n`);
}
