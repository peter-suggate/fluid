import { access, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import process from "node:process";
import { execFileSync } from "node:child_process";

const repository = resolve(import.meta.dirname, "../..");
const requested = process.argv.slice(2).filter(value =>
  value === "scalar" || value === "simd" || value === "threaded");
const artifacts = requested.length ? requested : ["scalar", "simd", "threaded"];

execFileSync("cargo", ["+1.96.1", "run", "--quiet", "--manifest-path",
  resolve(repository, "rust/Cargo.toml"), "-p", "fluid-artifact-check", "--",
  ...artifacts.flatMap(artifact => [artifact,
    resolve(repository, "public/wasm/fluid-wasm", artifact, "fluid_wasm_bg.wasm")])],
{ stdio: "inherit" });

const uleb = (bytes, cursor) => {
  let value = 0, shift = 0, byte;
  do { byte = bytes[cursor.at++]; value |= (byte & 0x7f) << shift; shift += 7; } while (byte & 0x80);
  return value >>> 0;
};
const sharedLimits = (bytes, cursor) => {
  const flags = uleb(bytes, cursor);
  uleb(bytes, cursor);
  if (flags & 1) uleb(bytes, cursor);
  return (flags & 2) !== 0;
};
const skipName = (bytes, cursor) => {
  const byteLength = uleb(bytes, cursor);
  cursor.at += byteLength;
};
function hasSharedMemory(bytes) {
  const cursor = { at: 8 };
  while (cursor.at < bytes.length) {
    const id = bytes[cursor.at++], size = uleb(bytes, cursor), end = cursor.at + size;
    if (id === 2) {
      const count = uleb(bytes, cursor);
      for (let i = 0; i < count; i++) {
        skipName(bytes, cursor); skipName(bytes, cursor);
        const kind = bytes[cursor.at++];
        if (kind === 0) uleb(bytes, cursor);
        else if (kind === 1) { cursor.at++; sharedLimits(bytes, cursor); }
        else if (kind === 2 && sharedLimits(bytes, cursor)) return true;
        else if (kind === 3) cursor.at += 2;
        else if (kind === 4) { cursor.at++; uleb(bytes, cursor); }
      }
    } else if (id === 5) {
      const count = uleb(bytes, cursor);
      for (let i = 0; i < count; i++) if (sharedLimits(bytes, cursor)) return true;
    }
    cursor.at = end;
  }
  return false;
}
for (const artifact of artifacts) {
  const directory = resolve(repository, "public/wasm/fluid-wasm", artifact);
  const jsPath = resolve(directory, "fluid_wasm.js");
  const wasmPath = resolve(directory, "fluid_wasm_bg.wasm");
  await access(jsPath);
  const [javascript, wasmBuffer] = await Promise.all([readFile(jsPath, "utf8"), readFile(wasmPath)]);
  const wasm = new Uint8Array(wasmBuffer);
  if (!wasmBuffer.subarray(0, 4).equals(Buffer.from([0, 97, 115, 109]))) {
    throw new Error(`${artifact} fluid_wasm_bg.wasm has no WebAssembly header`);
  }
  for (const symbol of ["FluidWorld", "from_scene", "run_pressure"]) {
    if (!javascript.includes(symbol)) throw new Error(`${artifact} glue does not expose ${symbol}`);
  }
  if (artifact === "threaded") {
    if (!javascript.includes("initThreadPool")) throw new Error("threaded glue does not expose initThreadPool");
    if (!hasSharedMemory(wasm)) throw new Error("threaded artifact does not declare shared linear memory");
  } else if (hasSharedMemory(wasm)) {
    throw new Error(`${artifact} fallback unexpectedly requires shared linear memory`);
  }
  process.stdout.write(`validated ${artifact} fluid Wasm artifact (${wasm.byteLength} bytes)\n`);
}
