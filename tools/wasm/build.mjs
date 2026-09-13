import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { resolve } from "node:path";
import process from "node:process";
import { createHash } from "node:crypto";

const repository = resolve(import.meta.dirname, "../..");
const rustRoot = resolve(repository, "rust");
const manifest = resolve(rustRoot, "crates/fluid-wasm/Cargo.toml");
const outputRoot = resolve(repository, "public/wasm/fluid-wasm");
const stable = "1.96.1";
// Keep the threaded build on a fixed nightly new enough for fluid-core's
// MSRV; rust-src is required because its std must be rebuilt with atomics.
const nightly = "nightly-2025-11-15";
const requested = process.argv.includes("--scalar") ? ["scalar"]
  : process.argv.includes("--simd") ? ["simd"]
  : process.argv.includes("--threaded") ? ["threaded"]
  : process.argv.includes("--single") ? ["scalar", "simd"]
  : ["scalar", "simd", "threaded"];

async function sourceFingerprint() {
  const hash = createHash("sha256");
  async function visit(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (entry.name === "target") continue;
      const path = resolve(directory, entry.name);
      if (entry.isDirectory()) await visit(path);
      else if (/\.(rs|toml)$/.test(entry.name) || entry.name === "Cargo.lock") {
        hash.update(path.slice(rustRoot.length));
        hash.update(await readFile(path));
      }
    }
  }
  await visit(rustRoot);
  return hash.digest("hex");
}
const sourceSha256 = await sourceFingerprint();

async function run(command, args, env = process.env) {
  await new Promise((resolveRun, reject) => {
    const child = spawn(command, args, { cwd: rustRoot, env, stdio: "inherit" });
    child.on("error", reject);
    child.on("exit", code => code === 0 ? resolveRun() : reject(new Error(`${command} exited ${code}`)));
  });
}

async function makeThreadWorkerDirectlyLoadable(output) {
  const snippets = resolve(output, "snippets");
  const packages = await readdir(snippets);
  const rayonPackage = packages.find(name => name.startsWith("wasm-bindgen-rayon-"));
  if (!rayonPackage) throw new Error("threaded artifact has no wasm-bindgen-rayon worker helper");
  const helper = resolve(snippets, rayonPackage, "src/workerHelpers.js");
  const source = await readFile(helper, "utf8");
  const direct = source.replace("import('../../..')", "import('../../../fluid_wasm.js')");
  if (direct === source) throw new Error("wasm-bindgen-rayon worker helper import layout changed");
  await writeFile(helper, direct);
}

await mkdir(outputRoot, { recursive: true });
for (const artifact of requested) {
  const threaded = artifact === "threaded";
  const toolchain = threaded ? nightly : stable;
  const targetFeatures = artifact === "scalar"
    ? "-simd128,-atomics,-bulk-memory"
    : threaded ? "+simd128,+atomics,+bulk-memory,+mutable-globals" : "+simd128,-atomics";
  const environment = {
    ...process.env,
    RUSTFLAGS: `-C target-feature=${targetFeatures}${threaded ? [
      "--import-memory", "--shared-memory", "--export=__heap_base",
      "--export=__wasm_init_tls", "--export=__tls_size", "--export=__tls_align",
      "--export=__tls_base",
      "--max-memory=2147483648",
    ].map(flag => ` -C link-arg=${flag}`).join("") : ""}`,
    ...(threaded ? {
      CARGO_UNSTABLE_BUILD_STD: "std,panic_abort",
    } : {}),
  };
  await run("cargo", [`+${toolchain}`, "build", "--manifest-path", manifest,
    "--target", "wasm32-unknown-unknown", "--release", "--no-default-features",
    "--features", threaded ? "threaded" : "single"], environment);
  const output = resolve(outputRoot, artifact);
  await mkdir(output, { recursive: true });
  await run("wasm-bindgen", [
    resolve(rustRoot, "target/wasm32-unknown-unknown/release/fluid_wasm.wasm"),
    "--target", "web", "--out-dir", output, "--out-name", "fluid_wasm",
  ]);
  if (threaded) await makeThreadWorkerDirectlyLoadable(output);
  if (await sourceFingerprint() !== sourceSha256) {
    throw new Error("Rust sources changed during the Wasm build; rebuild all artifacts from a stable source state");
  }
  await writeFile(resolve(output, "build-info.json"), JSON.stringify({
    sourceSha256, artifact, toolchain, targetFeatures,
    wasmSha256: createHash("sha256").update(await readFile(resolve(output, "fluid_wasm_bg.wasm"))).digest("hex"),
  }, null, 2) + "\n");
}
await run(process.execPath, [resolve(import.meta.dirname, "check-artifacts.mjs"), ...requested]);
