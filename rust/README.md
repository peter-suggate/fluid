# Rust/Wasm physics

The Rust workspace contains the CPU physics authority used by Advance Lab and the selectable Fluid Lab 3D CPU backend. `fluid-core` owns the numerical world and `fluid-wasm` exposes its binary command/publication boundary to the browser worker. The host does not fall back to a JavaScript numerical solver when a Rust artifact cannot load.

Run setup and build commands from the repository root. The scalar and SIMD artifacts use the pinned Rust 1.96.1 toolchain. The threaded artifact uses the pinned `nightly-2025-11-15` toolchain because it rebuilds the standard library with atomics. The generated JavaScript glue must come from the same wasm-bindgen release as the crate:

```sh
rustup toolchain install 1.96.1 --profile minimal --component rust-src --target wasm32-unknown-unknown
rustup toolchain install nightly-2025-11-15 --profile minimal --component rust-src --target wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.122 --locked
npm run build:physics-wasm
```

The build writes three variants under `public/wasm/fluid-wasm`:

- `scalar` disables SIMD, atomics, and bulk memory target features.
- `simd` enables `simd128` and remains single threaded.
- `threaded` enables `simd128`, atomics, bulk memory, and mutable globals, imports shared memory, and initializes a persistent Rayon worker pool through wasm-bindgen-rayon.

All three variants run the same Rust world APIs. Arithmetic excludes relaxed SIMD, and deterministic reduction and commit order is independent of Rayon worker count. The 3D UI continues to select the GPU backend by default; selecting a CPU lane chooses one of these Rust artifacts.

Useful native and headless Node checks are:

```sh
cargo +1.96.1 test --manifest-path rust/Cargo.toml -p fluid-core --features parallel
npm run check:physics-wasm
npm run test:physics-wasm:node
npm run test:physics-wasm:world
npm run test:physics-wasm:world3d
npm run test:physics-wasm:adapter-world3d
npm run test:physics-wasm:threaded
node --import tsx --test lib/physics-wasm/*.test.ts advance-lab/*.test.ts
```

The Node world checks exercise the generated artifacts through their real binary publication boundary; the 3D flow also runs UI defaults with eight workers. The threaded check initializes four Node worker threads and verifies deterministic pressure output against the scalar and SIMD artifacts. Browser integration remains necessary for actual Web Worker bootstrap and shared-memory hosting.

The threaded browser lane requires `SharedArrayBuffer` and `crossOriginIsolated === true`. Serve its page with these response headers:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The host rejects the threaded selection when those requirements are absent. Scalar and SIMD do not require shared linear memory.

`npm run build:physics-wasm` finishes by running `tools/wasm/check-artifacts.mjs`. That script invokes the `fluid-artifact-check` Rust binary, which uses `wasmparser` to validate the actual module feature set and parse code operators. It requires SIMD instructions in SIMD and threaded builds, atomic instructions in the threaded build, and rejects SIMD instructions in scalar along with unsupported relaxed SIMD. It then checks the Wasm header, exported glue symbols, and shared-memory declaration. This validation is what caught a previously measured artifact labelled scalar that actually contained SIMD instructions; artifact names and generated JavaScript are not accepted as proof of the instruction set.

`tools/wasm/README.md` documents individual artifact builds, the whole-frame lane benchmark, and the generated-worker import adjustment.

For the local production server, use `npm start` (optionally `-- --port 3001`).
The wrapper applies isolation headers before Vinext's hashed-asset fast path;
calling `vinext start` directly bypasses that wrapper. After building and
starting the server, verify the actual HTTP responses without a browser:

```sh
npm run check:physics-wasm:serving -- http://localhost:3001
```

This checks HTML, render/simulation worker bundles, Wasm glue/binary, and the
nested Rayon helper, including revalidation of the stable Wasm artifact URLs. Cloudflare uses `public/_headers` and the application
worker policy for the same headers.

Advance Lab defaults to SIMD (or scalar without SIMD support). Fluid Lab CPU
uses up to eight workers where isolated shared memory is available. The recorded
M1 Max whole-frame measurements and remaining limits are in
[`docs/research/advance-lab-rust-wasm-implementation.md`](../docs/research/advance-lab-rust-wasm-implementation.md).
