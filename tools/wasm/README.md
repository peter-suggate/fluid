# Fluid Wasm host tools

`npm run build:physics-wasm` builds three browser artifacts sequentially into
`public/wasm/fluid-wasm`: scalar, SIMD, and SIMD plus threads. The scalar and
SIMD builds use Rust 1.96.1. The threaded build uses the pinned
`nightly-2025-11-15` toolchain with `rust-src`, rebuilds `std` with atomics, and
uses wasm-bindgen/wasm-bindgen-cli 0.2.122 with wasm-bindgen-rayon 1.3.0. The
nightly is selected per command; the repository's default toolchain remains
stable. Install the two toolchains and matching wasm-bindgen CLI before building:

```bash
rustup toolchain install 1.96.1 --profile minimal --component rust-src --target wasm32-unknown-unknown
rustup toolchain install nightly-2025-11-15 --profile minimal --component rust-src --target wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.122 --locked
```

The build validates the Wasm header and ABI, rejects shared memory in fallback
artifacts, and requires SIMD plus shared memory and atomic instructions in the
threaded artifact. It also rewrites wasm-bindgen-rayon's package-style worker
import to the concrete generated module URL because these files are served as
public assets rather than passed through a bundler.

Useful non-browser checks are:

```bash
npm run check:physics-wasm
npm run test:physics-wasm:node
npm run test:physics-wasm:threaded
npm run test:physics-wasm:world
npm run test:physics-wasm:world3d
node --import tsx --test lib/physics-wasm/*.test.ts
```

Whole-frame lane benchmarks default to the 2D world. Pass `--dimension=3` to
exercise the 3D world with the same fixed method values, and `--dt=<seconds>`
to replace the default 1/30 second step:

```bash
node --import tsx tools/wasm/world-benchmark.ts --dimension=3 --dt=0.03333333333333333
```

The benchmark compares every published plane and the complete numerical
receipt across scalar, SIMD, and threaded lanes. Allocation-byte metadata is
excluded from receipt equality because Wasm memory growth differs by runtime;
artifact source fingerprints must still match exactly.

The world checks load the generated scalar artifact and exercise the complete
binary publication boundary. The 2D check covers load, advance, commands,
reset, and the Advance Lab read-only view. The 3D check covers dense render
planes, runtime values, tracers, liquid injection, and authored scene and rigid
body edits. It fails explicitly when the installed artifact predates 3D world
support.

The transitional TypeScript numerical parity generators were removed after
their fixtures and regression cases were frozen under `rust/core/testdata` and
the equivalent checks moved into Rust. Production and host tests do not load a
TypeScript physics oracle.

The threaded check adapts Node `worker_threads` to the Web Worker calls emitted
by wasm-bindgen-rayon, initializes four workers, and compares a 1,024-cell,
16-reduction-group pressure solve bit for bit across scalar, SIMD, and threaded
Wasm. It verifies the pool and deterministic kernel path; it is not a browser
integration or full-frame performance measurement.
