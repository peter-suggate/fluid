# Offline wet-cloth reference

The application continues to use its production narrowband FLIP solver. This directory contains diagnostic instrumentation for the original released solver, used **only as an offline comparison oracle**.

Upstream: [libwetcloth, commit b43993d775a0f9e3a9e32611f26936f2528fac37](https://github.com/raymondyfei/libwetcloth/tree/b43993d775a0f9e3a9e32611f26936f2528fac37), Mozilla Public License 2.0. `wetcloth-diagnostics.patch` adds synchronous state export, bounded headless execution, and an optional TBB concurrency limit. It does not modify material forces, contact, transfers, scripts, timesteps, or solver tolerances.

The diagnostic loop calls the original `ParticleSimulation::stepSystem(dt)` once per step. It bypasses the interactive continuation prompt and renderer export callback. Frame zero is exported before any step. CGS arrays retain original particle order, including four position/velocity entries per particle; the fourth is the rod angle/angular velocity. `softParticles` identifies the authored structural prefix. The exporter preserves full precision; output is not render-mesh geometry.

Build with the pinned source and bundled headers. For an Apple Silicon headless build, CMake 4.4.4, AppleClang 21, and Homebrew TBB 2023.2.0 were used:

```sh
git apply /absolute/path/to/tools/reference/wetcloth-diagnostics.patch
cmake -S . -B build -DUSE_OPENGL=OFF -DUSE_MKL=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/homebrew -DCMAKE_CXX_FLAGS=-I/opt/homebrew/include
cmake --build build -j 4
```

The explicit include path is necessary because upstream's application CMake file adds TBB include directories inside the OpenGL option, while the headless application also includes TBB headers. No source compatibility change was needed. The additional include path does not alter physics.

```sh
WETCLOTH_REFERENCE_THREADS=4 WETCLOTH_REFERENCE_STEPS=500 \
WETCLOTH_REFERENCE_EVERY=100 WETCLOTH_REFERENCE_OUTPUT=/absolute/path/to/state.jsonl \
  build/libWetCloth/App/WetClothApp -s /absolute/path/to/simple_cloth.xml -d 0 -p 0
```

Record the pinned commit, input/patch/executable/output SHA-256 hashes, compiler/dependencies, thread count, original timestep, requested/completed steps, and sampling interval in a manifest. The comparison tool rejects mismatched input hashes, edited diagnostics, incomplete runs, and superseded production probes:

```sh
node --import tsx tools/compare-wetcloth-reference.ts \
  --probe=/absolute/path/to/production.json \
  --reference=/absolute/path/to/state.jsonl \
  --manifest=/absolute/path/to/manifest.json --out=/absolute/path/to/comparison.json
```

Only the saved scene's declared source-to-world translation and CGS-to-SI conversion are applied. There is no fitted alignment or time warp. Error metrics do not automatically certify paper reproduction. Native APIC/AMLS-MPM and our NB-FLIP/Lagrangian discretizations differ; discrepancies must be investigated and compared under refinement.
