# Level-set-volume exact redistancing

## Enabled path

Selecting the existing `level-set-volume` transport now enables exact contour redistancing automatically. There is no separate algorithm switch. Sharpening and volume correction remain omitted, so this change isolates phase-distance maintenance from those later design choices.

The advance orders the work as follows:

1. Trace cell centres backward with RK2 and evaluate exact signed distance to the previous accepted RDF at the landing points.
2. Combine those redistanced phi values with transported conservative volume: phi gradients orient PLIC planes and volume fractions determine their offsets.
3. Rebuild and publish the accepted shared RDF without altering its contour during redistancing.
4. Evaluate exact signed distance to that immutable accepted contour at cell centres to seed phi for the next frame.

“Exact” here describes the distance query to the piecewise-linear shared contour. It is not an iterative PDE solve and has no residual tolerance. The query uses physical distance, so a segment and sample retain consistent units across coarse-fine cell boundaries.

## Receipt and UI

The optional `levelSetVolume` receipt identifies this path. Its redistancing fields are:

| Field | Meaning |
|---|---|
| `redistancedSamples` | Successful exact signed-distance queries at RK2 landings and accepted-surface cell centres |
| `redistanceFallbackSamples` | Queries where exact contour redistance was unavailable; they retain a finite bilinear RDF sample when possible, then use the existing phase fallback |
| `redistanceSegmentCount` | Accepted post-transport shared-interface segments indexed for the next state |
| `redistanceNanoseconds` | Native redistance diagnostic inside the LSV advance; zero in Wasm where stage timing is disabled |

Native stage timings are not additive. `redistanceNanoseconds` includes the source-distance work already enclosed by `phiGatherNanoseconds`. It excludes additional World topology-transition and refresh calls outside `levelset_volume::advance`.

The Advance Lab transport inspector labels redistancing as active whenever the LSV receipt is present and shows successful samples, fallbacks, and accepted contour segments. No sharpening control is presented.

## Verification

- Rust: 17 tests pass — six library tests (five redistancing and one existing), eight LSV integration tests including two new redistancing cases, and three Baseline regressions. The release `verify_world` build also passes.
- Wasm/UI: the scalar, SIMD, and frontend production builds pass. Four LSV contracts pass: scalar and SIMD live-resolution edits, Figure 7 for 30 frames, and half-pool for 10 frames.
- Serving: nine path checks pass, together with hash, MIME, no-cache, and Rust-source-fingerprint checks for all three Wasm binaries. The verified Rust source fingerprint is `411fccf90eaf115733ef422f3e944146049c50264c7af0a6c9756be669d12b20`; the SIMD binary hash is `3fa2a3fd0592d7390fc4c87bc7cd6c86323b9b72c2f7fe59c011a976a95d8332`.
- TypeScript: the full typecheck retains existing unrelated errors and reports none in the changed redistancing UI or Wasm contract paths (`/tmp/fluid-redistance-types.log`).

The Wasm contract covers the existing resolution-edit flow and sustained UI-default Figure 7 and half-pool flows. It requires every advanced LSV receipt to report successful redistance queries and a nonempty accepted contour, requires a valid nonnegative fallback count, and confirms Wasm stage timing remains zero. It retains the existing conservative-volume and visible-RDF publication assertions.
