# Ocean coarse-first presentation performance

The reported 5-second publication was reproduced using the authored
`ocean-seiche` scene, balanced adaptive-mass, coarse-first and the paper
1/30-second step. No minimum-cell override or reduced proof policy was used.
The browser viewport was unloaded and Dawn used the repository-wide GPU lease.

At step 9 (0.3 seconds), the first baseline measured 4,508.09 ms in FPP1 packet
publication and 214.24 ms in the surface proof. Page planning and verification
were negligible. This was GPU execution time, not an incorrect UI unit or a
host readback attributed to publication.

Macro pages were categorically excluded from the workgroup density stencil by
`sampleScale == 1`. Each of their 512 output samples could reconstruct forty
native density samples independently. Restriction then repeatedly resolved
spatial ownership and traversed finest-cell rows, even where one compiled
native donor represented those rows. Coarse-first exposed the problem as macro
leaves changed rung and acquired surface-support work.

Allowing fitting macro stencils into the existing cache reduced step-9 packet
publication to 72.68 ms. That was insufficient: proofs still cost 220.46 ms.
The final direction therefore uses the accepted transport execution image
(TEI) rather than rebuilding template addresses during presentation. TEI leaf
records already contain generation-owned first-cell, count, valid strides,
and native scale. Nearby records are staged once per page/proof workgroup;
restriction integrates native donor overlap volumes. Queries outside the
staged window still consume the accepted compiled leaf descriptor. Ordinary
conservative topology transfer retains its original restriction function.

The probe is `tools/probe-ocean-coarse-presentation-dawn.ts`. Run it with
`WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js` and
`FLUID_GPU_ISOLATE_PASS_LABELS=1`. `OCEAN_OUT` names its JSON receipt;
`OCEAN_CAPTURE_PREFIX` additionally captures final metadata and payload buffers.
Pass pairs isolate packet publication, verification and surface proof. Their
sum is an attribution diagnostic, not a replacement for the production advance
partition; encoder breaks can perturb overlap and timings.

## Compiled-path measurements and equivalence

At step 9 the compiled-path isolated reading is 4.390912 ms for page
publication, 20.447232 ms for surface proof, and 26.54208 ms across the whole
publication interval. Against the first baseline's 4724.16256 ms interval,
that is about 178× faster. Individual pass pairs include diagnostic encoder
breaks; this is not a claim that the complete simulation advances in 26 ms.
The ten-step compiled run spans 10.55–51.58 ms in publication.

The page/proof binding deliberately keeps renderer metadata at binding 14 and
binds the existing TEI buffer at the otherwise unused publication binding 3.
Transport's binding 14 cannot simply be reused: it refers to a different arena
in the presentation bind group. The test exercises that separate compiled
reader as well as mixed native widths and missing tiles.

All ten sampled topology censuses match the original. Final physical density
is byte-identical. Terminal metadata is byte-identical. Of 630,784 active
presentation samples (1,232 pages), nine differ by exactly one binary16 code;
no flags or signs change. The maximum scalar difference is 0.000244140625 m
in a submerged sample. Native-volume summation changes rounding order, so the
presentation payload is not claimed byte-identical. Stale unallocated payload
slots are excluded from the comparison.

Receipts: `artifacts/ocean-coarse-presentation/compiled.json`, `before.json`,
`cache-only.json`, and `output-comparison.json`.

The new compiled-restriction Dawn test passes 1,200 queries against exhaustive
CPU volume sums across native widths 1/2/4/8 and query widths 1/2/4/8/16,
including missing tiles and signed coordinates. All 11 existing CPU surface
proof, page-parallelism and signed-consumer tests pass.

The first canonical suite attempt passed its first five lanes, then another
GPU task acquired the lease between lanes. The remaining lanes refused to run;
that attempt is not an acceptance result. A second full attempt was likewise
interrupted by a new gravity-wave probe batch; a third attempt refused startup
because the lease was already held. A clean full rerun remains required.
No numerical assertion or performance ceiling was relaxed. The other active
task is `Fix max1 axis artifacts`; exclusive GPU coordination is pending.
The user viewport remains temporarily unloaded to respect GPU exclusivity.

The workspace-wide TypeScript check reports 50 existing errors outside the
changed files; neither new file nor the modified resident TypeScript adds a
reported error. Lint of the new files passes. Lint of the resident TypeScript
reports nine existing errors on unchanged lines (React hook naming and a
`module` variable), plus existing unused-variable warnings.
