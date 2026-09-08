# Native CM12 surface restored

The requested default remains coarse-first (`selectorMode: coarse-first`) with
paper timestep 1/30 s. The default density/surface path is `native-cm12`, including
the quarter pool-impact scene profile. Retained-density and current-map modes
remain explicit experimental alternatives; this is not a repository rollback.

The native mode does not construct the retained scene field, its initial
quadrature override, or current-map history. It uses the native CM12 density
initialization and transport/presentation path. The four native volume and
surface-proof reconstruction functions are copied from `57b6ae39` (7 September)
in `sparse-cm12-native-surface.wgsl.ts`. Native publication uses the original
support scale and cache apron, without today's added curvature demotion proof.
The adaptive fan mesh, wall alignment and Gaussian normal shaders are restored
byte-for-byte to that commit. Baseline fingerprint tests pin those sources.

Today's topology storage, feature composition, editor and camera code remains.
This is source equivalence of the restored surface algorithms, not a claim that
every solver field or the whole application is byte-identical to yesterday.
Current live-solid-edit admission requires retained support and therefore remains
available through the retained mode; no unsafe admission bypass was introduced.

Dawn validation: both authored coarse and all-fine quarter scenes completed
steps 0/6/15/30, with finite fields, fresh shipping GPU meshes and no WebGPU
validation errors. The all-fine run also asserts that the native path allocates
neither retained-field controls nor a current-map history. Receipts are under
`artifacts/native-surface-restored/quarter/{coarse,fine}`.

Ten focused CPU checks passed. The canonical Dawn suite was run at its
unchanged 180-second budget: five lanes passed, symmetric expansion failed its
D4 density assertion, five other lanes timed out, and six remained unrun.
An untouched `57b6ae39` checkout also fails the same D4 density assertion in
a standalone diagnostic run. Its canonical lane first timed out at the original
20-second limit; the standalone result does not count as a canonical pass.
The live Chrome quarter scene was verified with Native CM12 and Coarse first
selected, then stepped to 0.2000 s at 33.3 ms. It remains open for monitoring.
Logs: `/tmp/fluid-native-surface-canonical.log` and
`/tmp/fluid-yesterday-symmetry-diagnostic.log`. No thresholds were raised.
