# Sparse CM12 production cutover artifact gate

`tools/report-sparse-cm12-production-cutover.ts` is a read-only, fail-closed
production gate. It never launches WebGPU and never edits or selects a runtime
path. Missing fields, shortened runs, changed thresholds, or legacy receipt
shapes fail instead of being inferred as success.

## Invocation

```sh
node --import tsx tools/report-sparse-cm12-production-cutover.ts \
  --manifest artifacts/sparse-cm12-production-cutover-manifest.json \
  --output artifacts/sparse-cm12-production-cutover-report.json
```

The process exits nonzero unless both optimization equivalence and the
unchanged absolute physics gates pass. It still writes/prints the full report
on a gate failure. Static contract validation is:

```sh
node --import tsx tools/report-sparse-cm12-production-cutover.ts --self-check
```

## Manifest

```json
{
  "kind": "sparse-cm12-production-cutover-manifest",
  "version": 1,
  "defaultConfiguration": {
    "method": "sparse-cm12",
    "brickFineResolution": 16,
    "presentationPageResolution": 16,
    "isDefault": true
  },
  "artifacts": {
    "authority": "artifacts/sparse-cm12-production-authority-60.json",
    "damPaired": "artifacts/sparse-cm12-head-paired-dam-60.json",
    "symmetryPaired": "artifacts/sparse-cm12-head-paired-symmetry-60.json",
    "ocean": "artifacts/sparse-cm12-ocean-b16-p16-stage-cost-production.json"
  }
}
```

Paths are resolved from the invoking working directory. The report includes
absolute paths and SHA-256 hashes of the manifest and every source artifact.

## Authority receipt

The authority artifact has kind `sparse-cm12-production-authority-receipt`,
version 1, B16/P16 configuration, and exactly 60 ordered checkpoints. Every
checkpoint contains:

- `fca`: `authorityOwner:"gpu"`, `hostScheduling:false`,
  `hostSchedulingDecisionCount:0`, `externalUploadsOnly:true`, accepted,
  candidate, and sealed generations equal and positive, and zero
  `faultCount`/`omissionCount`;
- `authorities.PCM1|PCF1|VEX1|SAW1|FPA1|A4D2`: `accepted:true`, equal
  positive accepted/producer/consumer generations, `coverageComplete:true`,
  and zero `faultCount`/`omissionCount`;
- `fpl`: accepted equal producer/consumer generation, `stageCount:6`, and zero
  global/local faults and omissions;
- `fpp`: accepted generation equals generation receipt, scheduled equals
  executed equals published, and omitted/coverage-fault/fault counts are zero;
- `dirtyStages` with exactly the keys `facePreparation`, `massTransport`,
  `gammaTransport`, `surfaceConditioning`, `pressureCoefficients`, and
  `presentation`.

Each dirty-stage receipt supplies positive accepted/producer/consumer
generation equality, `coverageComplete:true`, eligible/direct/closure/
executed/skipped counts, executed plus skipped equal to eligible, direct plus
closure no greater than eligible, and zero unknown/uncovered/fault counts.
Roots do not count as execution; the logical stage owns the execution receipt.

## Paired equivalence receipts

Dam and weakened-symmetry artifacts have kind
`sparse-cm12-head-paired-physical-equivalence`, version 1, the requested lane,
B16/P16 configuration, `steps:60`, and exactly 60 ordered checkpoints. Every
checkpoint contains equal 64-character HEAD/candidate SHA-256 fields for:

- density;
- velocity;
- pressure;
- divergence;
- gamma;
- transported internal state.

Topology/workset hashes may be reported separately but are not physical gates.
Pressure-local versus full-refresh equality alone is not a HEAD-paired solver
receipt and is intentionally insufficient.

Each paired artifact also carries unchanged absolute-physics verdicts:

```json
{
  "absolutePhysics": {
    "head": {
      "passed": false,
      "failures": ["existing envelope failure"],
      "thresholds": { "...": "the canonical lane thresholds" }
    },
    "candidate": {
      "passed": false,
      "failures": ["existing envelope failure"],
      "thresholds": { "...": "the identical canonical lane thresholds" }
    }
  }
}
```

If the physical hashes match and the failure lists match, optimization
equivalence remains true and those failures are labelled as pre-existing HEAD
envelope failures. They still make `absolutePhysicsPassed` and the overall
production verdict false. The tool contains the existing canonical dam and
weakened-symmetry threshold objects and requires both arms to reproduce them
exactly; a manifest cannot replace or widen them.

## Ocean receipt

The existing `sparse-cm12-stage-cost` schema is consumed directly. It must be
the `ocean-seiche` scene, B16/P16, hardware GPU timestamps, exactly 24
nonpressure samples, no validation errors, and an internally consistent p95.
The embedded gate must explicitly be eligible and passed with immutable
`threshold_ms:10` and `minimumSamples:24`; the measured p95 must be strictly
less than 10 ms.

## Verdicts

The generated report exposes three distinct booleans:

- `optimizationEquivalent`: every configuration, authority, observability,
  paired physical equivalence, and performance gate passed;
- `absolutePhysicsPassed`: the unchanged candidate absolute gates passed;
- `passed`: both of the above.

`preExistingHeadEnvelopeFailures` and `candidateEnvelopeFailures` remain
separate. This prevents an existing HEAD failure from being mislabeled as an
optimization regression while also preventing it from being waived for
production.
