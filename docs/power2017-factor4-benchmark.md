# Power 2017 factor-4 benchmark

The restored Power backend is a comparison method for the fully adaptive
LoSasso path. It follows the two-grid construction in Aanjaneya et al. 2017,
Section 5: the coarse octree/Power solve is coupled to a separate fine SPGrid
level set. The benchmark fixes that surface grid at factor 4 and rebuilds its
topology every accepted advance.

## Selecting and resuming

In the Octree method panel, choose **Power 2017 · factor-4 benchmark** under
Coarse dynamics. The surface-tracking control becomes read-only at factor 4 so
a saved or shared state cannot accidentally resume a different Power variant.
The URL only needs to preserve:

```text
param.octree.coarseBackend=power2017
```

Method-value normalization restores factor 4 and cadence 1 when that state is
loaded. Switching back to LoSasso leaves the LoSasso default at factor 1.

## Acceptance gate

Run the dedicated symmetric-expansion gate:

```sh
npm run test:webgpu:symmetric-expansion:power2017
```

The lane advances three factor-4 generations and requires:

- the authoritative Power backend and persistent Power MGPCG pressure solve;
- three accepted advances with no rejected topology publication;
- a valid recurring fine-level-set generation at every checkpoint;
- exact D4 topology symmetry and bounded D4 field symmetry;
- clean structured and WebGPU validation.

The existing LoSasso factor-4 comparison remains:

```sh
npm run test:webgpu:symmetric-expansion:fine
```

The historical reference used during restoration is commit `c8c84ee`
(`Make band 4 canonical for symmetric expansion`), immediately before commit
`f7850f9` introduced the LoSasso octree cutover.
