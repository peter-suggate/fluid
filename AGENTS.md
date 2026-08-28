# Repository guidance

## Sparse CM12 Dawn regression gate

After a large change to Sparse CM12 simulation, sparse-world topology,
presentation publication, terrain boundaries, or live scene editing, run:

```bash
npm run test:dawn:sparse-cm12
```

This is the canonical short post-refactor confidence gate. It covers symmetry,
hydrostatic stability and adaptivity, mini32 correctness and performance,
mini64 performance, both far-wall dam fronts, live rigid/liquid insertion, and
floor-only symmetric collapse. Do not silently weaken a lane or raise a timing
ceiling to make a change pass. Use `-- --list` to inspect the matrix and
`-- --lane=<id>` while diagnosing one failure. See
`docs/SPARSE_CM12_DAWN_REGRESSION.md` for scope and baseline policy.

Do not run Dawn concurrently with the browser or another Dawn process; the
suite uses the repository-wide WebGPU lease and isolated processes.
