# Repository guidance

## Clean-repo gate

Uniform Geometric (method id `uniform-volume`) is the maintained method; Sparse
CM12, adaptive and Losasso coverage is retired. The repository is clean when all
three pass:

```bash
npm run check:types
npm run test:unit
npm run test:dawn
```

`test:dawn` runs every Dawn-gated test file (one that reads
`WEBGPU_NODE_MODULE`) serially, one isolated process per file. Pass path
filters (`npm run test:dawn -- uniform`) while diagnosing, and `-- --list` to
see the set. Do not silently weaken a Uniform lane or raise a timing ceiling to
make a change pass.

Do not run Dawn concurrently with the browser or another Dawn process; the
tests take the repository-wide WebGPU lease and the runner waits for it.
