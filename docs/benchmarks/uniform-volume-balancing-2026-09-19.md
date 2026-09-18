# Uniform liquid balancing dispatch comparison

Dawn/Metal, 30 frames per fresh solver, 64-round cap, 0.1% capacity-error
tolerance. Discard the first three frames; run fast-exit / indirect / indirect /
fast-exit for each resolution. Both arms execute the same numerical kernels.
GPU timestamps bracket the balancing stage, including gather and phi-capacity
publication. Queue-fenced full-frame wall times and every sample are in the
adjacent JSON. Compilation is excluded. These are local measurements, not
portable timing ceilings.

| Scene | Fast-exit stage median, ms (two runs) | Indirect stage median, ms (two runs) |
|---|---:|---:|
| mini32 dam | 16.12, 15.93 | 20.25, 20.58 |
| mini64 dam | 101.38, 106.96 | 117.77, 116.72 |
| mini32 resting pool | 2.75, 10.81 | 8.39, 10.29 |
| mini64 resting pool | 4.78, 4.00 | 13.50, 9.04 |

The mini32 resting result is noisy and overlaps. The other comparisons favor
fast exit. The production default therefore remains direct dispatch with a
uniform early return. The private indirect switch exists only for this probe.

All measured moving-dam frames hit the 64-round cap at 0.1%; all resting frames
performed zero corrective rounds. Adaptivity does not guarantee savings when
the chosen tolerance is never reached. The metric is the maximum positive
receiver overfill divided by its open capacity, measured before correction.
A row scan measures initial error; only violations are scaled, and the global
maximum gates donor renormalization and all subsequent work. The cap is a work
limit, not a guarantee of convergence. Remaining scratch clears and dispatch
commands still execute after convergence.

Reproduce sequentially under the GPU lease:

```sh
node --import tsx tools/benchmark-uniform-volume-balancing-dawn.ts
node --import tsx tools/benchmark-uniform-volume-balancing-dawn.ts --rest
```

Focused Dawn suite: 10 checks pass, including tolerance gating, zero-round
identity, the hard round cap, conservation, and 90-frame mini32 wall impact.
