# M0(a): the M1 Max GPU shape as Dawn sees it

Lane A, 2026-11-23. Plan: [docs/plans/gpu-shape-2026-11-23.md](../gpu-shape-2026-11-23.md) §M0(a).
Full generated tables and curves: [m0a-tables.md](m0a-tables.md). Raw data: `tools/experiments/gpu-shape/results/*.jsonl`.

**Short answer.** The hypothesis has the right shape but the wrong size for random gathers.

- A latency-bound dispatch is flat in width up to a slab, then linear.
- For **random gathers** the slab is set by a fixed number of misses in flight, about 4k lane-loads machine-wide. Occupancy does not set it: at D=16, M=1 the knee is about 8k lane-chains, which is 1/12 of the 98k residency figure.
- For **coalesced** (line-per-simdgroup) gathers the slab is about 110k lanes at M=1. It scales with 1/M and shrinks with register and threadgroup-memory pressure.
- Dawn encodes every compute pass with `MTLDispatchTypeSerial`. Each dispatch boundary costs about 2.3–2.8 µs of drain, and independent passes never overlap.
- Packing same-class bodies into one dispatch, or (natively) using a concurrent encoder, recovers that drain: up to 7.7× for small latency-bound bodies, and 1.3–1.8× once a body is past its knee.

## 1. Setup

- **Machine and runtime.** Apple M1 Max (32 GPU cores), macOS 26.6.2. Dawn through `node_modules/webgpu` 0.6.0 on the Metal backend. Device toggle `disable-dawn-features=timestamp_quantization`.
- **Run discipline.** Every run went through `tools/gpu/fluid-gpu-run.sh timing`, so it was exclusive. All runs returned rc 0 with 0 gpuError events, except run 2's 13 G=65536 records (see Caveats).
- **Kernel** (`tools/experiments/gpu-shape/kernels.mts`).
  - Each lane runs M independent dependent-load chains of depth D over `nxt: array<u32>`, then writes one u32.
  - *Random* (`base`): `nxt` is a Sattolo single-cycle permutation and each lane starts at a hashed index, so every link is an independent random 4-byte load.
  - *Coalesced* (`coal`): a block permutation `nxt[32b+j] = 32·next(b)+j`, so each simdgroup walks one 128-byte line per link.
  - Correctness is checked against a CPU walk at the start of every run: 0 of 256 lanes wrong.
- **Footprints.** 8 MB, which is below the 48 MB SLC, and 256 MB, which is DRAM- and TLB-dominated. The ladder covers 16 KB–256 MB.
- **Classes.**
  - Register pressure, with caps read from `regcap.sh` (psoinfo `maxTotalThreadsPerThreadgroup` with no attribute).
  - `var<workgroup>` of 1, 8, 16 and 24 KB, touched once after the chain with a barrier.
- **Timing** (`sweep.mts`).
  - Timestamp queries at pass begin and end, one pass per dispatch, minimum over 4 repetitions after one warm-up.
  - Each command buffer runs a heater (about 1.5 ms of ALU), then the timed passes, then a clock probe.
  - The heater and probe write the same output buffer as the timed passes. Metal hazard tracking therefore orders them; otherwise they overlap and the timestamps are garbage.
  - The probe is one 32-lane workgroup running a 2048-step dependent FMA chain: about 75.5 µs at the top clock.
  - Measurements are retaken after a ramp if probe > 1.5 × best.
  - Command buffers are capped at about 30 ms. G is capped at 32768.
- **Native harness** (`native/chase.swift`, build with `swiftc -O`). The same chase in MSL, timed by command-buffer `gpuEnd − gpuStart`. It compares four encodings:
  - a serial encoder;
  - a concurrent encoder;
  - a concurrent encoder with `memoryBarrier(scope: .buffers)` between dispatches;
  - one packed dispatch.

Runs:

| file | phases | notes |
|---|---|---|
| m0a-run1-partial | dvfs, ladder, foot, floor, base (partial) | the clock wandered (probe 100–125 µs); only the ladder, floor and DVFS are used |
| m0a-run2 | base, reg (multiply-add), wgmem, coal, coalreg, coalwg, pack, boundary | top clock throughout (probe 75.4–75.6) |
| m0a-run3 | regx, coalregx (xor-add per link) | clean |
| m0a-run4 | regp, coalregp (registers pinned by R loads) | clean, but the R prologue loads confound it |
| m0a-run5 | regq, coalregq (registers from a serial recurrence, no per-link work) | clean; **this is the register-residency probe** |
| m0a-native | serial / concurrent / concurrent+barrier / packed | 256 MB L256 G40 rows ran at a reduced clock (probe ≈ 290–380) |

## 2. Clock (DVFS): the first thing to control

- The same dispatch, L256 G40 D16 on 8 MB, took 61.8 µs cold (probe 281 µs) and 22.9 µs after a ramp (probe 80.5 µs).
- The probe range is 75.5 → about 247–281 µs, a **3.3–3.7× clock range**.
- A stream of small dispatches does not hold the top clock on its own: run 1's probes sat at 100–125 µs.
- In the native run, the 256 MB L256 G40 sequences (2–24 ms of DRAM-bound gathers) read probes of 290–380 µs right after. *Inference:* sustained memory-bound load with little ALU lets the GPU drop its clock.
- **Implication:** any fluid timing whose clock state is unknown can be off by up to 3×. The M0(b) census should record this probe inside a real step.

## 3. Latency and throughput by footprint

One 32-lane simdgroup chasing random addresses (D 256/1024), and saturated wide throughput (run 1; `*` = reduced clock):

| footprint | ns per dependent load | wide random G lane-loads/s |
|---|---|---|
| 16 KB | 121 | 33–38 |
| 128 KB | 141 | 8.5–20 |
| 1 MB | 293 | 17–19 |
| 4 MB | 273 | 14–15 |
| 8 MB | 276 | 13.3–14.2 (run 2: 14.5) |
| 32 MB | 270 | 6.1–6.9 |
| 64 MB | 385 | 1.27 |
| 128 MB | 511 | 0.80* |
| 256 MB | 456 | 0.67–0.76 (run 2: 0.68) |

- The latency step at about 1 MB is the L2. Footprints of 1–32 MB sit at about 270–290 ns (SLC).
- Random throughput falls 20× between 32 MB and 256 MB. Latency only rises 1.7× over the same range.
- By Little's law, 256 MB holds only 0.68 G/s × 456 ns ≈ **310 random loads in flight machine-wide**. *Inference:* the limit is address translation (page walks / TLB reach), not DRAM bandwidth: 0.68 G/s × 4 B is 2.7 GB/s of payload, or 87 GB/s if every load moves a full 128 B line.

## 4. Single-dispatch floor

One workgroup of 32 lanes, alone in its own timestamped pass (run 1, µs):

| footprint | D=0 | D=1 | D=4 | D=16 | D=64 |
|---|---|---|---|---|---|
| 8 MB | 5.29 | 5.54 | 5.96 | 9.04 | 20.9 |
| 256 MB | 5.71 | 6.17 | 6.21 | 13.0 | 32.5 |

- A lone pass costs about **5.3 µs** with no work. The native command-buffer floor is 5.50 µs, so this is Metal's, not Dawn's.
- Each dependent link adds about 0.22–0.25 µs at 8 MB and about 0.42 µs at 256 MB.
- Inside one pass, back-to-back dispatches cost **2.29 µs each** at D=0 (64 in a row). The floor rises with L because one workgroup runs on one core:

| L (8 MB, D16, M1) | 32–128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| floor, µs | 8.75–8.88 | 10.5 | 16.1 | 27.3 |

## 5. Time vs lanes per class

All 8 MB at D=16 unless noted. "2×" is the total lanes at which time reaches twice the floor. "Sat." is G lane-loads/s at the widest point. Full tables, including D=1 and D=4, every L and 256 MB, are in m0a-tables.md.

### 5.1 Random gathers (base): small slab, set by misses in flight

The curve for base, 8 MB, D16, M1 (µs per dispatch; total lanes = L × G):

| lanes | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k | 256k | 1M |
|---|---|---|---|---|---|---|---|---|---|---|
| L=64 | 9.50 | 9.58 | 10.7 | 16.5 | 28.6 | 50.6 | 116 | 154 | 296 | 1160 |
| L=256 | 13.6 | 16.1 | 16.1 | 16.3 | 27.7 | 50.4 | 82.2 | 150 | 296 | 1160 |

| D | M | floor µs (L ≤ 128) | lanes at 2× | lanes × M at 2× | sat. G/s |
|---|---|---|---|---|---|
| 1 | 1 | 5.0–5.8 | 37–58k | 37–58k | 13–14.5 |
| 4 | 1 | 5.8–5.9 | 17–18k | 17–18k | 14.5 |
| 16 | 1 | 8.8–8.9 | 8.5–8.7k | 8.5–8.7k | 14.5–14.8 |
| 16 | 2 | — | ~4.5k | ~9k | 14.5 |
| 16 | 4 | — | 2.8k (L64) | ~11k | 14.5 |

- Saturation is **14.5 G lane-loads/s whatever L, M, register cap or wgmem** (sections 5.3 and 5.4).
- The knee in lanes × M is about constant at 8–11k. Little's law gives 14.5 G/s × 276 ns ≈ **4.0k random loads in flight machine-wide, ≈125 per core**. That is far below what residency allows: 1024-thread caps, and the coalesced class reaches more than 100k lanes.
- **Wave quantization.** L=256 shows steps at multiples of 32 workgroups: 16.3 µs flat from 16 to 32 groups, 21.4 at 33, 27.5 at 40–64, 32.4 at 65. Workgroups spread one per core first, and each core has its own random-gather rate of about 0.45 G/s.
- At 256 MB: saturation is 0.68 G/s, the D16 floor is 12.8–13.3 µs, and time doubles from 64 to 256 lanes. The slab is a few hundred lanes. One workgroup of 1024 lanes already takes 98 µs.

### 5.2 Coalesced gathers (coal): large slab, set by residency × loads in flight

| D | M | lanes at 2× | lanes × M at 2× | sat. G lane-loads/s |
|---|---|---|---|---|
| 4 | 1 | 161–169k | 161–169k | 194–197 |
| 4 | 4 | 58–73k | 230–290k | 217–227 |
| 16 | 1 | 107–110k | 107–110k | 212–228 |
| 16 | 4 | 30–32k | 120–128k | 221 |
| 16 (256 MB) | 1 | 14–15k | 14–15k | 23.7 |

- D16 M1 at 8 MB is flat at 8.4–9.7 µs from 1k to 32k lanes. It reaches 12 µs at 64k, 18.7 at 128k, and is linear beyond (324 µs at 4M lanes).
- Saturation is about 7 G lines/s.
- The knee in lanes × M is about 110–128k at D16. It is the same for L = 64, 256 and 1024.
- *Inference:* the knee is residency (resident lanes × M loads in flight) up to the memory-request ceiling. It is about the 98k "32 cores × resident threads" figure in the hypothesis.

### 5.3 Register pressure

`regcap.sh` caps (max threads per threadgroup):

| R extra live u32 | 0 | 48 | 64 | 80 | 96 | 112–128 | 160 | ≥192 |
|---|---|---|---|---|---|---|---|---|
| recurrence (regq, run 5) | 1024 | 1024 | 1024 | 1024 | 832 | 640 | 576 | 448 |
| xor-add / pinned (runs 3–4), M1 | 1024 | 896 | 704 | 576 | 512 | 384 | 384 | 384 |

Run 5 (regq) is the clean residency probe. Its registers have no per-link work and no extra loads.

| R | cap | random M1 sat. / 2× | random M4 sat. | coalesced M1 sat. / 2× | coalesced M4 sat. / 2× |
|---|---|---|---|---|---|
| 0 | 1024 | 14.5 / 8.5k | 14.5 | 212–228 / 107–110k | 221 / 30–32k |
| 96 | 832 | 14.5 / 12–14k | 14.6 | 77–81 / 59–68k | 221 / 36–37k |
| 128 | 640 | 14.5 / 12k | 14.6 | 62–63 / 48–55k | 212–214 / 34–35k |
| 160 | 576 | 14.5 / 13–15k | 14.6 | 43–51 / 37–44k | 157–190 / 35k |
| 192 | 448 | 14.5 / 15–17k | 14.6 | 39–42 / 42–44k | 142–154 / 35k |

- **Random gathers are immune to register pressure**: 14.5 G/s at every R. This fits a miss-capacity limit, not an occupancy limit.
- **Coalesced M1 loses 2.7–5.5× of throughput** as R goes 96 → 192. The slab shrinks from about 110k to 40–60k lanes.
- At M4, registers cost little until R ≥ 160 (−15 to −35%). Four chains per lane win back what residency loses.
- The R=96 drop (cap 832, but 2.7× slower) is larger than the threadgroup cap implies. *Inference:* resident simdgroups per core fall with register use before the per-threadgroup cap does.
- The multiply-add (run 2) and xor-add (run 3) variants slow even random gathers at high R. They are ALU-bound, not residency probes. They are kept in the tables for completeness.

### 5.4 Threadgroup memory

- Random gathers are unaffected: 14.5 G/s at 1–24 KB. The floor rises 1–8 µs.
- Coalesced, D16 M1, saturated G lane-loads/s:

| L \ wgmem | 1 KB | 8 KB | 16 KB | 24 KB |
|---|---|---|---|---|
| 64 | 228 | 42 | 15 | 6.0 |
| 256 | 213 | 195 | 98 | 46 |
| 1024 | 211 | 211 | 212 | 182 |

- Throughput tracks **lanes per KB of workgroup memory**. It rises monotonically: 4/KB → 15, 8 → 42, 10.7 → 46, 16 → 98, 43 → 182, 64 → 212 G/s.
- About 40–64 lanes per KB costs nothing. Below that, the coalesced slab collapses: L64 with 24 KB has its 2× point at 9k lanes and its knee at 6k.
- *Inference:* a fixed per-core threadgroup-memory pool on the order of 64–100 KB limits resident groups. This is not measured directly. It conflicts with the older memory note that "≥16 KB caps 512 lanes at one per core"; L=512 was not tested here.

## 6. The slab model

Fitted to the run 2 base and coal sweeps at D ≥ 4 (n = 768 random and 222 coalesced points per footprint):

    t(G, L, D, M) ≈ t0 + max( D·ℓ ,  ⌈G/32⌉·L·M·D / Xc ,  L·G·M·D / X )
                    floor      per-core rate     machine rate

| class | t0 µs | ℓ ns/link | Xc G/s per core | X G lane-loads/s | median abs err | p90 |
|---|---|---|---|---|---|---|
| random 8 MB | 6 | 220 | 0.45 | 14.5 | 7.6% | 25% |
| random 256 MB | 6 | 150 | 0.064 | 0.68 | 1.5% | 48% |
| coalesced 8 MB | 6 | 180 | 6.25 | 200 | 6.6% | 18% |
| coalesced 256 MB | 6 | 210 | 2.06 | 22 | 7.4% | 24% |

- **Slab.** The dispatch stays near its floor while L·G·M ≲ ℓ·X (Little's law). That is about 3–4k lane-chains for random 8 MB, about 100–310 for random 256 MB (fitted ℓ vs ladder latency), and about 36–50k for coalesced 8 MB. The measured 2× points sit about 2× beyond these values: 8.5k, about 250 and about 110k.
- **Per-core term.** Workgroups fill cores one at a time, so ⌈G/32⌉ makes G = 33 cost as much as G = 64 when the per-core term dominates.
- **Large L.** A large L raises the floor (one core does all of one group's work). This is where the fit is worst: L = 1024 at small G, max error 53–188%.
- **Class modifiers** (measured, sections 5.3 and 5.4; not in the fit):
  - Register pressure and wgmem below about 40 lanes/KB scale X and Xc down for coalesced access only.
  - Raising M restores them.
  - Random access is insensitive to both.
- **t0** (about 6 µs here) includes the lone-pass cost. Inside a multi-dispatch pass, the per-dispatch overhead is the 2.3–2.8 µs boundary instead (section 8).

## 7. Packing baseline (Dawn)

Same class: k bodies as k serial dispatches in one pass, against one dispatch of k × the groups.

| body | k=2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| L64 × 8 groups, D16, 8 MB | 1.67× | 3.06× | 5.24× | 6.44× | 7.69× (220 → 28.6 µs) |
| L64 × 8 groups, D4, 8 MB | | | | | up to 10.1× |
| L256 × 8 groups, D16, 8 MB | | | | | up to 4.5× |
| L256 × 40 groups (10k lanes, past the knee), D16, 8 MB | 1.34× | | | | 1.78× max |
| L64 × 8 groups, D16 / D4, 256 MB | | | | | 3.9× / 4.7× |

- k separate passes cost the same as k serial dispatches. So do k passes with distinct output buffers. **Independent passes do not overlap under Dawn.**
- Different bodies (pack2: two chase bodies split on workgroup id):
  - The packed time is about max(A, B), and serial is about A + B.
  - For example, A = 9.33 and B = 52.0 µs cost 59.4 serial and 52.4 packed.
  - Carrying the second body costs A's groups nothing: 9.58 vs 9.33 µs.
  - There is no gain when one body dominates: 407 serial vs 410 packed at L256 × 40.

## 8. Dispatch boundary and Dawn's dispatch type

**Boundary.** Per-dispatch cost inside one pass (µs):

| body (L, G, D) | lone dispatch, own pass | 64 dependent in 1 pass | 64 independent in 1 pass | 64 passes |
|---|---|---|---|---|
| 32, 1, 1 | 6.21 | 2.72 | 2.40 | 2.62–2.65 |
| 64, 8, 16 | 11.1 | 7.15 | 6.96 | 6.98–7.13 |
| 256, 40, 16 | 27.1 | 20.5 | 19.6 | 19.6–20.5 |
| 256, 160, 4 | 17.2 | 14.6 | 11.9 | 12.0–14.4 |

Dependent and independent dispatches cost the same, and a pass boundary adds nothing over a dispatch boundary. The boundary is a **drain plus launch of about 2.3–2.7 µs**. A lone pass costs about 3.5 µs more than that.

**Dawn uses serial dispatch: verified** by disassembly (`otool -arch arm64 -tV` of `dawn.node`):

- In `dawn::native::metal::CommandBuffer::EncodeComputePass`, the path taken when counters are sampled at stage boundaries (our timestamp path) builds an `MTLComputePassDescriptor` and calls `setDispatchType:` with `x2 = 0`, which is `MTLDispatchTypeSerial`.
- The other path calls plain `computeCommandEncoder`. That this is serial is from Apple's documentation, not verified here.
- The binary contains no `computeCommandEncoderWithDispatchType:`, `memoryBarrierWithScope:` or `memoryBarrierWithResources:` selectors. No toggle exposes a concurrent encoder.
- WebGPU offers no way to request one.

**Native Metal, same kernels** (µs per command buffer; N dispatches of one body):

| body (8 MB) | N | serial | concurrent | concurrent + barrier each | packed |
|---|---|---|---|---|---|
| L32 G1 D1 | 64 | 184 | 9.8 | 183 | 5.8 |
| L64 G8 D4 | 64 | 271 | 19.1 | 288 | 18.3 |
| L64 G8 D16 | 64 | 542 | 44.8 | 586 | 42.4 |
| L256 G40 D16 | 64 | 1295 | 736 | 1404 | 729 |
| L64 G8 D16, 256 MB | 64 | 3176 | 772 | 3244 | 772 |

- Native serial costs (184 − 5.4)/63 = **2.8 µs per boundary**, which matches Dawn's 2.3–2.7. So the boundary cost is Metal's serial drain, not Dawn's encoding.
- A concurrent encoder costs about 0.07 µs per extra tiny dispatch, and **matches one packed dispatch** within about 10% across every configuration. The exception is the reduced-clock 256 MB L256 rows, where it ranged from 0.85× to 1.14× of packed.
- A concurrent encoder with a barrier between dispatches is the same as serial. The barrier is the drain.

**Would concurrent dispatch help?** Only for **independent** dispatches. There it gives exactly the packing gain, without writing packed kernels. For dependent dispatches a barrier is required, and it costs the full serial drain.

Dawn cannot emit concurrent dispatches. The available lever is therefore packing independent same-pass work into one dispatch in WGSL, which section 7 shows reaches the same numbers. *Inference:* patching Dawn to use concurrent dispatch with tracked barriers would only pay where the fluid step has runs of mutually independent dispatches. M0(b)'s census should count them.

## 9. Caveats

- **Synthetic access patterns.** Real kernels mix ALU, coalesced and random loads, and their D and M are not explicit. The model gives a bound and a shape, not a per-kernel prediction.
- **Clock state is the largest unknown** (3.3× range). Every number here is at the top clock (probe 75.4–75.6 µs) unless marked. The real sim's clock state is not known.
- **Register variants.** Multiply-add (run 2) and xor-add (run 3) are ALU-bound. The pinned variant (run 4) adds R scattered prologue loads, which are what bound it. Only run 5 (recurrence) isolates residency.
- **"Latency" per footprint** is a single 32-lane simdgroup of random addresses, so it already contains 32-way memory-level parallelism inside the simdgroup. It is not a single-thread load-to-use latency.
- **Invalid records.** Run 2 issued 13 dispatches at G = 65536, above `maxComputeWorkgroupsPerDimension` = 65535. They produced 26 gpuError events and are excluded by `analyze.mts`. The sweep now caps G at 32768. Every other record comes from runs with 0 gpuError events.
- **Run 1** base rows ran at a reduced clock and are superseded by run 2. Only run 1's ladder, floor and DVFS numbers are used.
- **Native rows.** The 256 MB L256 G40 rows ran with probe 290–380 µs (reduced clock). Use their ratios, not their absolute values.
- **Threadgroup-memory budget.** The per-core pool size is inferred from throughput, not read from hardware. L = 512 was not tested.

## Reproduce

    tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/sweep.mts out.jsonl check,base,coal,regq,coalregq,wgmem,coalwg,pack,boundary
    node --import tsx tools/experiments/gpu-shape/analyze.mts out.jsonl [--series=base:8:16:1]
    X=q node --import tsx tools/experiments/gpu-shape/dump-wgsl.mts <dir> 96,128,160,192   # then regcap.sh <dir>/*.wgsl
    swiftc -O tools/experiments/gpu-shape/native/chase.swift -o <scratch>/chase && tools/gpu/fluid-gpu-run.sh timing <scratch>/chase native.jsonl
