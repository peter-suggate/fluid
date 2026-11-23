// GPU-shape microbench kernels (M0(a), docs/plans/gpu-shape-2026-11-23.md).
// A dependent-load chain: every lane follows M independent chains of D loads through a random single-cycle
// permutation (Sattolo), so each load's address is the previous load's value. Optional extra live registers (R
// values updated every link, forcing Metal's thread cap down) and an optional var<workgroup> array (bytes WGB).

export interface ChaseKernel {
  L: number; // lanes per workgroup
  M: number; // independent chains per lane
  R?: number; // extra live u32 registers carried through the chain
  WGB?: number; // var<workgroup> bytes (0 = none)
  X?: boolean; // the R registers update by xor-add (2 full-rate ops) instead of a 32-bit multiply-add (run 2's
  // multiply-add variant turned out ALU-bound: its saturated rate fell with R whether the chase was coalesced or not)
  P?: boolean; // pinned registers: the R values are loaded from outb before the chain and only combined after it; a
  // never-taken store to outb inside the loop keeps the compiler from sinking or re-loading them, so they stay live
  // with no per-link work (xor-add, run 3, was still ALU-bound at saturation)
  Q?: boolean; // like P, but the R values come from a cheap serial recurrence (r_k = (r_{k-1} ^ r_{k-1} >> 7) + c_k)
  // instead of R loads (run 4's P variant was bound by its R scattered loads); combined once after the chain.
  C?: boolean; // coalesced: the 32 lanes of a simdgroup walk one 128-byte line a link (needs a block permutation)
}

export const kernelKey = (k: ChaseKernel) => `L${k.L}_M${k.M}_R${k.R ?? 0}${k.X ? "x" : ""}${k.P ? "p" : ""}${k.Q ? "q" : ""}_W${k.WGB ?? 0}${k.C ? "_C" : ""}`;

// Parameters: mask = N-1 (N a power of two), depth = D, seed, base = lane offset (for packed bodies).
const PARAMS = `struct Prm { mask: u32, depth: u32, seed: u32, base: u32 }`;
const HASH = `fn hsh(x0: u32) -> u32 { var x = x0 * 747796405u + 2891336453u; x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u; return (x >> 22u) ^ x; }`;

// The chase body as a function over one binding name, so packed modules can hold several bodies.
function body(name: string, buf: string, prm: string, k: ChaseKernel) {
  const M = k.M, R = k.R ?? 0;
  const lines: string[] = [];
  lines.push(`fn ${name}(lane: u32) -> u32 {`);
  for (let m = 0; m < M; m++) lines.push(k.C
    ? `  var i${m} = ((hsh((lane >> 5u) * ${M}u + ${m}u + ${prm}.seed) & ${prm}.mask) & ~31u) | (lane & 31u);`
    : `  var i${m} = hsh(lane * ${M}u + ${m}u + ${prm}.seed) & ${prm}.mask;`);
  for (let r = 0; r < R; r++) lines.push(k.Q
    ? (r === 0 ? `  let r0 = hsh(lane + ${prm}.seed);` : `  let r${r} = (r${r - 1} ^ (r${r - 1} >> 7u)) + ${((r * 2654435761) % 4294967296 | 1) >>> 0}u;`)
    : k.P
    ? `  let r${r} = outb[(lane * 7u + ${r * 131071}u) & 0x3fffffu];`
    : `  var r${r} = hsh(lane + ${r * 7919 + 13}u);`);
  lines.push(`  for (var d = 0u; d < ${prm}.depth; d++) {`);
  for (let m = 0; m < M; m++) lines.push(`    i${m} = ${buf}[i${m}];`);
  // Distinct odd multipliers: no reassociation can merge the R accumulators into one.
  if (k.P && R) lines.push(`    if (i0 == 0xffffffffu) { outb[lane & 0x3fffffu] = d; }`);
  for (let r = 0; r < (k.P || k.Q ? 0 : R); r++) lines.push(k.X
    ? `    r${r} = (r${r} ^ i${r % M}) + ${((r * 2654435761) % 4294967296 | 1) >>> 0}u;`
    : `    r${r} = r${r} * ${(((2 * r + 3) * 2654435761) % 4294967296 | 1) >>> 0}u + i${r % M};`);
  lines.push(`  }`);
  if (k.P || k.Q) {
    // Horner in the chain's result: the loop-invariant r values cannot be pre-combined before the loop.
    lines.push(`  var acc = ${[...Array(M).keys()].map(m => `i${m}`).join(" ^ ")}; let h = i0 | 1u;`);
    for (let r = 0; r < R; r++) lines.push(`  acc = acc * h + r${r};`);
    lines.push(`  return acc;`);
  } else {
    const terms = [...Array(M).keys()].map(m => `i${m}`).concat([...Array(R).keys()].map(r => `r${r}`));
    lines.push(`  return ${terms.join(" ^ ")};`);
  }
  lines.push(`}`);
  return lines.join("\n");
}

export function chaseWGSL(k: ChaseKernel): string {
  const W = (k.WGB ?? 0) / 4;
  return `${PARAMS}
@group(0) @binding(0) var<storage, read> nxt: array<u32>;
@group(0) @binding(1) var<storage, read_write> outb: array<u32>;
@group(0) @binding(2) var<uniform> prm: Prm;
${W ? `var<workgroup> wgm: array<u32, ${W}>;` : ""}
${HASH}
${body("chase", "nxt", "prm", k)}
@compute @workgroup_size(${k.L}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  let lane = g.x + prm.base;
  var v = chase(lane);
${W ? `  wgm[li % ${W}u] = v;\n  workgroupBarrier();\n  v = v ^ wgm[(li + 1u) % ${W}u];` : ""}
  outb[g.x] = v;
}
`;
}

// Packed module: n bodies split on the workgroup index (gpb workgroups each). Body b reads its own buffer nxt<b%2>
// (two distinct permutations) with its own params, so the bodies are independent code paths and data.
export function packedWGSL(k: ChaseKernel, n: number, gpb: number): string {
  return `${PARAMS}
@group(0) @binding(0) var<storage, read> nxtA: array<u32>;
@group(0) @binding(1) var<storage, read> nxtB: array<u32>;
@group(0) @binding(2) var<storage, read_write> outb: array<u32>;
@group(0) @binding(3) var<uniform> prmA: Prm;
@group(0) @binding(4) var<uniform> prmB: Prm;
${HASH}
${body("chaseA", "nxtA", "prmA", k)}
${body("chaseB", "nxtB", "prmB", k)}
@compute @workgroup_size(${k.L}) fn main(@builtin(workgroup_id) w: vec3u, @builtin(local_invocation_index) li: u32, @builtin(global_invocation_id) g: vec3u) {
  let b = w.x / ${gpb}u;
  let lane = (w.x % ${gpb}u) * ${k.L}u + li + b * 65536u;
  var v = 0u;
  if ((b & 1u) == 0u) { v = chaseA(lane); } else { v = chaseB(lane); }
  outb[g.x] = v;
}
`;
}

// Dependent variant for the dispatch-boundary test: the start index comes from the previous dispatch's output.
export function chainedWGSL(L: number): string {
  return `${PARAMS}
@group(0) @binding(0) var<storage, read> nxt: array<u32>;
@group(0) @binding(1) var<storage, read> inb: array<u32>;
@group(0) @binding(2) var<storage, read_write> outb: array<u32>;
@group(0) @binding(3) var<uniform> prm: Prm;
@compute @workgroup_size(${L}) fn main(@builtin(global_invocation_id) g: vec3u) {
  var i = (inb[g.x] + g.x) & prm.mask;
  for (var d = 0u; d < prm.depth; d++) { i = nxt[i]; }
  outb[g.x] = i;
}
`;
}
