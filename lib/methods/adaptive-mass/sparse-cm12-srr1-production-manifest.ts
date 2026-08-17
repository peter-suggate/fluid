/** Mechanical resident cutover manifest for the accepted standalone SRR1 ABI. */

export interface SparseCM12SRR1PatchSite {
  readonly file: string;
  readonly anchor: string;
  readonly operation: "import" | "insert-after" | "replace" | "extend";
  readonly patch: string;
  readonly proof: string;
}

export const SPARSE_CM12_SRR1_PATCH_SITES: readonly SparseCM12SRR1PatchSite[]
  = Object.freeze([
    { file: "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
      anchor: "createSparseCM12SRR1IngressLayout({",
      operation: "insert-after",
      patch: "Create SIR1 ingress at the next aligned activity-tail word; make initialActivity length SIR1.totalWords and initialize its construction-static mode.",
      proof: "Only capacity/layout changes; all pre-existing activity offsets remain byte-identical." },
    { file: "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
      anchor: "const scalarResultAuthority = await WebGPUSparseCM12SRR1RuntimeAdapter.create(",
      operation: "replace",
      patch: "Construct WebGPUSparseCM12SRR1RuntimeAdapter asynchronously through GPUCompilationManager using temporal or immutable-full-oracle construction mode.",
      proof: "No synchronous pipeline construction and no evolving host authority." },
    { file: "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
      anchor: "this.scalarResultAuthority.encodePlan(encoder, this.activity);",
      operation: "replace",
      patch: "Encode SRR1 plan, then copy its canonical work list into the SIR1 activity tail. Keep the current full mass dispatch in slice 1.",
      proof: "Receipt construction is observational while the physical path remains identical." },
    { file: "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
      anchor: "this.scalarResultAuthority.encodePublish(encoder, this.activity);",
      operation: "replace",
      patch: "Dispatch compareSparseCM12MassResult over the SRR1 compare indirect family, then encode SRR1 receipt import/validation/promotion/commit.",
      proof: "Comparator runs only after gatherConservativeDensity authored HEAD destination bytes." },
    { file: "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
      anchor: "stage(\"gamma-diffusion\"",
      operation: "replace",
      patch: "Remove legacy SAW gamma execution only after mass cutover is exact; introduce an independent gamma SRR transaction in a later hash-gated slice.",
      proof: "Mass acceptance cannot silently change gamma/surface authority." },
    { file: "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
      anchor: "stage(\"surface-sharpening\"",
      operation: "replace",
      patch: "Remove legacy SAW surface execution only with the later surface SRR transaction; SRR1 mass commits at conservative-transport, not surface-sharpening.",
      proof: "Mass result publication is decoupled from downstream topology preparation." },
    { file: "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
      anchor: "this.scalarResultAuthority.destroy();",
      operation: "replace",
      patch: "Destroy SRR1 authority, ingress, and indirect buffers.",
      proof: "Allocation receipt remains exact and no buffer leaks." },
    { file: "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
      anchor: "fn scaInvalidateBrickTopologyClosure(brick:u32)",
      operation: "extend",
      patch: "Append generation-stamped SIR1 topology events for old/new owner spans, incidence receivers, activation/retirement page identities; do not add another radius.",
      proof: "Every changed identity is explicit producer evidence and duplicates dedupe by tile generation." },
    { file: "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
      anchor: "fn sirExactScalarDependency",
      operation: "replace",
      patch: "Publish SIR1 velocity/dependency change events only for tiles whose producer generation changed; delete full cell/bank classifier scans after paired cutover.",
      proof: "Runtime planner repairs candidates only; it never hides a full scan in planning." },
    { file: "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
      anchor: "fn traceGammaAndBeta",
      operation: "replace",
      patch: "Replace acceptedTemplateCellInvocation with compiled HTP1 tile→accepted-cell packets indexed by SIR1 work rank in the production cut; the separate immutable oracle retains the full traversal.",
      proof: "HEAD invocation arithmetic and per-cell operation order stay identical." },
    { file: "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
      anchor: "fn scatterDensityDeficit",
      operation: "replace",
      patch: "Use the same sealed work generation and compiled donor/receiver closure packet as trace; never infer closure by dilation.",
      proof: "Every possible scatter receiver is included before any tile can skip." },
    { file: "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
      anchor: "fn compareSparseCM12MassResult",
      operation: "insert-after",
      patch: "Add tile-owned compareSparseCM12MassResult: compare u32 words in physical bank0 and bank1 against HEAD destination, cover clipped edge counts, and write one compact rank receipt.",
      proof: "Dual-bank exactness is collision-free; the authority never writes physics." },
    { file: "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
      anchor: "fn validateAndCommitShadowTopology",
      operation: "extend",
      patch: "Seal topology events as next-FCA-generation SIR1 leaves before clearing topology candidate state.",
      proof: "Post-commit invalidations cannot be lost at the frame boundary." },
    { file: "tools/run-sparse-cm12-temporal-regressions.ts",
      anchor: "createScalarFullPathOracleForQA(",
      operation: "replace",
      patch: "Construct paired SRR1 temporal/full-oracle solvers from the same static layout and assert density/velocity/pressure/divergence SHA-256 at every dam5 and canonical60 checkpoint.",
      proof: "No tolerances, active-slot filtering, or diagnostic-time parity substitution." },
    { file: "tools/probe-sparse-cm12-stage-cost.ts",
      anchor: "scalarAuthority",
      operation: "extend",
      patch: "Report candidate events, persistent work/clean counts, comparator, planner, mass arithmetic, and commit separately for ocean B16/P16.",
      proof: "Planner work cannot be hidden inside transport or pressure-topology attribution." },
  ]);

export const SPARSE_CM12_SRR1_CUTOVER_GATES = Object.freeze([
  "shared tsc and all resident WGSL pipelines compile",
  "paired dam5 all four physical hashes exact at every step",
  "paired canonical60 all four physical hashes exact at every checkpoint",
  "FCA/SRR phase fault zero and scheduled receipts equal executed receipts",
  "construction full oracle is immutable and unavailable as runtime fallback",
  "ocean B16/P16 reports planner, comparator, mass, clean/work/event counts separately",
  "mini dam64 front and weakened symmetric expansion retain physical hashes/front properties",
] as const);

export function validateSparseCM12SRR1ProductionManifest(): void {
  const keys = new Set<string>();
  for (const site of SPARSE_CM12_SRR1_PATCH_SITES) {
    const key = `${site.file}\0${site.anchor}`;
    if (keys.has(key)) throw new Error(`duplicate SRR1 patch anchor ${key}`);
    keys.add(key);
    if (!site.patch || !site.proof) throw new Error(`incomplete SRR1 patch ${key}`);
  }
}
