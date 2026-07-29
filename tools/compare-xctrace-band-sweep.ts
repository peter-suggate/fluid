/** Compare the fully-labelled representative frames emitted by
 * profile-mini-dam-xctrace.ts for interface bands 0, 1, and 4. */
import { readFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";

interface PassRecord {
  readonly label: string;
  readonly gpuMsPerFrame: number;
}

interface Summary {
  readonly wall: { readonly baselineMsPerAdvance: number; readonly tracedMsPerAdvance: number };
  readonly gpu: {
    readonly intervalMsPerFrame: number;
    readonly busyMsPerFrame: number;
    readonly gapMsPerFrame: number;
    readonly diagnosticBlitMsPerFrame: number;
  };
  readonly counters: { readonly meanOccupancy: number; readonly meanAlu: number };
  readonly passes: readonly PassRecord[];
}

interface SmokeResult {
  readonly phase?: string;
  readonly globalFineLevelSetLogicalBrickCount: number;
  readonly finalGlobalFineGeneration: {
    readonly activePages: number;
    readonly topologyInterfaceBricks: number;
    readonly topologyDilationBrickRings: number;
    readonly redistanceAcceptedCells: number;
    readonly transportProcessed: number;
    readonly transportVelocityUnavailable: number;
  };
}

const flag = (name: string): string | undefined => process.argv
  .find((argument) => argument.startsWith(`--${name}=`))?.slice(name.length + 3);
const directory = resolve(flag("root") ?? "artifacts/xctrace-band-sweep-2026-07-29");
const bands = [0, 1, 4] as const;
type Band = (typeof bands)[number];
const summaries = new Map<Band, Summary>(bands.map((band) => [band, JSON.parse(readFileSync(
  `${directory}/band${band}/summary.json`, "utf8")) as Summary]));
const baselineResults = new Map<Band, SmokeResult>(bands.map((band) => {
  let result: SmokeResult | undefined;
  for (const line of readFileSync(`${directory}/band${band}/baseline.log`, "utf8").split("\n")) {
    if (!line.startsWith("{")) continue;
    try {
      const record = JSON.parse(line) as SmokeResult;
      if (record.phase === "result") result = record;
    } catch { /* progress lines are not all JSON */ }
  }
  if (!result) throw new Error(`band ${band} baseline log has no result record`);
  return [band, result];
}));
const passMaps = new Map<Band, Map<string, number>>(bands.map((band) => [band,
  new Map(summaries.get(band)?.passes.map((pass) => [pass.label, pass.gpuMsPerFrame]))]));

const family = (label: string): string => {
  if (/^Fine JFA/.test(label)) return "redistance / JFA";
  if (/Advect fine phi|structured fine transport/i.test(label)) return "fine transport";
  if (/SPGrid|MGPCG|pressure|Section 4\.3|Section 6\.3|divergence RHS/i.test(label)) {
    return "pressure / multigrid";
  }
  if (/volume/i.test(label)) return "volume correction";
  if (/velocity|Project structured|Force and constrain|Advect structured|Reconstruct projected|Transfer accepted/i.test(label)) {
    return "structured velocity / projection";
  }
  if (/fine-band|fine band|global fine|fine-summary|fine seed|fine-seed|brick residency|page delta|changed-key|air-support|Fine-to-coarse|coarse level set|topology|owner page|power descriptor|power cell/i.test(label)) {
    return "topology / fine publication";
  }
  return "other recurring work";
};

const labels = [...new Set(bands.flatMap((band) => [...(passMaps.get(band)?.keys() ?? [])]))];
const rows = labels.map((label) => {
  const ms = (band: Band): number => passMaps.get(band)?.get(label) ?? 0;
  return {
    label,
    family: family(label),
    band0Ms: ms(0),
    band1Ms: ms(1),
    band4Ms: ms(4),
    band0DeltaMs: ms(0) - ms(4),
    band1DeltaMs: ms(1) - ms(4),
  };
}).sort((left, right) => right.band4Ms - left.band4Ms || left.label.localeCompare(right.label));

const families = [...new Set(rows.map((row) => row.family))].map((name) => {
  const selected = rows.filter((row) => row.family === name);
  const sum = (key: "band0Ms" | "band1Ms" | "band4Ms"): number =>
    selected.reduce((total, row) => total + row[key], 0);
  return {
    family: name,
    band0Ms: sum("band0Ms"),
    band1Ms: sum("band1Ms"),
    band4Ms: sum("band4Ms"),
    band0DeltaMs: sum("band0Ms") - sum("band4Ms"),
    band1DeltaMs: sum("band1Ms") - sum("band4Ms"),
  };
}).sort((left, right) => right.band4Ms - left.band4Ms);

const global = bands.map((band) => {
  const summary = summaries.get(band);
  if (!summary) throw new Error(`missing band ${band} summary`);
  const baseline = baselineResults.get(band);
  if (!baseline) throw new Error(`missing band ${band} baseline result`);
  const fine = baseline.finalGlobalFineGeneration;
  return {
    band,
    cleanMsPerAdvance: summary.wall.baselineMsPerAdvance,
    isolatedMsPerAdvance: summary.wall.tracedMsPerAdvance,
    attributedGpuMs: summary.gpu.intervalMsPerFrame,
    gpuBusyMs: summary.gpu.busyMsPerFrame,
    gpuGapMs: summary.gpu.gapMsPerFrame,
    isolationBlitMs: summary.gpu.diagnosticBlitMsPerFrame,
    meanOccupancy: summary.counters.meanOccupancy,
    meanAlu: summary.counters.meanAlu,
    activeFinePages: fine.activePages,
    logicalFinePages: baseline.globalFineLevelSetLogicalBrickCount,
    interfaceFinePages: fine.topologyInterfaceBricks,
    dilationBrickRings: fine.topologyDilationBrickRings,
    redistanceAcceptedCells: fine.redistanceAcceptedCells,
    transportProcessed: fine.transportProcessed,
    transportVelocityUnavailable: fine.transportVelocityUnavailable,
  };
});

const material = (band: 0 | 1): { savedMs: number; regressedMs: number;
  cheaper: number; dearer: number; flat: number } => {
  let savedMs = 0;
  let regressedMs = 0;
  let cheaper = 0;
  let dearer = 0;
  let flat = 0;
  for (const row of rows) {
    const delta = band === 0 ? row.band0DeltaMs : row.band1DeltaMs;
    if (delta < -0.01) { savedMs -= delta; cheaper += 1; }
    else if (delta > 0.01) { regressedMs += delta; dearer += 1; }
    else flat += 1;
  }
  return { savedMs, regressedMs, cheaper, dearer, flat };
};

const comparison = {
  generatedAt: new Date().toISOString(),
  referenceBand: 4,
  materialThresholdMs: 0.01,
  global,
  material: { band0: material(0), band1: material(1) },
  families,
  stages: rows,
};
await writeFile(`${directory}/comparison.json`, `${JSON.stringify(comparison, null, 2)}\n`);

const csvCell = (value: string | number): string => {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};
const csv = [
  ["family", "label", "band0_ms", "band1_ms", "band4_ms", "band0_minus_4_ms", "band1_minus_4_ms"],
  ...rows.map((row) => [row.family, row.label, row.band0Ms, row.band1Ms, row.band4Ms,
    row.band0DeltaMs, row.band1DeltaMs]),
].map((line) => line.map(csvCell).join(",")).join("\n");
await writeFile(`${directory}/stage-comparison.csv`, `${csv}\n`);

const fixed = (value: number): string => value.toFixed(3);
const percentGain = (candidate: number, reference: number): string =>
  `${(100 * (reference - candidate) / reference).toFixed(1)}%`;
const reference = global.find((entry) => entry.band === 4)!;
const top = (band: 0 | 1, direction: "saving" | "regression") => [...rows]
  .sort((left, right) => {
    const leftDelta = band === 0 ? left.band0DeltaMs : left.band1DeltaMs;
    const rightDelta = band === 0 ? right.band0DeltaMs : right.band1DeltaMs;
    return direction === "saving" ? leftDelta - rightDelta : rightDelta - leftDelta;
  }).slice(0, 12);
const markdown = `# Mini-dam xctrace band comparison

Band 4 is the reference. Each capture includes a clean 500-advance wall-clock control and one fully labelled, hardware-counter-attributed representative advance. Negative deltas are savings.

| Band | Clean ms/advance | Clean gain vs 4 | Attributed GPU ms | GPU gain vs 4 | Mean occupancy |
|---:|---:|---:|---:|---:|---:|
${global.map((entry) => `| ${entry.band} | ${fixed(entry.cleanMsPerAdvance)} | ${percentGain(entry.cleanMsPerAdvance, reference.cleanMsPerAdvance)} | ${fixed(entry.attributedGpuMs)} | ${percentGain(entry.attributedGpuMs, reference.attributedGpuMs)} | ${(100 * entry.meanOccupancy).toFixed(1)}% |`).join("\n")}

## Workload census at advance 500

| Band | Active fine pages | Fine-page occupancy | Interface pages | Dilation rings | Redistance cells | Transport cells | Unavailable velocity |
|---:|---:|---:|---:|---:|---:|---:|---:|
${global.map((entry) => `| ${entry.band} | ${entry.activeFinePages} | ${(100 * entry.activeFinePages / entry.logicalFinePages).toFixed(1)}% | ${entry.interfaceFinePages} | ${entry.dilationBrickRings} | ${entry.redistanceAcceptedCells} | ${entry.transportProcessed} | ${entry.transportVelocityUnavailable} |`).join("\n")}

## Stage-family totals

| Family | Band 0 ms | Band 1 ms | Band 4 ms | 0 minus 4 | 1 minus 4 |
|---|---:|---:|---:|---:|---:|
${families.map((entry) => `| ${entry.family} | ${fixed(entry.band0Ms)} | ${fixed(entry.band1Ms)} | ${fixed(entry.band4Ms)} | ${fixed(entry.band0DeltaMs)} | ${fixed(entry.band1DeltaMs)} |`).join("\n")}

At a 0.01 ms/stage materiality threshold, band 0 has ${material(0).cheaper} cheaper, ${material(0).dearer} dearer, and ${material(0).flat} flat stages; gross savings are ${fixed(material(0).savedMs)} ms and regressions ${fixed(material(0).regressedMs)} ms. Band 1 has ${material(1).cheaper} cheaper, ${material(1).dearer} dearer, and ${material(1).flat} flat stages; gross savings are ${fixed(material(1).savedMs)} ms and regressions ${fixed(material(1).regressedMs)} ms.

## Interpretation

- Band 1 is the lowest moving-surface setting that passed the strict 500-step quality run. Its intended fine-band wins are present: redistance/JFA saves 1.106 ms, fine transport 0.907 ms, structured velocity/projection 0.199 ms, and volume correction 0.122 ms in the isolated frame.
- Those wins are mostly offset in the diagnostic graph by pressure/multigrid (+1.457 ms) and topology/fine publication (+0.317 ms), especially \`SPGrid V-cycle - candidate link parent chains\` (+0.935 ms). That leaves a 1.0% isolated-stage gain, while the clean 500-step shipping graph gains 5.1%.
- Band 0 shows the strong endpoint response: 23.0% faster clean and 43.4% less attributed GPU work. Its redistance/JFA and transport families alone save 10.844 ms. It is still a performance probe, not a quality recommendation: its under-expanded front changes the pressure workload, and its 5.858 ms pressure-family saving cannot be attributed solely to sparse-band efficiency.
- The clean controls preserve normal pass fusion and are authoritative for shipping wall time. Full label isolation deliberately splits every stage into a Metal encoder, so its absolute wall time and some cross-stage scheduling differ from production.
- Family assignment is a label-based roll-up for navigation. The individual label timings in the CSV and JSON are the exact xctrace attribution.

Reports: [band 0](band0/report.html), [band 1](band1/report.html), [band 4](band4/report.html). The corresponding \`mini-dam.trace\` bundles are retained beside each report.

## Largest individual deltas

| Band | Direction | Delta ms | Candidate ms | Band 4 ms | Stage |
|---:|---|---:|---:|---:|---|
${([0, 1] as const).flatMap((band) => (["saving", "regression"] as const).flatMap((direction) =>
    top(band, direction).map((row) => {
      const candidate = band === 0 ? row.band0Ms : row.band1Ms;
      const delta = band === 0 ? row.band0DeltaMs : row.band1DeltaMs;
      return `| ${band} | ${direction} | ${fixed(delta)} | ${fixed(candidate)} | ${fixed(row.band4Ms)} | ${row.label} |`;
    }))).join("\n")}

The complete 255-stage table is in \`stage-comparison.csv\`; machine-readable global, family, and stage records are in \`comparison.json\`.
`;
await writeFile(`${directory}/comparison.md`, markdown);
console.log(`wrote ${rows.length} stage rows to ${directory}`);
