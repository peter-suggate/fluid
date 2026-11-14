/** The S2g audit report (lane H): G3's delivered volume, the jet trajectory
 * (G8, G6) and the right-chamber profile difference, from seep-bench runs.
 *
 * node --import tsx tools/seep-audit-report.ts --nb-audit=RUN.jsonl --seep-audit=RUN.jsonl
 *   --refs=A.jsonl,B.jsonl --seep-check=RUN.jsonl --out=DIR [--name=audit]
 *
 * --nb-audit / --seep-audit: fenced seep-bench runs with --audit (tools/seep-audit-cells.ts rows);
 * --refs: the NB-FLIP check runs that set the gates; --seep-check: the judged Seep run.
 * Writes DIR/NAME.md and DIR/NAME.json. Changes no gate: it reports what a
 * redefinition would give, for the coordinator to decide. */
import assert from "node:assert/strict";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import type { AuditRow } from "./seep-audit-cells";
import { SEEP_BIN_CELLS, type SeepObservation } from "./seep-observables";

const args = process.argv.slice(2);
const option = (key: string) => args.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
for (const a of args) assert.ok(/^--(nb-audit|seep-audit|refs|seep-check|out|name)=.+$/.test(a), `Unknown option: ${a}`);
const load = (file: string) => readFileSync(resolve(file), "utf8").trim().split("\n").map(l => JSON.parse(l) as Record<string, unknown>);
const samplesOf = (rows: Record<string, unknown>[]) => rows.filter(r => r.kind === "sample") as unknown as SeepObservation[];
const auditsOf = (rows: Record<string, unknown>[]) => rows.filter(r => r.kind === "audit") as unknown as AuditRow[];
const need = (k: string) => { const v = option(k); assert.ok(v, `--${k}=...`); return v; };
const nbRun = load(need("nb-audit")), seepRun = load(need("seep-audit"));
const refs = need("refs").split(",").map(f => samplesOf(load(f)));
const seepCheck = samplesOf(load(need("seep-check")));
const out = resolve(need("out")), name = option("name") ?? "audit";
mkdirSync(out, { recursive: true });

const at = (s: SeepObservation[], step: number) => { const r = s.find(x => x.step === step); assert.ok(r, `no sample at step ${step}`); return r; };
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
const f = (v: number | null | undefined, d = 3) => v === null || v === undefined ? "-" : v.toFixed(d);
const nbAudit = auditsOf(nbRun), seepAudit = auditsOf(seepRun);
const auditAt = (a: AuditRow[], step: number) => a.find(r => r.step === step);
const md: string[] = [`# S2g audit: G3 delivered, jet trajectory, right-chamber profile`, "",
  `NB-FLIP audit run: ${need("nb-audit")}; Seep audit run: ${need("seep-audit")}; NB-FLIP references: ${need("refs")}; judged Seep run: ${need("seep-check")}.`,
  `Generated ${new Date().toISOString()} by tools/seep-audit-report.ts. Volumes in litres, x and y in cells (h = 5 mm), u in m/s.`, ""];
const json: Record<string, unknown> = {};

// (a) The wall zone, open and solid.
{
  md.push("## (a) Wall zone: open throat cells vs solid cells", "",
    "| step | NB open | NB solid | NB 4h | NB total (audit) | NB observer wall | Seep open | Seep solid | Seep observer wall |", "|---|---|---|---|---|---|---|---|---|");
  const rows = nbAudit.map(a => {
    const s = seepAudit.find(r => r.step === a.step);
    const nbObs = samplesOf(nbRun).find(r => r.step === a.step)?.liquid_L.wall, seepObs = samplesOf(seepRun).find(r => r.step === a.step)?.liquid_L.wall;
    md.push(`| ${a.step} | ${f(a.wall_L.open, 4)} | ${f(a.wall_L.solid, 4)} | ${f(a.wall_L.coarse, 4)} | ${f(a.wall_L.total, 4)} | ${f(nbObs, 4)} | ${f(s?.wall_L.open, 4)} | ${f(s?.wall_L.solid, 4)} | ${f(seepObs, 4)} |`);
    return { step: a.step, nb: a.wall_L, nbObserver: nbObs, seep: s?.wall_L, seepObserver: seepObs, seepOpenMismatch: s?.openMismatch };
  });
  const a0 = auditAt(nbAudit, 0)!, growth = nbAudit.filter(a => a.step > 0).map(a => ({ step: a.step, open: a.wall_L.open - a0.wall_L.open, solid: a.wall_L.solid - a0.wall_L.solid, coarse: a.wall_L.coarse - a0.wall_L.coarse }));
  md.push("", "NB-FLIP wall-zone growth since step 0: " + growth.map(g => `${g.step}: open ${g.open >= 0 ? "+" : ""}${f(g.open, 4)}, solid ${g.solid >= 0 ? "+" : ""}${f(g.solid, 4)}, 4h ${g.coarse >= 0 ? "+" : ""}${f(g.coarse, 4)}`).join("; ") + ".");
  const mismatch = seepAudit.map(a => a.openMismatch ?? 0);
  md.push(`Seep's own open flag against the scene's throat patch: ${Math.max(...mismatch)} disagreeing cells at most.`, "");
  // G3 with delivered = left loss - change in open wall liquid (the NB audit run's open change; its last audit for later checkpoints).
  const lastOpen = (step: number) => { const c = nbAudit.filter(a => a.step <= step).at(-1)!; return { open: c.wall_L.open - a0.wall_L.open, from: c.step }; };
  md.push("G3 as written (delivered = left loss - wall change) and with delivered = left loss - change in open-cell wall liquid:", "",
    "| step | Seep right | NB right | lower | upper as written | upper, open-cell delivered | as written | open-cell |", "|---|---|---|---|---|---|---|---|");
  const g3 = [300, 600, 900, 1200].map(step => {
    const right = refs.map(r => at(r, step).liquid_L.right), nbRight = mean(right), S = Math.max(...right) - Math.min(...right);
    const leftLoss = mean(refs.map(r => r[0]!.liquid_L.left - at(r, step).liquid_L.left)), wallChange = mean(refs.map(r => at(r, step).liquid_L.wall - r[0]!.liquid_L.wall));
    const open = lastOpen(step), low = nbRight - Math.max(0.05 * nbRight, 3 * S);
    const upper = (leftLoss - wallChange) * 1.05, upperOpen = (leftLoss - open.open) * 1.05, v = at(seepCheck, step).liquid_L.right;
    md.push(`| ${step} | ${f(v)} | ${f(nbRight)} | ${f(low)} | ${f(upper)} | ${f(upperOpen)}${open.from !== step ? ` (open change at ${open.from})` : ""} | ${v >= low && v <= upper ? "pass" : "FAIL"} | ${v >= low && v <= upperOpen ? "pass" : "FAIL"} |`);
    return { step, seepRight: v, nbRight, low, upper, upperOpen, openChangeFrom: open.from, leftLoss, wallChange, openChange: open.open };
  });
  json.wall = { rows, growth, g3 };
  md.push("");
}

// (b) The jet trajectory.
{
  md.push("## (b) Jet trajectory: 8-cell slices of the right chamber, liquid at y >= 8", "",
    "Ballistic centreline y(x) = y0 - g x^2 / (2 u0^2) from the throat exit (x = 256), y0 = 65, u0 = the throat's u at that step.",
    "Measured impact: the x where the jet's mean y first falls below 10 cells; ballistic impact: where the centreline reaches y = 8.", "");
  const trajectory: Record<string, unknown>[] = [];
  for (const step of [300, 600]) {
    const nb = auditAt(nbAudit, step), sp = auditAt(seepAudit, step);
    if (!nb || !sp) continue;
    md.push(`### Step ${step}`, "",
      `| | NB-FLIP | Seep |`, "|---|---|---|",
      `| throat u0 m/s | ${f(nb.throatU_m_s)} | ${f(sp.throatU_m_s)} |`,
      `| jet column (y >= 8, x < 320) L | ${f(nb.jetColumn_L, 4)} | ${f(sp.jetColumn_L, 4)} |`,
      `| ballistic column (u0 A x 64h / u0) L | ${f(nb.ballisticColumn_L, 4)} | ${f(sp.ballisticColumn_L, 4)} |`,
      `| impact x measured / ballistic | ${f(nb.measuredImpactX, 1)} / ${f(nb.ballisticImpactX, 1)} | ${f(sp.measuredImpactX, 1)} / ${f(sp.ballisticImpactX, 1)} |`, "",
      "| x | NB jet L | NB mean y | NB mean u | Seep jet L | Seep mean y | Seep mean u | ballistic y (NB u0 / Seep u0) | NB all L | Seep all L |", "|---|---|---|---|---|---|---|---|---|---|");
    nb.slices.forEach((a, i) => {
      const b = sp.slices[i]!;
      md.push(`| ${a.x0}-${a.x1} | ${f(a.jet_L, 4)} | ${f(a.meanY, 1)} | ${f(a.meanU, 2)} | ${f(b.jet_L, 4)} | ${f(b.meanY, 1)} | ${f(b.meanU, 2)} | ${f(a.ballisticY, 1)} / ${f(b.ballisticY, 1)} | ${f(a.all_L, 4)} | ${f(b.all_L, 4)} |`);
    });
    md.push("");
    trajectory.push({ step, nb, seep: sp });
  }
  json.trajectory = trajectory;
}

// (c) The right-chamber profile difference.
{
  md.push("## (c) Right-chamber bins: Seep minus NB-FLIP", "",
    `Bins of ${SEEP_BIN_CELLS} cells from x = 256. NB-FLIP = mean of the reference runs. "missing" = NB-FLIP's delivered volume (left loss - open-cell wall change, (a)) minus its right volume:`,
    "liquid the chamber should hold but does not. Scenario: that liquid restored evenly over the floor (a loss proportional to film area).", "");
  const nbOpen = (step: number) => { const a0 = auditAt(nbAudit, 0)!, c = nbAudit.filter(a => a.step <= step).at(-1)!; return c.wall_L.open - a0.wall_L.open; };
  const centroid = (bins: number[]) => { const t = bins.reduce((a, b) => a + b, 0); return 256 + bins.reduce((a, b, i) => a + b * (i + 0.5) * SEEP_BIN_CELLS, 0) / t; };
  const profile: Record<string, unknown>[] = [];
  for (const step of [600, 1200]) {
    const nbBins = refs[0]!.length ? at(refs[0]!, step).rightBins_L.map((_, i) => mean(refs.map(r => at(r, step).rightBins_L[i]!))) : [];
    const spBins = at(seepCheck, step).rightBins_L, diff = spBins.map((v, i) => v - nbBins[i]!);
    const nbRight = nbBins.reduce((a, b) => a + b, 0), spRight = spBins.reduce((a, b) => a + b, 0);
    const delivered = mean(refs.map(r => r[0]!.liquid_L.left - at(r, step).liquid_L.left)) - nbOpen(step), missing = delivered - nbRight;
    const restored = nbBins.map(b => b + missing / nbBins.length);
    const cNb = centroid(nbBins), cSp = centroid(spBins), cRestored = centroid(restored);
    md.push(`### Step ${step}`, "",
      `NB right ${f(nbRight)} L, delivered ${f(delivered)} L, missing ${f(missing)} L (${f(100 * missing / delivered, 1)}% of delivered); Seep right ${f(spRight)} L.`,
      `Centroid: NB ${f(cNb, 1)}, Seep ${f(cSp, 1)} (gap ${f(cSp - cNb, 1)}); NB with the missing liquid restored evenly over the floor ${f(cRestored, 1)}, which closes ${f(100 * (cRestored - cNb) / (cSp - cNb), 0)}% of the gap.`, "",
      "| bin x | NB L | Seep L | Seep - NB L | NB depth cells | Seep depth cells |", "|---|---|---|---|---|---|");
    const L = 1.25e-4, area = SEEP_BIN_CELLS * 64;
    nbBins.forEach((b, i) => md.push(`| ${256 + i * SEEP_BIN_CELLS}-${256 + (i + 1) * SEEP_BIN_CELLS} | ${f(b)} | ${f(spBins[i])} | ${diff[i]! >= 0 ? "+" : ""}${f(diff[i])} | ${f(b / L / area, 2)} | ${f(spBins[i]! / L / area, 2)} |`));
    md.push("");
    profile.push({ step, nbBins, seepBins: spBins, diff, nbRight, seepRight: spRight, delivered, missing, centroid: { nb: cNb, seep: cSp, nbRestored: cRestored } });
  }
  json.profile = profile;
}

writeFileSync(`${out}/${name}.md`, md.join("\n") + "\n");
writeFileSync(`${out}/${name}.json`, JSON.stringify(json, null, 1) + "\n");
console.log(md.join("\n"));
console.log(`wrote ${out}/${name}.md and .json`);
