#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const residentTSPath = resolve(
  "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
);
const residentWGSLPath = resolve(
  "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
);
const residentTS = readFileSync(residentTSPath, "utf8");
const residentWGSL = readFileSync(residentWGSLPath, "utf8");

interface TokenRequirement {
  readonly file: "resident-ts" | "resident-wgsl";
  readonly token: string;
  readonly reason: string;
}

const required: readonly TokenRequirement[] = [
  { file: "resident-ts", token: "WebGPUSparseCM12PressureTailAuthority",
    reason: "two copy-isolated INDIRECT tail banks are allocated" },
  { file: "resident-ts", token: ".encodeCopy(encoder, tailBank)",
    reason: "published storage triplets cross a pass boundary and are copied" },
  { file: "resident-ts", token: ".dispatch(activePass, tailBank, family)",
    reason: "eligible tail kernels consume only copied indirect triplets" },
  { file: "resident-ts", token: "publishSparseCM12PressureTailA",
    reason: "tail A publisher is scheduled" },
  { file: "resident-ts", token: "publishSparseCM12PressureTailB",
    reason: "tail B publisher is scheduled" },
  { file: "resident-ts", token: "publishSparseCM12PressureRecoveryTailA",
    reason: "curvature-recovery tail A uses its exact GPU predicate" },
  { file: "resident-ts", token: "publishSparseCM12PressureRecoveryTailB",
    reason: "curvature-recovery tail B uses its exact GPU predicate" },
  { file: "resident-wgsl", token: "psaPressureRecoveryActive",
    reason: "recovery publication is gated by scalars[5] and curvature-loss state" },
  { file: "resident-wgsl", token: "pcePublishFacePreparationCellTile",
    reason: "post-scalar exact-bit changes coalesce preparation roots" },
  { file: "resident-wgsl", token: "pcePublishFaceProjectionPressureTile",
    reason: "pressure exact-bit changes coalesce projection roots" },
  { file: "resident-wgsl", token: "pcePublishPCFThetaRowLeaf",
    reason: "theta changes coalesce PCF row roots" },
  { file: "resident-wgsl", token: "pcePublishPCFMembershipCellTile",
    reason: "PCM transitions coalesce membership roots" },
  { file: "resident-wgsl", token: "pcePublishTopologyBrick",
    reason: "PTR changes retain bounded topology blast" },
  { file: "resident-wgsl", token: "expandSparseCM12PressureProducerPackets",
    reason: "packets expand through immutable HTP1 incidence" },
  { file: "resident-wgsl", token: "finalizeSparseCM12PressureProducerPackets",
    reason: "coverage and zero-indirect fail-close are published" },
  { file: "resident-ts", token: "pressureEventCoalescingIndirectArguments",
    reason: "packet expansion is dispatched from copied GPU counts" },
];

const forbidden: readonly TokenRequirement[] = [
  { file: "resident-ts", token: "dispatchActivity(\"markSparseCM12FacePreparationFromActivity\")",
    reason: "activity-wide preparation authority walk must be removed" },
  { file: "resident-ts", token: "dispatchPressureCell(\"markSparseCM12FaceProjectionFromPressure\")",
    reason: "per-pressure-cell projection incidence publication must be removed" },
  { file: "resident-ts", token: "dispatchPressureCell(\"updatePipelinedState\")",
    reason: "convergence-tail cell arithmetic must use PTL1 copied indirects" },
  { file: "resident-ts", token: "dispatchPressureCell(\"applyPipelinedImage\")",
    reason: "convergence-tail cell image must use PTL1 copied indirects" },
  { file: "resident-ts", token: "dispatchPressureCell(\"applyPipelinedRecovery\")",
    reason: "recovery-tail cell arithmetic must use PTL1 copied indirects" },
];

const sourceFor = (file: TokenRequirement["file"]) =>
  file === "resident-ts" ? residentTS : residentWGSL;
const missing = required.filter(({ file, token }) => !sourceFor(file).includes(token));
const retained = forbidden.filter(({ file, token }) => sourceFor(file).includes(token));

const report = {
  abi: ["PTL1/v1", "PCE1/v1"],
  integrated: missing.length === 0 && retained.length === 0,
  missing: missing.map(({ file, token, reason }) => ({ file, token, reason })),
  forbiddenRetained: retained.map(({ file, token, reason }) => ({ file, token, reason })),
};
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
if (!report.integrated) {
  throw new Error(`PTL1/PCE1 resident integration incomplete: ${missing.length} missing, `
    + `${retained.length} forbidden tokens retained`);
}
