#!/usr/bin/env node
/** Fail-closed static integration gate for the VEX1 + A4D2 production cutover. */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  SPARSE_CM12_VEX_ACTIVITY_PRODUCTION_MANIFEST,
  createSparseCM12VexActivityProductionComposition,
  type SparseCM12ProductionRegion,
  type SparseCM12ProductionStage,
  type SparseCM12VexActivityProductionManifest,
} from "../lib/methods/adaptive-mass/sparse-cm12-vex-activity-production";
import { SPARSE_CM12_VELOCITY_EXTENSION_ROOT_CONTRACT } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";

type MutableManifest = {
  version: number;
  profile: "B16/P16";
  requiredVexRoots: string[];
  requiredActivityHooks: string[];
  requiredVexHooks: string[];
  stages: Array<Omit<SparseCM12ProductionStage, "after" | "dispatch" | "phase"> & {
    after: string[]; dispatch: string; phase: SparseCM12ProductionStage["phase"] }>;
  qaPaths: Array<{ id: string; constructionOnly: boolean;
    runtimeSelectable: boolean; outputSelectable: boolean }>;
  forbiddenDispatchTokens: string[];
};

const fail = (message: string): never => { throw new Error(message); };

function validateRegions(regions: readonly SparseCM12ProductionRegion[]): void {
  for (const region of regions) {
    if (!Number.isSafeInteger(region.start) || !Number.isSafeInteger(region.end)
      || region.start < 0 || region.end <= region.start) {
      fail(`invalid region ${region.name} [${region.start}, ${region.end})`);
    }
  }
  const byBuffer = new Map<string, SparseCM12ProductionRegion[]>();
  for (const region of regions) {
    const key = `${region.buffer}:${region.unit}`;
    byBuffer.set(key, [...(byBuffer.get(key) ?? []), region]);
  }
  for (const [buffer, values] of byBuffer) {
    const sorted = [...values].sort((a, b) => a.start - b.start);
    for (let index = 1; index < sorted.length; index += 1) {
      const prior = sorted[index - 1]!; const current = sorted[index]!;
      if (current.start < prior.end) {
        fail(`${buffer} overlap: ${prior.name} [${prior.start},${prior.end}) and `
          + `${current.name} [${current.start},${current.end})`);
      }
    }
  }
}

function validateManifest(manifest: MutableManifest | SparseCM12VexActivityProductionManifest): void {
  if (manifest.version !== 1 || manifest.profile !== "B16/P16") {
    fail("production manifest must be V1 B16/P16");
  }
  const expectedRoots = SPARSE_CM12_VELOCITY_EXTENSION_ROOT_CONTRACT
    .map((entry) => entry.producer);
  for (const root of expectedRoots) {
    if (!manifest.requiredVexRoots.includes(root)) fail(`missing VEX root: ${root}`);
  }
  if (new Set(manifest.requiredVexRoots).size !== expectedRoots.length) {
    fail("VEX root contract contains duplicates or unrecognized roots");
  }
  for (const hook of ["cm12ActivityBuildTileTrigger",
    "cm12ActivityPublishFramePlanRoot", "cm12ActivityPublishExactBrick"]) {
    if (!manifest.requiredActivityHooks.includes(hook)) fail(`missing A4D2 hook: ${hook}`);
  }
  for (const hook of ["VelocityExtensionRoot", "VelocityExtensionClosure",
    "VelocityExtensionScheduled", "VelocityExtensionOwner", "VelocityExtensionFault"]) {
    if (!manifest.requiredVexHooks.includes(hook)) fail(`missing VEX hook: ${hook}`);
  }

  const ids = new Set<string>();
  for (const stage of manifest.stages) {
    if (ids.has(stage.id)) fail(`duplicate lifecycle stage ${stage.id}`);
    ids.add(stage.id);
    if ((stage.dispatch as string).includes("globalAccepted")) {
      fail(`${stage.id} uses forbidden global accepted dispatch authority`);
    }
  }
  for (const stage of manifest.stages) {
    for (const dependency of stage.after) {
      if (!ids.has(dependency)) fail(`${stage.id} has missing lifecycle dependency ${dependency}`);
    }
  }
  const visited = new Set<string>(); const visiting = new Set<string>();
  const byId = new Map(manifest.stages.map((stage) => [stage.id, stage]));
  const visit = (id: string) => {
    if (visiting.has(id)) fail(`lifecycle cycle at ${id}`);
    if (visited.has(id)) return;
    visiting.add(id);
    for (const dependency of byId.get(id)!.after) visit(dependency);
    visiting.delete(id); visited.add(id);
  };
  for (const id of ids) visit(id);

  const ordered = manifest.stages;
  const position = new Map(ordered.map((stage, index) => [stage.id, index]));
  for (const stage of ordered) {
    for (const dependency of stage.after) {
      if (position.get(dependency)! >= position.get(stage.id)!) {
        fail(`${stage.id} is scheduled before dependency ${dependency}`);
      }
    }
  }
  const phaseRank: Record<string, number> = {
    accepted: 0, collecting: 1, classified: 2, sealed: 3,
    executing: 4, receipted: 5, committed: 6,
  };
  for (const [authority, stages] of (() => {
    const result = new Map<string, typeof ordered[number][]>();
    for (const stage of ordered) {
      result.set(stage.authority, [...(result.get(stage.authority) ?? []), stage]);
    }
    return result;
  })()) {
    let prior = -1;
    for (const stage of stages) {
      const rank = phaseRank[stage.phase];
      if (rank === undefined || rank < prior) {
        fail(`${authority} lifecycle regresses at ${stage.id} (${stage.phase})`);
      }
      prior = rank;
    }
  }
  const coalescingGroups = new Map<string, typeof ordered[number][]>();
  for (const stage of ordered.filter((entry) => entry.coalescingGroup)) {
    const group = stage.coalescingGroup!;
    coalescingGroups.set(group, [...(coalescingGroups.get(group) ?? []), stage]);
  }
  for (const [group, stages] of coalescingGroups) {
    const indices = stages.map((stage) => position.get(stage.id)!).sort((a, b) => a - b);
    if (new Set(stages.map((stage) => stage.authority)).size !== 1
      || indices.at(-1)! - indices[0]! + 1 !== indices.length) {
      fail(`coalescing group ${group} must be contiguous and single-authority`);
    }
    if (stages.slice(0, -1).some((stage) => stage.barrierAfter)) {
      fail(`coalescing group ${group} crosses an internal acceptance barrier`);
    }
    for (let index = indices[0]!; index < indices.at(-1)!; index += 1) {
      const seam = ordered[index]!;
      if (seam.barrierAfter && !stages.includes(seam)) {
        fail(`coalescing group ${group} crosses ${seam.id}'s acceptance barrier`);
      }
    }
  }

  for (let stage = 0; stage < 6; stage += 1) {
    const receipts = ordered.filter((entry) => entry.fplStage === stage);
    if (!receipts.some((entry) => entry.receipt === "executed"
      || entry.receipt === "skipped")) {
      fail(`FPL stage ${stage} lacks truthful executed/skipped receipt`);
    }
  }
  const stage0 = ordered.filter((entry) => entry.fplStage === 0);
  if (!stage0.some((entry) => entry.receipt === "direct")
    || !stage0.some((entry) => entry.receipt === "closure")) {
    fail("FPL stage 0 must distinguish VEX direct roots from recurrence closure");
  }
  const activity = ordered.filter((entry) => entry.authority === "A4D2");
  if (!activity.some((entry) => entry.receipt === "root")
    || !activity.some((entry) => entry.phase === "committed")) {
    fail("A4D2 lacks FPL root publication or accepted activity receipt");
  }
  for (const qa of manifest.qaPaths) {
    if (!qa.constructionOnly || qa.runtimeSelectable || qa.outputSelectable) {
      fail(`QA path ${qa.id} is runtime/output selectable`);
    }
  }
}

const functions = (source: string): ReadonlyMap<string, string> => {
  const starts = [...source.matchAll(/\bfn\s+([A-Za-z0-9_]+)\s*\(/g)];
  return new Map(starts.map((match, index) => [match[1]!, source.slice(match.index!,
    starts[index + 1]?.index ?? source.length)]));
};

function validateCutoverSources(manifest: SparseCM12VexActivityProductionManifest): void {
  const activityPath = fileURLToPath(new URL(
    "../lib/methods/adaptive-mass/sparse-cm12-production-activity.wgsl.ts", import.meta.url));
  const vexPath = fileURLToPath(new URL(
    "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl.ts", import.meta.url));
  const sources = [readFileSync(activityPath, "utf8"), readFileSync(vexPath, "utf8")];
  const runtimeFunctions = new Set([
    "beginProductionActivity", "buildProductionActivityTriggers",
    "classifyProductionActivityBricks", "scanProductionActivityBrickBlocks",
    "scanProductionActivityBlockSums", "scatterProductionActivityBricks",
    "rebuildProductionActivityBricks", "reduceProductionActivityCensusBlocks",
    "commitProductionActivityCensus", "acceptProductionActivity",
    "beginVelocityExtensionCandidate", "sealVelocityExtensionRoots",
    "seedVelocityExtensionRoots", "prepareVelocityExtensionFrontier",
    "expandVelocityExtensionFrontier", "sealVelocityExtensionFrontier",
    "sealVelocityExtensionSeedFrontier", "finalizeVelocityExtensionBlast",
    "initializeVelocityExtensionCandidates", "advanceVelocityExtensionCandidates",
    "commitVelocityExtensionCandidates", "finalizeVelocityExtensionCandidate",
  ]);
  for (const source of sources) {
    const table = functions(source);
    for (const name of runtimeFunctions) {
      const body = table.get(name);
      if (!body) continue;
      for (const token of manifest.forbiddenDispatchTokens) {
        if (body.includes(token)) fail(`${name} contains forbidden dispatch token ${token}`);
      }
    }
  }
  const vex = sources[1]!;
  if (!functions(vex).get("bootstrapVelocityExtensionRoots")
    ?.includes("acceptedTemplateCellInvocation")) {
    fail("construction-only VEX bootstrap lost its explicit full coverage path");
  }
  for (const hook of manifest.requiredActivityHooks) {
    if (!sources[0]!.includes(`${hook}(`)) fail(`A4D2 generator no longer consumes ${hook}`);
  }
  for (const hook of manifest.requiredVexHooks) {
    if (!vex.includes(hook)) fail(`VEX generator no longer consumes ${hook}`);
  }
}

function cloneManifest(): MutableManifest {
  return JSON.parse(JSON.stringify(SPARSE_CM12_VEX_ACTIVITY_PRODUCTION_MANIFEST)) as
    MutableManifest;
}

function expectFailure(label: string, action: () => void): void {
  try { action(); } catch { return; }
  fail(`negative self-check did not reject ${label}`);
}

function main(): void {
  const profiles = [
    { name: "dam64", brickCapacity: 64, cellCapacity: 262_144,
      activityPrefixWords: 131_072, statePrefixFloats: 3_145_728,
      topologyPrefixWords: 524_288, cellWorkgroups: 4096, rowWorkgroups: 12_288 },
    { name: "ocean-b16-p16", brickCapacity: 600, cellCapacity: 2_457_600,
      activityPrefixWords: 1_048_576, statePrefixFloats: 29_491_200,
      topologyPrefixWords: 4_194_304, cellWorkgroups: 38_400, rowWorkgroups: 115_200 },
  ] as const;
  validateManifest(SPARSE_CM12_VEX_ACTIVITY_PRODUCTION_MANIFEST);
  validateCutoverSources(SPARSE_CM12_VEX_ACTIVITY_PRODUCTION_MANIFEST);
  for (const profile of profiles) {
    const composition = createSparseCM12VexActivityProductionComposition(profile);
    validateRegions(composition.regions);
    if (composition.activity.framePlan.baseWords < profile.activityPrefixWords
      || composition.state.acceptedVelocityFloatBase < profile.statePrefixFloats
      || composition.topology.baseWords < profile.topologyPrefixWords) {
      fail(`${profile.name}: composition overlaps an accepted prefix`);
    }
    console.log(`${profile.name}: activity=${composition.activity.totalWords}u32 `
      + `state=${composition.state.floatCount}f32 topology=${composition.topology.totalWords}u32 `
      + `SAW=${composition.scalarAuthority.totalWords}u32 indirect=87u32`);
    for (const region of composition.regions) {
      console.log(`  ${region.buffer}:${region.name} [${region.start},${region.end}) ${region.unit}`);
    }
  }

  const overlap = createSparseCM12VexActivityProductionComposition(profiles[0]).regions
    .map((region) => ({ ...region }));
  overlap[1] = { ...overlap[1]!, start: overlap[0]!.end - 1 };
  expectFailure("arena overlap", () => validateRegions(overlap));
  const missingRoot = cloneManifest(); missingRoot.requiredVexRoots.pop();
  expectFailure("missing root", () => validateManifest(missingRoot));
  const badPhase = cloneManifest(); badPhase.stages[1]!.after = ["fpp-commit"];
  expectFailure("lifecycle cycle/order", () => validateManifest(badPhase));
  const phaseRegression = cloneManifest(); phaseRegression.stages[2]!.phase = "collecting";
  expectFailure("lifecycle phase regression", () => validateManifest(phaseRegression));
  const globalDispatch = cloneManifest();
  globalDispatch.stages[2]!.dispatch = "globalAcceptedCells";
  expectFailure("global accepted dispatch", () => validateManifest(globalDispatch));
  const qaFallback = cloneManifest(); qaFallback.qaPaths[0]!.runtimeSelectable = true;
  expectFailure("runtime QA fallback", () => validateManifest(qaFallback));
  console.log("VEX1+A4D2 production manifest: PASS (overlap/root/lifecycle/dispatch/QA mutations rejected)");
}

main();
