#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { createSparseCM12FramePlanLayout } from "../lib/core/sparse-cm12-frame-plan";
import { createSparseCM12CanonicalMembershipLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-canonical-membership";
import { createSparseCM12FaceProjectionAuthorityLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-face-projection-authority";
import { createSparseCM12FrameControl } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-control";
import { createSparseCM12FramePlanPresentationLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-plan-presentation";
import { createSparseCM12HotTopology } from
  "../lib/methods/adaptive-mass/sparse-cm12-hot-topology";
import { createSparseCM12PersistentPressureCacheLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-persistent-pressure-cache";
import { createSparseCM12PhaseArenaPlan } from
  "../lib/methods/adaptive-mass/sparse-cm12-phase-arenas";
import {
  SPARSE_CM12_PRODUCTION_RESIDENT_TOKEN_CONTRACT,
  assertSparseCM12ProductionIntegrationManifest,
  createSparseCM12ProductionIntegrationManifest,
  type SparseCM12ProductionIntegrationManifestInput,
} from "../lib/methods/adaptive-mass/sparse-cm12-production-integration-manifest";
import { createSparseCM12PressureSolveAuthorityLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-solve-authority";
import { createSparseCM12ScalarWorkAuthority } from
  "../lib/methods/adaptive-mass/sparse-cm12-scalar-work-authority";
import { createSparseCM12ResidentScalarAuthorityLayout } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-scalar-authority";

const fixtureTopology = () => {
  const logical = [2, 1, 1] as const;
  const brick = (coordinate: readonly [number, number, number],
    resolution: SparseBrickResolution): SparseAdaptiveMassBrick => ({
    key: sparseBrickKey(coordinate, logical), coordinate, resolution,
    density: new Float64Array(resolution ** 3),
    gamma: new Float64Array(resolution ** 3).fill(1),
  });
  const atlas = createSparseAdaptiveMassAtlas([32, 16, 16],
    [brick([0, 0, 0], 8), brick([1, 0, 0], 16)], 17, undefined, 16);
  return createSparseCM12HotTopology(buildSparseAtlasCompositeGrid(atlas));
};

function buildInput(): SparseCM12ProductionIntegrationManifestInput {
  const topology = fixtureTopology();
  const cells = topology.layout.cellCount, rows = topology.layout.rowCount;
  const activityPCM = createSparseCM12CanonicalMembershipLayout({
    cellCapacity: cells, rowCapacity: rows, baseWords: 1024,
  });
  const fpl = createSparseCM12FramePlanLayout({ brickCapacity: 2,
    brickFineResolution: 16, baseWords: activityPCM.totalWords });
  const fpp = createSparseCM12FramePlanPresentationLayout({ pageCapacity: 2,
    brickFineResolution: 16, pageResolution: 16, baseWords: fpl.totalWords });
  const scalarCandidate = createSparseCM12ResidentScalarAuthorityLayout({
    baseWords: fpp.totalWords, tileCapacity: 128,
  });
  const fpa = createSparseCM12FaceProjectionAuthorityLayout({ rowCapacity: rows,
    cellCapacity: cells, baseWords: scalarCandidate.totalWords });
  const fca = createSparseCM12FrameControl({ cellWorkgroups: Math.ceil(cells / 64),
    rowWorkgroups: Math.ceil(rows / 64), bodyCapacity: 4, baseWords: 4096,
    d4Capable: true, rigidCapable: true, boundaryCapable: true }).layout;
  const scalarAuthority = createSparseCM12ScalarWorkAuthority({ tileCapacity: 128,
    brickFineResolution: 16, presentationPageResolution: 16 }).layout;

  const plan = (membershipWords: number) => createSparseCM12PhaseArenaPlan({
    cellCount: cells, rowCount: rows, brickCount: 2, candidateBrickCount: 2,
    pressureEdgeCount: topology.layout.directedEdgeCount, pressureCoarseEdgeCount: 4,
    pressureHierarchy: [{ groupCount: 2, edgeCount: 1 }, { groupCount: 1, edgeCount: 0 }],
    pressureMembershipWords: membershipWords, pressureStaticTopologyWords: 256,
    topologyCandidateControlWords: 8192,
    presentation: { metadataBytes: 128, worklistBytes: 128, payloadBytes: 32768 },
    htp1: topology.layout,
  });
  // Membership's absolute base is independent of its own capacity. The large
  // planning pass derives the exact required serial tail without hand arithmetic.
  const planningArena = plan(2_000_000);
  const membership = planningArena.buffers.find((entry) =>
    entry.id === "cm12.pressure-cache")!.regions.find((entry) =>
    entry.name === "membership")!;
  const membershipBase = membership.offsetBytes / 4;
  const planningPCM = createSparseCM12CanonicalMembershipLayout({
    cellCapacity: cells, rowCapacity: rows, baseWords: membershipBase,
  });
  const planningPCF = createSparseCM12PersistentPressureCacheLayout({
    phaseArena: planningArena, htp1: topology.layout,
    controlOffsetWords: planningPCM.totalWords - membershipBase,
  });
  const planningPSA = createSparseCM12PressureSolveAuthorityLayout({ brickCapacity: 2,
    hierarchyLevelCounts: [2, 1], baseWords: planningPCF.controlEndWords });
  const exactMembershipWords = planningPSA.totalWords - membershipBase;
  const phaseArena = plan(exactMembershipWords);
  const pcm = createSparseCM12CanonicalMembershipLayout({ cellCapacity: cells,
    rowCapacity: rows, baseWords: membershipBase });
  const pcf = createSparseCM12PersistentPressureCacheLayout({ phaseArena,
    htp1: topology.layout, controlOffsetWords: pcm.totalWords - membershipBase });
  const psa = createSparseCM12PressureSolveAuthorityLayout({ brickCapacity: 2,
    hierarchyLevelCounts: [2, 1], baseWords: pcf.controlEndWords });
  return { activity: { pcmTombstone: activityPCM, fpl, fpp, scalarCandidate, fpa },
    topology: { fca }, scalarAuthority, pressure: { phaseArena, pcm, pcf, psa } };
}

function requireResidentTokens(): void {
  const hostPath = "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts";
  const wgslPath = "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts";
  const host = readFileSync(hostPath, "utf8"), wgsl = readFileSync(wgslPath, "utf8");
  const missingHost = SPARSE_CM12_PRODUCTION_RESIDENT_TOKEN_CONTRACT.host
    .filter((token) => !host.includes(token));
  const missingWGSL = SPARSE_CM12_PRODUCTION_RESIDENT_TOKEN_CONTRACT.wgsl
    .filter((token) => !wgsl.includes(token));
  if (missingHost.length || missingWGSL.length) {
    throw new Error(`resident integration is incomplete; host=[${missingHost.join(", ")}], `
      + `wgsl=[${missingWGSL.join(", ")}]`);
  }
}

function main(): void {
  const input = buildInput();
  const manifest = createSparseCM12ProductionIntegrationManifest(input);
  if (manifest.regions.length < 12 || manifest.bindings.length !== 42
    || manifest.copies.length !== 27 || manifest.authorities.length !== 7
    || manifest.gates.length !== 6) {
    throw new Error("manifest family count drift");
  }
  const pressureRegions = manifest.regions.filter((entry) =>
    ["pressure.pcm1", "pressure.pcf1-pca1", "pressure.psa1"].includes(entry.id));
  if (pressureRegions.map((entry) => entry.id).join(",")
      !== "pressure.pcm1,pressure.pcf1-pca1,pressure.psa1") {
    throw new Error("pressure authority order drift");
  }
  const broken = { ...manifest, authorities: manifest.authorities.map((entry) =>
    entry.id === "PCF1.fine" ? { ...entry, closures: [] } : entry) };
  let rejected = false;
  try { assertSparseCM12ProductionIntegrationManifest(broken, input); }
  catch { rejected = true; }
  if (!rejected) throw new Error("manifest checker accepted an omitted PCF closure");
  if (process.argv.includes("--resident")) requireResidentTokens();
  console.log(`Sparse CM12 production manifest: ${manifest.regions.length} regions, `
    + `${manifest.bindings.length} phase bindings, ${manifest.copies.length} indirect seams, `
    + `${manifest.passes.length} ordered passes; producer/closure/receipt omissions rejected`
    + (process.argv.includes("--resident") ? "; resident tokens complete" : ""));
}

main();
