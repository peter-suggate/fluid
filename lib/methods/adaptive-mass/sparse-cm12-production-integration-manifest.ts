import type { SparseCM12FramePlanLayout } from "../../core/sparse-cm12-frame-plan";
import type { SparseCM12CanonicalMembershipLayout } from "./sparse-cm12-canonical-membership";
import {
  SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER,
  type SparseCM12FaceProjectionAuthorityLayout,
} from "./sparse-cm12-face-projection-authority";
import {
  SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT,
  SPARSE_CM12_FRAME_CONTROL_INDIRECT_WORDS,
  type SparseCM12FrameControlLayout,
} from "./sparse-cm12-frame-control";
import type { SparseCM12FramePlanPresentationLayout } from
  "./sparse-cm12-frame-plan-presentation";
import {
  SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER,
  SPARSE_CM12_PRESSURE_CACHE_HEADER,
  type SparseCM12PersistentPressureCacheLayout,
} from "./sparse-cm12-persistent-pressure-cache";
import type { SparseCM12PhaseArenaPlan } from "./sparse-cm12-phase-arenas";
import {
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER,
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER,
  type SparseCM12PressureSolveAuthorityLayout,
} from "./sparse-cm12-pressure-solve-authority";
import type { SparseCM12ScalarWorkAuthorityLayout } from
  "./sparse-cm12-scalar-work-authority";
import type { SparseCM12ResidentScalarAuthorityLayout } from
  "./webgpu-sparse-cm12-scalar-authority";

export const SPARSE_CM12_PRODUCTION_INTEGRATION_MANIFEST_VERSION = 1;
export const SPARSE_CM12_PRODUCTION_ALIGNMENT_WORDS = 64;

export type SparseCM12ProductionArena = "activity" | "topology" | "scalar-authority"
  | "pressure-cache" | "indirect";

export interface SparseCM12ProductionRegion {
  readonly id: string;
  readonly arena: SparseCM12ProductionArena;
  readonly buffer: string;
  readonly firstWord: number;
  readonly endWord: number;
  readonly authority: "current" | "candidate" | "accepted" | "tombstone" | "immutable";
}

export interface SparseCM12ProductionBinding {
  readonly phase: string;
  readonly binding: number;
  readonly buffer: string;
  readonly access: "uniform" | "read-only-storage" | "storage";
}

export interface SparseCM12ProductionCopySeam {
  readonly id: string;
  readonly sourceBuffer: string;
  readonly sourceOffsetBytes: number;
  readonly destinationBuffer: string;
  readonly destinationOffsetBytes: number;
  readonly sizeBytes: number;
  readonly sealedBy: string;
  readonly consumedBy: readonly string[];
}

export interface SparseCM12ProductionAuthorityContract {
  readonly id: string;
  readonly producers: readonly string[];
  readonly closures: readonly string[];
  readonly receipts: readonly string[];
  readonly faultIndirects: readonly string[];
}

export interface SparseCM12ProductionPass {
  readonly id: string;
  readonly pass: string;
  readonly dispatch: string;
  readonly after: readonly string[];
  readonly indirect?: string;
}

export interface SparseCM12ProductionSliceGate {
  readonly slice: string;
  readonly requiredStaticChecks: readonly string[];
  readonly dam: readonly string[];
  readonly ocean: readonly string[];
}

export interface SparseCM12ProductionIntegrationManifest {
  readonly version: 1;
  readonly regions: readonly SparseCM12ProductionRegion[];
  readonly bindings: readonly SparseCM12ProductionBinding[];
  readonly copies: readonly SparseCM12ProductionCopySeam[];
  readonly authorities: readonly SparseCM12ProductionAuthorityContract[];
  readonly passes: readonly SparseCM12ProductionPass[];
  readonly orderingSafePassCoalescings: readonly string[];
  readonly forbiddenDispatchCoalescings: readonly string[];
  readonly qa: {
    readonly constructionOnly: true;
    readonly productionOracleEnabled: false;
    readonly comparedAuthorities: readonly string[];
  };
  readonly gates: readonly SparseCM12ProductionSliceGate[];
}

export interface SparseCM12ProductionIntegrationManifestInput {
  /** Current option-B activity layouts. Activity PCM becomes an inert tombstone at cutover. */
  readonly activity: {
    readonly pcmTombstone: SparseCM12CanonicalMembershipLayout;
    readonly fpl: SparseCM12FramePlanLayout;
    readonly fpp: SparseCM12FramePlanPresentationLayout;
    readonly scalarCandidate: SparseCM12ResidentScalarAuthorityLayout;
    readonly fpa: SparseCM12FaceProjectionAuthorityLayout;
  };
  readonly topology: { readonly fca: SparseCM12FrameControlLayout };
  readonly scalarAuthority: SparseCM12ScalarWorkAuthorityLayout;
  readonly pressure: {
    readonly phaseArena: SparseCM12PhaseArenaPlan;
    /** Sole production PCM authority, relocated into PressureCache.membership. */
    readonly pcm: SparseCM12CanonicalMembershipLayout;
    readonly pcf: SparseCM12PersistentPressureCacheLayout;
    readonly psa: SparseCM12PressureSolveAuthorityLayout;
  };
}

const alignWords = (value: number): number => Math.ceil(value
  / SPARSE_CM12_PRODUCTION_ALIGNMENT_WORDS) * SPARSE_CM12_PRODUCTION_ALIGNMENT_WORDS;

const region = (id: string, arena: SparseCM12ProductionArena, buffer: string,
  firstWord: number, endWord: number, authority: SparseCM12ProductionRegion["authority"]):
SparseCM12ProductionRegion => Object.freeze({ id, arena, buffer, firstWord, endWord, authority });

const snapshot = (id: string, sourceBuffer: string, sourceOffsetBytes: number,
  destinationBuffer: string, destinationOffsetBytes: number, sealedBy: string,
  consumedBy: readonly string[], sizeBytes = 12): SparseCM12ProductionCopySeam => Object.freeze({
  id, sourceBuffer, sourceOffsetBytes, destinationBuffer, destinationOffsetBytes,
  sizeBytes, sealedBy, consumedBy: Object.freeze([...consumedBy]),
});

const authority = (id: string, producers: readonly string[], closures: readonly string[],
  receipts: readonly string[], faultIndirects: readonly string[]):
SparseCM12ProductionAuthorityContract => Object.freeze({ id,
  producers: Object.freeze([...producers]), closures: Object.freeze([...closures]),
  receipts: Object.freeze([...receipts]), faultIndirects: Object.freeze([...faultIndirects]) });

const pass = (id: string, passName: string, dispatch: string, after: readonly string[],
  indirect?: string): SparseCM12ProductionPass => Object.freeze({ id, pass: passName,
  dispatch, after: Object.freeze([...after]), ...(indirect ? { indirect } : {}) });

const pressureBuffer = (plan: SparseCM12PhaseArenaPlan) => {
  const matches = plan.buffers.filter((entry) => entry.id === "cm12.pressure-cache");
  if (matches.length !== 1) throw new Error("manifest requires one cm12.pressure-cache");
  return matches[0]!;
};

const pressureMembership = (plan: SparseCM12PhaseArenaPlan) => {
  const buffer = pressureBuffer(plan);
  const matches = buffer.regions.filter((entry) => entry.name === "membership");
  if (matches.length !== 1) throw new Error("manifest requires PressureCache.membership");
  return { buffer, region: matches[0]! };
};

/**
 * Build the single serial cutover contract. All offsets are absolute u32-word
 * offsets in their named physical allocation. No field is runtime-selectable.
 */
export function createSparseCM12ProductionIntegrationManifest(
  input: SparseCM12ProductionIntegrationManifestInput,
): SparseCM12ProductionIntegrationManifest {
  const { pcmTombstone, fpl, fpp, scalarCandidate, fpa } = input.activity;
  const { phaseArena, pcm, pcf, psa } = input.pressure;
  const membership = pressureMembership(phaseArena);
  const membershipFirst = membership.region.offsetBytes / 4;
  const membershipEnd = membershipFirst + membership.region.sizeBytes / 4;

  const regions: SparseCM12ProductionRegion[] = [
    region("activity.pcm-tombstone", "activity", "cm12.activity", pcmTombstone.baseWords,
      pcmTombstone.totalWords, "tombstone"),
    region("activity.fpl1", "activity", "cm12.activity", fpl.baseWords, fpl.totalWords,
      "current"),
    region("activity.fpp1", "activity", "cm12.activity", fpp.baseWords, fpp.totalWords,
      "current"),
    region("activity.sca1-candidate", "activity", "cm12.activity",
      scalarCandidate.candidateBaseWords,
      scalarCandidate.candidateBaseWords + scalarCandidate.candidateWords, "candidate"),
    region("activity.sca1-accepted-mass", "activity", "cm12.activity",
      scalarCandidate.acceptedMassBaseWords, scalarCandidate.totalWords, "accepted"),
    region("activity.fpa1", "activity", "cm12.activity", fpa.baseWords, fpa.totalWords,
      "accepted"),
    region("topology.fca1", "topology", "cm12.topology-arena",
      input.topology.fca.baseWords, input.topology.fca.totalWords, "accepted"),
    region("scalar.saw1-control", "scalar-authority", "cm12.scalar-authority", 0,
      input.scalarAuthority.totalWords, "accepted"),
    region("scalar.sca1-copy", "scalar-authority", "cm12.scalar-authority",
      input.scalarAuthority.totalWords,
      input.scalarAuthority.totalWords + scalarCandidate.candidateWords, "candidate"),
    region("pressure.pcm1", "pressure-cache", "cm12.pressure-cache", pcm.baseWords,
      pcm.totalWords, "accepted"),
    region("pressure.pcf1-pca1", "pressure-cache", "cm12.pressure-cache",
      pcf.headerBaseWords, pcf.controlEndWords, "accepted"),
    region("pressure.psa1", "pressure-cache", "cm12.pressure-cache", psa.baseWords,
      psa.totalWords, "accepted"),
  ];
  if (pcmTombstone.baseWords > 0) regions.push(region("activity.option-b-prefix", "activity",
    "cm12.activity", 0, pcmTombstone.baseWords, "current"));
  if (input.topology.fca.baseWords > 0) regions.push(region("topology.mutable-prefix", "topology",
    "cm12.topology-arena", 0, input.topology.fca.baseWords, "current"));
  for (const value of membership.buffer.regions) {
    if (value.name === "membership" || value.sizeBytes === 0) continue;
    regions.push(region(`pressure.value.${value.name}`, "pressure-cache",
      "cm12.pressure-cache", value.offsetBytes / 4,
      (value.offsetBytes + value.sizeBytes) / 4,
      value.persistence === "immutable" ? "immutable" : "accepted"));
  }

  const bindingPhases = [
    ["transport-conditioning", [[0, "cm12.htp1", "read-only-storage"],
      [1, "cm12.physics-state", "storage"], [2, "cm12.scalar-scratch", "storage"],
      [8, "cm12.activity", "storage"], [9, "cm12.topology-arena", "storage"]]],
    ["pressure-preparation", [[0, "cm12.htp1", "read-only-storage"],
      [1, "cm12.physics-state", "storage"], [2, "cm12.scalar-scratch", "storage"],
      [3, "cm12.pressure-cache", "storage"], [8, "cm12.activity", "storage"],
      [9, "cm12.topology-arena", "storage"]]],
    ["pressure-solve", [[0, "cm12.htp1", "read-only-storage"],
      [1, "cm12.physics-state", "storage"], [2, "cm12.scalar-scratch", "storage"],
      [3, "cm12.pressure-cache", "storage"], [8, "cm12.activity", "storage"],
      [9, "cm12.topology-arena", "read-only-storage"]]],
    ["velocity-projection", [[0, "cm12.htp1", "read-only-storage"],
      [1, "cm12.physics-state", "storage"], [2, "cm12.scalar-scratch", "storage"],
      [3, "cm12.pressure-cache", "read-only-storage"],
      [8, "cm12.activity", "storage"], [9, "cm12.topology-arena", "storage"]]],
    ["topology-candidate", [[0, "cm12.htp1", "read-only-storage"],
      [1, "cm12.physics-state", "read-only-storage"],
      [3, "cm12.pressure-cache", "read-only-storage"],
      [4, "cm12.topology-candidate", "storage"], [8, "cm12.activity", "storage"],
      [9, "cm12.topology-arena", "storage"]]],
    ["presentation-publication", [[0, "cm12.htp1", "read-only-storage"],
      [1, "cm12.physics-state", "read-only-storage"],
      [4, "cm12.topology-candidate", "read-only-storage"],
      [5, "cm12.presentation-parameters", "uniform"],
      [6, "cm12.presentation", "storage"], [8, "cm12.activity", "storage"],
      [9, "cm12.topology-arena", "read-only-storage"]]],
    ["diagnostics-copy", [[1, "cm12.physics-state", "read-only-storage"],
      [2, "cm12.scalar-scratch", "read-only-storage"],
      [3, "cm12.pressure-cache", "read-only-storage"],
      [4, "cm12.topology-candidate", "read-only-storage"],
      [6, "cm12.presentation", "read-only-storage"],
      [7, "cm12.diagnostics-device", "storage"]]],
  ] as const;
  const bindings: SparseCM12ProductionBinding[] = bindingPhases.flatMap(([phase, entries]) =>
    entries.map(([binding, buffer, access]) => Object.freeze({ phase, binding, buffer, access })));

  const copies: SparseCM12ProductionCopySeam[] = [];
  copies.push(
    snapshot("fpl.current-packets", "cm12.activity", fpl.fixedIndirectBinding.offset,
      "cm12.fpl1-indirect", 0, "fpl-current-seal", ["frame-plan-consumers"],
      fpl.fixedIndirectBinding.size),
    snapshot("fpp.presentation", "cm12.activity", fpp.indirectBinding.offset,
      "cm12.fpp1-indirect", 0, "fpp-seal", ["presentation-publication"]),
    snapshot("fca.families", "cm12.topology-arena", 4 * input.topology.fca.indirectBaseWords,
      "cm12.fca1-indirect", 0, "fca-seal", ["frame-authority-consumers"],
      4 * SPARSE_CM12_FRAME_CONTROL_INDIRECT_WORDS * SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT),
  );
  const appendSnapshots = (buffer: string, values: readonly {
    id: string; source: number; sealedBy: string; consumers: readonly string[] }[]) => {
    values.forEach((value, index) => copies.push(snapshot(value.id, "cm12.activity",
      value.source, buffer, 12 * index, value.sealedBy, value.consumers)));
  };
  appendSnapshots("cm12.fpa1-indirect", [
    { id: "fpa.preparation.bootstrap", source: 4 * (fpa.preparation.headerBaseWords
        + SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER.reserved0),
      sealedBy: "fpa-preparation-begin", consumers: ["fpa-preparation-bootstrap"] },
    { id: "fpa.preparation.repair", source: 4 * (fpa.preparation.headerBaseWords
        + SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER.repairIndirectX),
      sealedBy: "fpa-preparation-frontier-finalize", consumers: ["fpa-preparation-repair"] },
    { id: "fpa.preparation.work", source: 4 * (fpa.preparation.headerBaseWords
        + SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER.workIndirectX),
      sealedBy: "fpa-preparation-plan-finalize", consumers: ["face-preparation"] },
    { id: "fpa.projection.bootstrap", source: 4 * (fpa.projection.headerBaseWords
        + SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER.reserved0),
      sealedBy: "fpa-projection-begin", consumers: ["fpa-projection-bootstrap"] },
    { id: "fpa.projection.repair", source: 4 * (fpa.projection.headerBaseWords
        + SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER.repairIndirectX),
      sealedBy: "fpa-projection-frontier-finalize", consumers: ["fpa-projection-repair"] },
    { id: "fpa.projection.work", source: 4 * (fpa.projection.headerBaseWords
        + SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER.workIndirectX),
      sealedBy: "fpa-projection-plan-finalize", consumers: ["project-faces"] },
  ]);

  const pressureCopies: { id: string; source: number; sealedBy: string;
    consumers: readonly string[] }[] = [
    { id: "pcf.fine.repair", source: 4 * (pcf.headerBaseWords
        + SPARSE_CM12_PRESSURE_CACHE_HEADER.repairIndirectX),
      sealedBy: "pcf-fine-frontier-finalize", consumers: ["pcf-fine-repair"] },
  ];
  for (const family of ["brick", "aggregateEdge", "hierarchyNode", "hierarchyEdge"] as const) {
    const base = pcf.aggregateFamilies[family].headerBaseWords;
    pressureCopies.push(
      { id: `pca.${family}.seed`, source: 4 * (base
          + SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER.seedIndirectX),
        sealedBy: "pcf-begin", consumers: ["pca-previous-leaf-seed"] },
      { id: `pca.${family}.repair`, source: 4 * (base
          + SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER.repairIndirectX),
        sealedBy: family === "brick" || family === "aggregateEdge"
          ? "pcf-fine-finalize" : "pca-aggregate-execution-finalize",
        consumers: [family === "brick" || family === "aggregateEdge"
          ? "pca-aggregate-workset-repair" : "pca-hierarchy-workset-repair"] },
      { id: `pca.${family}.work`, source: 4 * (base
          + SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER.workIndirectX),
        sealedBy: family === "brick" || family === "aggregateEdge"
          ? "pca-aggregate-plan-finalize" : "pca-hierarchy-plan-finalize",
        consumers: [family === "brick" || family === "aggregateEdge"
          ? "pca-aggregate-numerical" : "pca-hierarchy-numerical"] },
    );
  }
  pressureCopies.forEach((value, index) => copies.push(snapshot(value.id,
    "cm12.pressure-cache", value.source, "cm12.pcf1-indirect", 12 * index,
    value.sealedBy, value.consumers)));

  const psaSources = [
    ["psa.bootstrap", 4 * (psa.baseWords
      + SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER.bootstrapIndirectX),
      "psa-begin", ["psa-bootstrap"]],
    ["psa.brick.repair", 4 * (psa.brick.headerBaseWords
      + SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER.repairIndirectX),
      "psa-brick-frontier-finalize", ["psa-brick-repair"]],
    ["psa.brick.work", 4 * (psa.brick.headerBaseWords
      + SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER.workIndirectX),
      "psa-finalize", ["pressure-solve"]],
    ["psa.node.repair", 4 * (psa.hierarchyNode.headerBaseWords
      + SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER.repairIndirectX),
      "psa-brick-repair-finalize", ["psa-node-repair"]],
    ["psa.node.work", 4 * (psa.hierarchyNode.headerBaseWords
      + SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER.workIndirectX),
      "psa-finalize", ["pressure-solve"]],
  ] as const;
  psaSources.forEach(([id, source, sealedBy, consumers], index) => copies.push(snapshot(id,
    "cm12.pressure-cache", source, "cm12.psa1-indirect", 12 * index, sealedBy, consumers)));

  const authorities = [
    authority("PCM1.pressure-cache-cutover",
      ["current scalar authority", "current topology generation", "activity PCM tombstone comparator"],
      ["GPU bootstrap on uninitialized phase", "local cell incidence to row closure thereafter"],
      ["cell/row membership and count-tree rank equality at one-way cutover",
        "expected=covered closure", "accepted PCM generation"],
      ["cell repair", "row repair", "all pressure-family work"]),
    authority("FPA1.preparation",
      ["FCA1 frame/parity/body/boundary", "SAW1 density/phase", "VEX1 velocity-validity",
        "HTP1 topology-cell blast", "source face/CFL policy"],
      ["cell incidence", "one row/cell ring", "previous active leaves"],
      ["expected=covered producers", "repaired leaves", "executed rows",
        "verified active leaves", "accepted generation tuple"],
      ["bootstrap", "repair", "work"]),
    authority("FPA1.projection",
      ["FPA1 prepared authority", "PCM1 row membership", "PCF1 theta/coefficient",
        "pressure endpoint bits", "force/solid words"],
      ["preparation work rows", "incident projection rows", "previous active leaves"],
      ["expected=covered producers", "repaired leaves", "executed rows",
        "verified active leaves", "accepted generation tuple"],
      ["bootstrap", "repair", "work"]),
    authority("PCF1.fine",
      ["PCM1 cell transition", "PCM1 row/theta transition", "topology row/cell event",
        "solid coefficient event"],
      ["incident directed edges", "source cell diagonal", "aggregate owner"],
      ["expected=covered events", "repaired leaves", "changed edges/diagonals",
        "fine generation seal"], ["fine repair", "all PCA families"]),
    authority("PCA1.aggregate-hierarchy",
      ["PCF1 changed edge", "PCF1 changed diagonal", "aggregate topology generation"],
      ["aggregate brick/edge owner", "every hierarchy parent", "hierarchy edge/internal owner",
        "previous active leaves"],
      ["seed/repair/work counts per family", "executed counts per family",
        "aggregate topology generation", "single PCF acceptance"],
      ["brick seed/repair/work", "aggregate-edge seed/repair/work",
        "hierarchy-node seed/repair/work", "hierarchy-edge seed/repair/work"]),
    authority("PSA1.execution-domain",
      ["PCM1 cell create/retire", "topology brick blast", "PCF1 accepted generation"],
      ["wet-brick leaf", "all hierarchy parents", "previous dirty leaves"],
      ["expected=covered producers", "brick repaired leaves", "node repaired leaves",
        "frame/topology/PCM/PCF tuple", "accepted wet/node counts"],
      ["bootstrap", "brick repair/work", "node repair/work", "both tail banks"]),
    authority("PSA1.converged-tail",
      ["exact pipelinedPressureActive gate", "PSA1 accepted generation"],
      ["alternate A/B publication bank"],
      ["tail published generation", "active bank", "encoded/executed counts"],
      ["cell", "wet-brick", "hierarchy-node", "scalar"]),
  ];

  const passes = [
    pass("fca-seal", "frame-control", "seal current FCA1 generation", []),
    pass("fca-copy", "copy-seam", "copy all FCA1 family triplets", ["fca-seal"]),
    pass("frame-authority-consumers", "frame-control-consumers",
      "consume fixed FCA1 indirect families", ["fca-copy"]),
    pass("fpl-current-seal", "frame-plan", "seal current FPL1 packets", []),
    pass("fpl-current-copy", "copy-seam", "copy all six FPL1 packet triplets",
      ["fpl-current-seal"]),
    pass("frame-plan-consumers", "frame-plan-consumers",
      "consume fixed FPL1 indirect packets", ["fpl-current-copy"]),
    pass("pcm-begin", "pressure-membership-0",
      "begin PressureCache PCM1 (GPU bootstrap x=full only while uninitialized)",
      ["frame-plan-consumers"]),
    pass("pcm-producers", "pressure-membership-0",
      "scalar/topology transitions and bounded incidence closure", ["pcm-begin"]),
    pass("pcm-repair", "pressure-membership-1",
      "stable-ID cell/row leaf repair and count-tree update", ["pcm-producers"]),
    pass("pcm-accept", "pressure-membership-1",
      "accept only complete PCM generation; first cutover compares tombstone authority",
      ["pcm-repair"]),
    pass("fpa-preparation-begin", "face-authority-plan-0", "beginSparseCM12FacePreparationAuthority",
      ["frame-authority-consumers", "frame-plan-consumers"]),
    pass("fpa-preparation-copy-bootstrap", "copy-seam", "copy fpa.preparation.bootstrap",
      ["fpa-preparation-begin"]),
    pass("fpa-preparation-bootstrap", "face-authority-plan-bootstrap",
      "seedSparseCM12FacePreparationBootstrap", ["fpa-preparation-copy-bootstrap"],
      "fpa.preparation.bootstrap"),
    pass("fpa-preparation-producers", "face-authority-plan-0", "producer-local marks",
      ["fpa-preparation-bootstrap"]),
    pass("fpa-preparation-frontier-finalize", "face-authority-plan-0",
      "finalizeSparseCM12FacePreparationFrontier", ["fpa-preparation-producers"]),
    pass("fpa-preparation-copy-repair", "copy-seam", "copy fpa.preparation.repair",
      ["fpa-preparation-frontier-finalize"]),
    pass("fpa-preparation-repair", "face-authority-plan-1", "repairSparseCM12FacePreparationLeaves",
      ["fpa-preparation-copy-repair"], "fpa.preparation.repair"),
    pass("fpa-preparation-plan-finalize", "face-authority-plan-1",
      "finalizeSparseCM12FacePreparationPlan", ["fpa-preparation-repair"]),
    pass("fpa-preparation-copy-work", "copy-seam", "copy fpa.preparation.work",
      ["fpa-preparation-plan-finalize"]),
    pass("face-preparation", "face-preparation", "prepareTransportFaces unchanged body",
      ["fpa-preparation-copy-work"], "fpa.preparation.work"),
    pass("fpa-preparation-accept", "face-preparation", "verify + finalize execution",
      ["face-preparation"]),
    pass("pcf-begin", "pressure-topology-0", "begin PCF/PCA transaction", ["pcm-accept"]),
    pass("pca-seed-copy", "copy-seam", "copy four PCA seed triplets", ["pcf-begin"]),
    pass("pca-previous-leaf-seed", "pressure-topology-seed",
      "seed four previous-active-leaf families", ["pca-seed-copy"], "pca.brick.seed"),
    pass("pcf-producers", "pressure-topology-0", "PCM/topology/solid producer events",
      ["pca-previous-leaf-seed"]),
    pass("pcf-fine-frontier-finalize", "pressure-topology-0", "finalize PCF fine frontier",
      ["pcf-producers"]),
    pass("pcf-fine-copy-repair", "copy-seam", "copy pcf.fine.repair",
      ["pcf-fine-frontier-finalize"]),
    pass("pcf-fine-repair", "pressure-topology-1", "repairPersistentPressureCache",
      ["pcf-fine-copy-repair"], "pcf.fine.repair"),
    pass("pcf-fine-finalize", "pressure-topology-1", "finalizePersistentPressureFineCache",
      ["pcf-fine-repair"]),
    pass("pca-aggregate-repair-copy", "copy-seam", "copy aggregate repair triplets",
      ["pcf-fine-finalize"]),
    pass("pca-aggregate-workset-repair", "pressure-topology-2",
      "repair aggregate brick and edge worksets", ["pca-aggregate-repair-copy"],
      "pca.brick.repair"),
    pass("pca-aggregate-plan-finalize", "pressure-topology-1",
      "finalizePersistentPressureAggregatePlan", ["pca-aggregate-workset-repair"]),
    pass("pca-aggregate-copy", "copy-seam", "copy aggregate repair/work triplets",
      ["pca-aggregate-plan-finalize"]),
    pass("pca-aggregate-numerical", "pressure-topology-2",
      "repair aggregate edges then brick diagonals", ["pca-aggregate-copy"],
      "pca.brick.work"),
    pass("pca-aggregate-execution-finalize", "pressure-topology-2",
      "finalizePersistentPressureAggregateExecution", ["pca-aggregate-numerical"]),
    pass("pca-hierarchy-repair-copy", "copy-seam", "copy hierarchy repair triplets",
      ["pca-aggregate-execution-finalize"]),
    pass("pca-hierarchy-workset-repair", "pressure-topology-3",
      "repair hierarchy node and edge worksets", ["pca-hierarchy-repair-copy"],
      "pca.hierarchyNode.repair"),
    pass("pca-hierarchy-plan-finalize", "pressure-topology-3",
      "finalizePersistentPressureHierarchyPlan", ["pca-hierarchy-workset-repair"]),
    pass("pca-hierarchy-copy", "copy-seam", "copy hierarchy work triplets",
      ["pca-hierarchy-plan-finalize"]),
    pass("pca-hierarchy-numerical", "pressure-topology-3",
      "repair hierarchy edges then diagonals", ["pca-hierarchy-copy"]),
    pass("pcf-accept", "pressure-topology-3", "finalizePersistentPressureCache",
      ["pca-hierarchy-numerical"]),
    pass("psa-begin", "pressure-execution-plan-0", "beginSparseCM12PressureSolveAuthority",
      ["pcf-accept"]),
    pass("psa-bootstrap-copy", "copy-seam", "copy psa.bootstrap", ["psa-begin"]),
    pass("psa-bootstrap", "pressure-execution-bootstrap",
      "seedSparseCM12PressureSolveBootstrap", ["psa-bootstrap-copy"], "psa.bootstrap"),
    pass("psa-producers", "pressure-execution-plan-0", "PCM/topology producer events",
      ["psa-bootstrap"]),
    pass("psa-brick-frontier-finalize", "pressure-execution-plan-0",
      "finalizeSparseCM12PressureBrickFrontier", ["psa-producers"]),
    pass("psa-brick-copy-repair", "copy-seam", "copy psa.brick.repair",
      ["psa-brick-frontier-finalize"]),
    pass("psa-brick-repair", "pressure-execution-plan-1", "repair wet-brick leaves",
      ["psa-brick-copy-repair"], "psa.brick.repair"),
    pass("psa-brick-repair-finalize", "pressure-execution-plan-1",
      "finalizeSparseCM12PressureBrickRepair", ["psa-brick-repair"]),
    pass("psa-node-copy-repair", "copy-seam", "copy psa.node.repair",
      ["psa-brick-repair-finalize"]),
    pass("psa-node-repair", "pressure-execution-plan-2", "repair hierarchy-node leaves",
      ["psa-node-copy-repair"], "psa.node.repair"),
    pass("psa-finalize", "pressure-execution-plan-2",
      "finalizeSparseCM12PressureSolveAuthority", ["psa-node-repair"]),
    pass("psa-work-copy", "copy-seam", "copy PSA brick/node work triplets",
      ["psa-finalize"]),
    pass("pressure-solve", "pressure-solve", "unchanged PCM/PSA rank-ordered solve",
      ["psa-work-copy"]),
    pass("fpa-projection-begin", "projection-authority-plan-0",
      "beginSparseCM12FaceProjectionAuthority", ["pressure-solve", "fpa-preparation-accept"]),
    pass("fpa-projection-copy-bootstrap", "copy-seam", "copy fpa.projection.bootstrap",
      ["fpa-projection-begin"]),
    pass("fpa-projection-bootstrap", "projection-authority-bootstrap",
      "seedSparseCM12FaceProjectionBootstrap", ["fpa-projection-copy-bootstrap"],
      "fpa.projection.bootstrap"),
    pass("fpa-projection-frontier-finalize", "projection-authority-plan-0",
      "producer marks + finalizeSparseCM12FaceProjectionFrontier", ["fpa-projection-bootstrap"]),
    pass("fpa-projection-copy-repair", "copy-seam", "copy fpa.projection.repair",
      ["fpa-projection-frontier-finalize"]),
    pass("fpa-projection-repair", "projection-authority-plan-1",
      "repairSparseCM12FaceProjectionLeaves", ["fpa-projection-copy-repair"],
      "fpa.projection.repair"),
    pass("fpa-projection-plan-finalize", "projection-authority-plan-1",
      "finalizeSparseCM12FaceProjectionPlan", ["fpa-projection-repair"]),
    pass("fpa-projection-copy-work", "copy-seam", "copy fpa.projection.work",
      ["fpa-projection-plan-finalize"]),
    pass("project-faces", "velocity-projection", "projectFaces unchanged body",
      ["fpa-projection-copy-work"], "fpa.projection.work"),
    pass("fpa-projection-accept", "velocity-projection", "verify + finalize execution",
      ["project-faces"]),
    pass("fpp-seal", "presentation-plan", "seal current FPP1 page packet",
      ["fpa-projection-accept"]),
    pass("fpp-copy", "copy-seam", "copy FPP1 presentation triplet", ["fpp-seal"]),
    pass("presentation-publication", "presentation-publication",
      "publish page-transactional FPP1 work", ["fpp-copy"]),
  ];

  const gates: SparseCM12ProductionSliceGate[] = [
    { slice: "FPA1-preparation", requiredStaticChecks: ["tsc", "eslint", "integrated Dawn B16/P16",
      "manifest --resident"], dam: ["two independent 5-step physical hashes equal",
      "60-step/2s local equals construction oracle", "zero FPA faults/coverage gaps"],
      ocean: ["24 measured frames", "face preparation strictly faster", "no full-row dispatch"] },
    { slice: "FPA1-projection", requiredStaticChecks: ["same as preparation"],
      dam: ["60-step physical/pressure authority bytes equal oracle", "zero FPA faults"],
      ocean: ["24 measured frames", "face preparation + projection p95 < 1 ms"] },
    { slice: "PCF1-fine", requiredStaticChecks: ["PCF CPU/Naga checker", "integrated Dawn B16/P16",
      "manifest --resident"], dam: ["5-step paired exact then 60-step paired exact",
      "edge/theta/diagonal/RHS bytes equal full oracle", "zero PCF faults"],
      ocean: ["24 measured frames", "quiescent fine repair is dirty-leaf proportional"] },
    { slice: "PCA1-aggregate-hierarchy", requiredStaticChecks: ["PCF/PCA CPU/Naga checker",
      "integrated Dawn B16/P16", "manifest --resident"],
      dam: ["aggregate/hierarchy coefficient and diagonal bytes equal full oracle through 2s",
        "first topology burst has complete ancestor receipts"],
      ocean: ["pressure topology p95 < 1 ms", "zero global aggregate/hierarchy bakes"] },
    { slice: "PSA1-domains", requiredStaticChecks: ["PSA CPU/Naga checker",
      "integrated Dawn B16/P16", "manifest --resident"],
      dam: ["wet-brick/node sets and physical fields equal full oracle through 2s",
        "zero PSA faults/coverage gaps"],
      ocean: ["brick/node work equals accepted counts", "no empty hierarchy dispatch"] },
    { slice: "PSA1-converged-tail", requiredStaticChecks: ["PSA tail A/B checker",
      "integrated Dawn B16/P16", "manifest --resident"],
      dam: ["encoded arithmetic/order and physical fields equal oracle through 2s",
        "journals/final residuals remain encoded"],
      ocean: ["encoded/executed tail receipts published", "all non-pressure p95 < 10 ms"] },
  ].map((gate) => Object.freeze({ ...gate,
    requiredStaticChecks: Object.freeze(gate.requiredStaticChecks), dam: Object.freeze(gate.dam),
    ocean: Object.freeze(gate.ocean) }));

  const result: SparseCM12ProductionIntegrationManifest = Object.freeze({
    version: 1, regions: Object.freeze(regions), bindings: Object.freeze(bindings),
    copies: Object.freeze(copies), authorities: Object.freeze(authorities),
    passes: Object.freeze(passes),
    orderingSafePassCoalescings: Object.freeze([
      "producer-local dirty mark with its physical producer invocation",
      "direct planner dispatches in one compute pass while preserving dispatch order",
      "all already-sealed disjoint indirect triplet copies in one copy seam",
      "PCF effective-edge repair and preparePressure per-cell body in original sequence",
      "PSA stable-rank translation with unchanged retained numerical invocation body",
    ]),
    forbiddenDispatchCoalescings: Object.freeze([
      "FPA preparation with FPA projection", "PCM cell classification with row/theta publication",
      "PCF fine repair with aggregate numerical repair",
      "PCA aggregate repair with hierarchy repair", "PSA wet-brick repair with node repair",
      "any Krylov reduction/update/gate boundary", "PSA tail publisher with gate-writing reduction",
      "projectFaces with collocation", "topology commit with presentation publication",
    ]),
    qa: Object.freeze({ constructionOnly: true, productionOracleEnabled: false,
      comparedAuthorities: Object.freeze(["density", "gamma", "cell velocity", "face banks",
        "prepared face authority", "pressure", "divergence", "PCM membership/theta",
        "PCF fine/aggregate/hierarchy coefficients and diagonals", "RHS",
        "PSA wet-brick/node membership"]) }),
    gates: Object.freeze(gates),
  });
  assertSparseCM12ProductionIntegrationManifest(result, input, {
    membershipFirst, membershipEnd, pressureBufferWords: membership.buffer.sizeBytes / 4,
  });
  return result;
}

const REQUIRED_AUTHORITIES = Object.freeze(["PCM1.pressure-cache-cutover", "FPA1.preparation",
  "FPA1.projection", "PCF1.fine", "PCA1.aggregate-hierarchy", "PSA1.execution-domain",
  "PSA1.converged-tail"]);

/** Fail if a region, producer/closure/receipt, indirect seam, or ordering edge is omitted. */
export function assertSparseCM12ProductionIntegrationManifest(
  manifest: SparseCM12ProductionIntegrationManifest,
  input: SparseCM12ProductionIntegrationManifestInput,
  derived?: { readonly membershipFirst: number; readonly membershipEnd: number;
    readonly pressureBufferWords: number },
): void {
  if (manifest.version !== SPARSE_CM12_PRODUCTION_INTEGRATION_MANIFEST_VERSION) {
    throw new Error("unsupported production integration manifest version");
  }
  const ids = new Set<string>();
  for (const entry of manifest.regions) {
    if (ids.has(entry.id)) throw new Error(`duplicate region ${entry.id}`);
    ids.add(entry.id);
    if (!Number.isSafeInteger(entry.firstWord) || !Number.isSafeInteger(entry.endWord)
      || entry.firstWord < 0 || entry.endWord <= entry.firstWord) {
      throw new Error(`invalid region ${entry.id}`);
    }
  }
  for (const buffer of new Set(manifest.regions.map((entry) => entry.buffer))) {
    const ranges = manifest.regions.filter((entry) => entry.buffer === buffer)
      .sort((a, b) => a.firstWord - b.firstWord);
    for (let index = 1; index < ranges.length; index += 1) {
      if (ranges[index]!.firstWord < ranges[index - 1]!.endWord) {
        throw new Error(`${ranges[index]!.id} overlaps ${ranges[index - 1]!.id}`);
      }
    }
  }
  const { activity, pressure } = input;
  if (activity.fpl.baseWords < activity.pcmTombstone.totalWords
    || activity.fpp.baseWords < activity.fpl.totalWords
    || activity.scalarCandidate.candidateBaseWords < activity.fpp.totalWords
    || activity.fpa.baseWords !== alignWords(activity.scalarCandidate.totalWords)) {
    throw new Error("activity option-B chain must be PCM tombstone -> FPL -> FPP -> SCA -> FPA");
  }
  const membership = pressureMembership(pressure.phaseArena);
  const membershipFirst = derived?.membershipFirst ?? membership.region.offsetBytes / 4;
  const membershipEnd = derived?.membershipEnd
    ?? membershipFirst + membership.region.sizeBytes / 4;
  if (pressure.pcm.baseWords !== alignWords(membershipFirst)
    || pressure.pcf.headerBaseWords !== alignWords(pressure.pcm.totalWords)
    || pressure.psa.baseWords !== alignWords(pressure.pcf.controlEndWords)
    || pressure.psa.totalWords > membershipEnd
    || pressure.pcf.bufferId !== "cm12.pressure-cache") {
    throw new Error("PressureCache membership chain must be PCM -> PCF/PCA -> PSA");
  }
  const pressureWords = derived?.pressureBufferWords ?? pressureBuffer(pressure.phaseArena).sizeBytes / 4;
  if (pressure.psa.totalWords > pressureWords) throw new Error("pressure authority exceeds arena");
  if (activity.fpa.qaFullOracle || pressure.pcf.qaFullOracle || pressure.psa.qaFullOracle) {
    throw new Error("production manifest cannot enable a construction QA oracle");
  }
  const authorityById = new Map(manifest.authorities.map((entry) => [entry.id, entry]));
  for (const id of REQUIRED_AUTHORITIES) {
    const entry = authorityById.get(id);
    if (!entry || entry.producers.length === 0 || entry.closures.length === 0
      || entry.receipts.length === 0 || entry.faultIndirects.length === 0) {
      throw new Error(`${id} omitted producer, closure, receipt, or fault-zero family`);
    }
  }
  const copyIds = new Set(manifest.copies.map((entry) => entry.id));
  const passIds = new Set(manifest.passes.map((entry) => entry.id));
  for (const entry of manifest.copies) {
    if (entry.sizeBytes < 12 || entry.sizeBytes % 12 !== 0 || (entry.sourceOffsetBytes & 3) !== 0
      || (entry.destinationOffsetBytes % 12) !== 0 || !passIds.has(entry.sealedBy)) {
      throw new Error(`invalid or unsealed indirect seam ${entry.id}`);
    }
    if (entry.consumedBy.length === 0 || entry.consumedBy.some((id) => !passIds.has(id))) {
      throw new Error(`indirect seam ${entry.id} has an unknown consumer`);
    }
  }
  for (const buffer of new Set(manifest.copies.map((entry) => entry.destinationBuffer))) {
    const ranges = manifest.copies.filter((entry) => entry.destinationBuffer === buffer)
      .sort((a, b) => a.destinationOffsetBytes - b.destinationOffsetBytes);
    for (let index = 1; index < ranges.length; index += 1) {
      const prior = ranges[index - 1]!, current = ranges[index]!;
      if (current.destinationOffsetBytes < prior.destinationOffsetBytes + prior.sizeBytes) {
        throw new Error(`indirect destination overlap ${prior.id}/${current.id}`);
      }
    }
  }
  for (const entry of manifest.passes) {
    for (const dependency of entry.after) {
      if (!passIds.has(dependency)) {
        throw new Error(`${entry.id} depends on missing ${dependency}`);
      }
    }
    if (entry.indirect && !copyIds.has(entry.indirect)) {
      throw new Error(`${entry.id} uses uncopied indirect ${entry.indirect}`);
    }
  }
  const state = new Map<string, 0 | 1 | 2>();
  const visit = (id: string): void => {
    const prior = state.get(id) ?? 0;
    if (prior === 1) throw new Error(`pass dependency cycle at ${id}`);
    if (prior === 2) return;
    state.set(id, 1);
    manifest.passes.find((entry) => entry.id === id)!.after
      .forEach(visit);
    state.set(id, 2);
  };
  manifest.passes.forEach((entry) => visit(entry.id));
  if (!manifest.qa.constructionOnly || manifest.qa.productionOracleEnabled) {
    throw new Error("QA full oracle must be construction-only and absent from production policy");
  }
  if (manifest.gates.length !== 6 || manifest.gates.some((gate) =>
    gate.requiredStaticChecks.length === 0 || gate.dam.length === 0 || gate.ocean.length === 0)) {
    throw new Error("every serial slice requires static, dam, and ocean gates");
  }
}

/** Tokens required only after resident cutover; standalone checks intentionally do not call this. */
export const SPARSE_CM12_PRODUCTION_RESIDENT_TOKEN_CONTRACT = Object.freeze({
  wgsl: Object.freeze([
    "createSparseCM12FaceProjectionAuthorityWGSL",
    "createSparseCM12PersistentPressureCacheWGSL",
    "createSparseCM12PressureSolveAuthorityWGSL",
    "fpaMarkPreparationRow", "fpaMarkProjectionRow", "fpaMarkTopologyCellBlast",
    "fpaPreparationRowInvocation", "fpaProjectionRowInvocation",
    "fpaPreparationComplete", "fpaProjectionComplete", "fpaStorePreparedAuthority",
    "pcfRecordCellMembershipEvent", "pcfStoreThetaAndRecord", "pcfRecordSolidRowEvent",
    "psaMarkPCMCellTransition", "psaMarkTopologyBrickBlast",
    "psaWetBrickInvocation", "psaActiveHierarchyNodeAddress",
  ]),
  host: Object.freeze([
    "createSparseCM12FaceProjectionAuthorityLayout",
    "createSparseCM12PersistentPressureCacheLayout",
    "createSparseCM12PressureSolveAuthorityLayout",
    "cm12.fpa1-indirect", "cm12.pcf1-indirect", "cm12.psa1-indirect",
    "seedSparseCM12PreviousFacePreparationLeaves",
    "seedSparseCM12PreviousFaceProjectionLeaves", "seedSparseCM12ProjectionFromPreparation",
    "finalizeSparseCM12FacePreparationExecution", "finalizeSparseCM12FaceProjectionExecution",
    "finalizePersistentPressureFineCache", "repairPersistentPressureAggregateEdges",
    "repairPersistentPressureBrickDiagonals", "repairPersistentPressureHierarchyEdges",
    "repairPersistentPressureHierarchyDiagonals", "finalizePersistentPressureCache",
    "seedPreviousPCFBrickLeaves", "seedPreviousPCFAggregateEdgeLeaves",
    "seedPreviousPCFHierarchyNodeLeaves", "seedPreviousPCFHierarchyEdgeLeaves",
    "finalizeSparseCM12PressureSolveAuthority",
  ]),
});
