import type { ClothSolverSettings } from './types';
import { CLOTH_NEWTON_SCENE_DEFAULTS } from './newton-defaults';

/** The viewport runs validated optimizations by default; headless reference callers opt in explicitly. */
export function clothExecutionSettings(settings: ClothSolverSettings = {}, liveDefaults = false) {
  const enabled = settings.optimizations ?? liveDefaults;
  const broadphase = enabled ? settings.contactBroadphase ?? 'bvh' : 'grid';
  const requestedEdgeEvaluation = enabled ? settings.contactEdgeEvaluation ?? (broadphase === 'bvh' ? 'packet' : 'directed') : 'directed';
  const edgeEvaluation = (requestedEdgeEvaluation === 'staged' || requestedEdgeEvaluation === 'packet') && broadphase !== 'bvh' ? 'unique' : requestedEdgeEvaluation;
  const stagedEdges=edgeEvaluation === 'staged' || edgeEvaluation === 'packet';
  const vertexEvaluation=enabled && stagedEdges ? settings.contactVertexEvaluation ?? 'packet' : 'direct';
  const conflictReuse=enabled ? settings.contactConflictReuse ?? 'all' : 'off';
  const solidCandidateReuse=enabled && broadphase === 'bvh' && (settings.contactSolidCandidateReuse ?? true);
  const newton={ ...CLOTH_NEWTON_SCENE_DEFAULTS,
    // No inexact-solve forcing: forcingMax 0.5 let the cloth slip on the rotating sphere (spin 0.57 of 0.785 rad/s,
    // 46-60 mm from the full-tolerance pose within 20 steps from tick 120); at full PCG tolerance it holds 0.775.
    ...(enabled ? { contactLinearization:'factorized' as const } : {}), ...settings.newton };
  return {
    solidCandidateReuse,
    macroCacheCapacity:solidCandidateReuse ? 64 : undefined,
    contactEntryCapacity:enabled ? 256 : undefined,
    mandatoryRedoCap:enabled ? 64 : undefined,
    // Fail-safe memory: the rotating sphere's least shift ended in the fail-safe on 198 of 211 steps (ticks 20-231);
    // starting 7 of every 8 there cut the mean step 290 -> 248 ms with the same spin and fewer over-strained edges.
    failSafeMemory:enabled ? 7 : undefined,
    vertexEvaluation,
    vertexStage:vertexEvaluation !== 'direct',
    vertexStagePacket:vertexEvaluation === 'packet',
    lazyLineSearch:enabled && newton.trustRegion !== 'hybrid' && (settings.lazyLineSearch ?? true),
    conflictReuse,
    conflictWitnesses:conflictReuse !== 'off',
    solidWitnesses:conflictReuse === 'all',
    enabled,
    edgeEvaluation,
    uniqueEdgePairs: edgeEvaluation !== 'directed',
    bvhEdgePruning: broadphase === 'bvh' && (edgeEvaluation === 'pruned' || edgeEvaluation === 'staged' || edgeEvaluation === 'packet'),
    edgeStagePacket: edgeEvaluation === 'packet',
    edgeStageBlockSize: edgeEvaluation === 'packet' ? 4 as const : 1 as const,
    edgeStageCapacity: (edgeEvaluation === 'staged' || edgeEvaluation === 'packet') ? 1_000_000 : undefined,
    bvhLeafSize: enabled ? settings.contactBvhLeafSize ?? 4 : 4,
    broadphase,
    contactBroadphase: broadphase === 'bvh' ? 'bvh' as const : 'grid' as const,
    candidateCache: broadphase === 'cached-grid' ? { skin_m: 0.0005, facesPerQuery: 64 } : undefined,
    separationCulling: enabled && (settings.contactSeparationCulling ?? true),
    multigrid: enabled && (settings.multigrid ?? false),
    newton,
  };
}
