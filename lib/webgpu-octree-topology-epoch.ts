/**
 * Fail-closed A/B topology publication.
 *
 * Candidate builders write only the inactive table and a compact report. The
 * report is validated at the tail of substep N. A separately encoded
 * `publishReadyCandidate` invocation is the only operation allowed to change
 * the active table at the beginning of substep N+1.
 */

import { isOctreeEpochNewer } from "./webgpu-octree-worksets";
import type { PassBroker } from "./webgpu-pass-broker";

export const OCTREE_TOPOLOGY_EPOCH_ERROR = Object.freeze({
  staleEpoch: 1 << 0,
  generationMismatch: 1 << 1,
  publisherRejected: 1 << 2,
  capacity: 1 << 3,
  supportClosure: 1 << 4,
  catalogVersion: 1 << 5,
  emptyPublication: 1 << 6,
  reciprocity: 1 << 7,
  grading: 1 << 8,
  coefficientPositivity: 1 << 9,
  hierarchy: 1 << 10,
  indirectDispatch: 1 << 11,
} as const);

export const OCTREE_TOPOLOGY_EPOCH_STATE_BYTES = 64;

export interface OctreeTopologyEpochGPUResources {
  readonly ownerArena: GPUBuffer;
  readonly ownerCandidate: GPUBuffer;
  readonly frontier: GPUBuffer;
  readonly descriptorCandidateControl: GPUBuffer;
  readonly topologyCandidateControl: GPUBuffer;
  readonly structuredCandidateControl: GPUBuffer;
  /** Last GPU-accepted structured publication; owns the retained row count. */
  readonly structuredAcceptedControl: GPUBuffer;
  readonly boundaryCandidateControl: GPUBuffer;
  readonly spgridCandidateControl: GPUBuffer;
  readonly candidateLeafHeaders: GPUBuffer;
  readonly acceptedLeafHeaders: GPUBuffer;
  readonly candidatePressure: GPUBuffer;
  readonly candidatePressureHistory?: GPUBuffer;
  readonly acceptedPressureHistory?: GPUBuffer;
  readonly pressureA: GPUBuffer;
  readonly pressureB: GPUBuffer;
  readonly rowCountControl: GPUBuffer;
}

export interface OctreeTopologyEpochGPUOptions {
  readonly rowCapacity: number;
  readonly slotCapacity: number;
  readonly catalogVersion: number;
  readonly carryPressureHistory?: boolean;
}

/**
 * One GPU-resident reduction over every inactive topology family. The kernel
 * either marks the complete tuple ready or poisons every component's existing
 * fail-closed commit gate. No accepted selector is touched by validation.
 */
export class WebGPUOctreeTopologyEpoch {
  readonly state: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly validatePipeline: GPUComputePipeline;
  private readonly commitGatePipeline: GPUComputePipeline;
  private readonly prepareCommitRowsPipeline: GPUComputePipeline;
  private readonly commitRowsPipeline: GPUComputePipeline;
  private readonly validateGroup: GPUBindGroup;
  private readonly commitGateGroup: GPUBindGroup;
  private readonly prepareCommitRowsGroup: GPUBindGroup;
  private readonly commitRowsGroup: GPUBindGroup;
  private readonly commitRowsDispatch: GPUBuffer;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, resources: OctreeTopologyEpochGPUResources,
    private readonly options: OctreeTopologyEpochGPUOptions) {
    for (const [label, value] of [["row capacity", options.rowCapacity],
      ["slot capacity", options.slotCapacity], ["catalog version", options.catalogVersion]] as const) {
      if (!Number.isSafeInteger(value) || value < 1 || value > 0xffff_ffff) {
        throw new RangeError(`Topology epoch ${label} must be a positive uint32`);
      }
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.state = device.createBuffer({ label: "Coupled octree topology epoch state",
      size: OCTREE_TOPOLOGY_EPOCH_STATE_BYTES, usage: storage });
    device.queue.writeBuffer(this.state, 0, new Uint32Array(16));
    this.params = device.createBuffer({ label: "Coupled octree topology epoch parameters", size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, new Uint32Array([
      this.options.catalogVersion, this.options.rowCapacity, this.options.slotCapacity,
      0, 0, 0, 0, 0,
    ]));
    if (options.carryPressureHistory
      && (!resources.candidatePressureHistory || !resources.acceptedPressureHistory)) {
      throw new Error("Topology epoch pressure-history carry requires both history buffers");
    }
    const module = device.createShaderModule({ label: "Coupled octree topology epoch reduction",
      code: octreeTopologyEpochHistoryShader(options.carryPressureHistory === true) });
    const make = (entryPoint: string) => device.createComputePipeline({ label: entryPoint,
      layout: "auto", compute: { module, entryPoint } });
    this.validatePipeline = make("validateInactiveTopologyEpoch");
    this.commitGatePipeline = make("beginReadyTopologyCommit");
    this.prepareCommitRowsPipeline = make("prepareCandidateRowCommitDispatch");
    this.commitRowsPipeline = make("commitCandidateRows");
    const buffers = [this.params, this.state, resources.ownerArena, resources.ownerCandidate,
      resources.frontier, resources.descriptorCandidateControl, resources.topologyCandidateControl,
      resources.structuredCandidateControl, resources.boundaryCandidateControl,
      resources.spgridCandidateControl];
    this.validateGroup = device.createBindGroup({ layout: this.validatePipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.commitGateGroup = device.createBindGroup({
      layout: this.commitGatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.state } },
        { binding: 3, resource: { buffer: resources.ownerCandidate } },
        { binding: 4, resource: { buffer: resources.frontier } },
        { binding: 5, resource: { buffer: resources.descriptorCandidateControl } },
        { binding: 6, resource: { buffer: resources.topologyCandidateControl } },
        { binding: 7, resource: { buffer: resources.structuredCandidateControl } },
        { binding: 8, resource: { buffer: resources.boundaryCandidateControl } },
        { binding: 9, resource: { buffer: resources.spgridCandidateControl } },
      ],
    });
    this.commitRowsDispatch = device.createBuffer({
      label: "Accepted topology row commit dispatch",
      size: 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.commitRowsDispatch, 0, new Uint32Array([0, 1, 1]));
    this.prepareCommitRowsGroup = device.createBindGroup({
      layout: this.prepareCommitRowsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.state } },
        { binding: 16, resource: { buffer: this.commitRowsDispatch } },
      ],
    });
    this.commitRowsGroup = device.createBindGroup({ layout: this.commitRowsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: { buffer: this.state } },
        { binding: 10, resource: { buffer: resources.candidateLeafHeaders } },
        { binding: 11, resource: { buffer: resources.acceptedLeafHeaders } },
        { binding: 12, resource: { buffer: resources.candidatePressure } },
        { binding: 13, resource: { buffer: resources.pressureA } },
        { binding: 14, resource: { buffer: resources.pressureB } },
        { binding: 15, resource: { buffer: resources.rowCountControl } },
        { binding: 17, resource: { buffer: resources.structuredAcceptedControl } },
        ...(options.carryPressureHistory ? [
          { binding: 18, resource: { buffer: resources.candidatePressureHistory! } },
          { binding: 19, resource: { buffer: resources.acceptedPressureHistory! } },
        ] : []),
      ] });
  }

  private validateExpectedGeneration(expectedGeneration: number): void {
    if (!Number.isSafeInteger(expectedGeneration) || expectedGeneration < 1
      || expectedGeneration > 0xffff_ffff) {
      throw new RangeError("Expected coupled topology generation must be a positive uint32");
    }
  }

  encodeCandidateValidation(broker: PassBroker, expectedGeneration: number): void {
    if (this.destroyed) throw new Error("Coupled topology epoch is destroyed");
    this.validateExpectedGeneration(expectedGeneration);
    const pass = broker.compute({ label: "Validate complete inactive topology epoch" });
    pass.setPipeline(this.validatePipeline); pass.setBindGroup(0, this.validateGroup); pass.dispatchWorkgroups(1);
    broker.fence("inactive topology epoch validation reduction complete");
  }

  /** Beginning of N+1. A mismatched caller poisons all pending commits before
   * any component-specific ready-commit dispatch is encoded. */
  encodeReadyCommitGate(broker: PassBroker, expectedGeneration: number): void {
    if (this.destroyed) throw new Error("Coupled topology epoch is destroyed");
    this.validateExpectedGeneration(expectedGeneration);
    const pass = broker.compute({ label: "Open coupled topology ready-commit gate" });
    pass.setPipeline(this.commitGatePipeline); pass.setBindGroup(0, this.commitGateGroup); pass.dispatchWorkgroups(1);
    broker.fence("coupled topology ready-commit gate resolved");
    const prepareRows = broker.compute({ label: "Prepare exact accepted topology row commit" });
    prepareRows.setPipeline(this.prepareCommitRowsPipeline);
    prepareRows.setBindGroup(0, this.prepareCommitRowsGroup);
    prepareRows.dispatchWorkgroups(1);
    broker.fence("accepted topology row commit dispatch prepared");
    const rows = broker.compute({ label: "Commit accepted topology row identities and pressure seed" });
    rows.setPipeline(this.commitRowsPipeline); rows.setBindGroup(0, this.commitRowsGroup);
    rows.dispatchWorkgroupsIndirect(this.commitRowsDispatch, 0);
    broker.fence("accepted topology rows published");
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.state.destroy(); this.params.destroy(); this.commitRowsDispatch.destroy();
  }
}

export const octreeTopologyEpochWGSL = /* wgsl */ `
struct P{catalogVersion:u32,rowCapacity:u32,slotCapacity:u32,pad0:u32,pad:vec4u}
struct Epoch{activeEpoch:u32,activeGeneration:u32,readyEpoch:u32,readyGeneration:u32,error:u32,ready:u32,rowCount:u32,topologyHash:u32,reserved:array<u32,8>}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<storage,read_write>epoch:Epoch;
@group(0)@binding(2)var<storage,read_write>owner:array<u32>;@group(0)@binding(3)var<storage,read_write>ownerCandidate:array<u32>;
@group(0)@binding(4)var<storage,read_write>frontier:array<u32>;@group(0)@binding(5)var<storage,read_write>descriptor:array<u32>;
@group(0)@binding(6)var<storage,read_write>topology:array<u32>;@group(0)@binding(7)var<storage,read_write>structured:array<u32>;
@group(0)@binding(8)var<storage,read_write>boundary:array<u32>;@group(0)@binding(9)var<storage,read_write>spgrid:array<u32>;
@group(0)@binding(10)var<storage,read_write>candidateHeaders:array<u32>;
@group(0)@binding(11)var<storage,read_write>acceptedHeaders:array<u32>;
@group(0)@binding(12)var<storage,read_write>candidatePressure:array<f32>;
@group(0)@binding(13)var<storage,read_write>pressureA:array<f32>;
@group(0)@binding(14)var<storage,read_write>pressureB:array<f32>;
@group(0)@binding(15)var<storage,read_write>rowCountControl:array<u32>;
@group(0)@binding(16)var<storage,read_write>commitRowsDispatch:array<u32>;
@group(0)@binding(17)var<storage,read>acceptedStructured:array<u32>;
const STALE:u32=${OCTREE_TOPOLOGY_EPOCH_ERROR.staleEpoch}u;const GENERATION:u32=${OCTREE_TOPOLOGY_EPOCH_ERROR.generationMismatch}u;
const REJECTED:u32=${OCTREE_TOPOLOGY_EPOCH_ERROR.publisherRejected}u;const CAPACITY:u32=${OCTREE_TOPOLOGY_EPOCH_ERROR.capacity}u;
const SUPPORT:u32=${OCTREE_TOPOLOGY_EPOCH_ERROR.supportClosure}u;const CATALOG:u32=${OCTREE_TOPOLOGY_EPOCH_ERROR.catalogVersion}u;
const EMPTY:u32=${OCTREE_TOPOLOGY_EPOCH_ERROR.emptyPublication}u;const RECIPROCITY:u32=${OCTREE_TOPOLOGY_EPOCH_ERROR.reciprocity}u;
const GRADING:u32=${OCTREE_TOPOLOGY_EPOCH_ERROR.grading}u;const COEFFICIENT:u32=${OCTREE_TOPOLOGY_EPOCH_ERROR.coefficientPositivity}u;
const HIERARCHY:u32=${OCTREE_TOPOLOGY_EPOCH_ERROR.hierarchy}u;const INDIRECT:u32=${OCTREE_TOPOLOGY_EPOCH_ERROR.indirectDispatch}u;
const OWNER_READY:u32=0x80000000u;const STRUCTURED_READY:u32=0x5356454cu;
// Reserved word 3 latches the stash to the FIRST poison of a candidate. Both
// validateInactiveTopologyEpoch and beginReadyTopologyCommit poison, and the
// second one used to stash the value the first had already written into
// structured word 0, so a rejection reported the epoch's own error word back
// to itself and the publisher's real failure code was lost.
fn poison(error:u32){if(epoch.reserved[3u]==0u){epoch.reserved[3u]=1u;
  epoch.reserved[1u]=structured[0u];epoch.reserved[2u]=structured[1u];}
 ownerCandidate[28u]=0u;frontier[6u]=0u;frontier[9u]=error;
 descriptor[2u]|=error;descriptor[4u]|=error;
 topology[2u]|=error;structured[0u]=error;boundary[0u]|=error;spgrid[8u]|=error;}
fn candidateCount()->u32{let selector=frontier[7u];return select(p.rowCapacity+1u,frontier[selector],selector<=1u);}
fn candidateErrors()->u32{var error=0u;let count=candidateCount();let generation=ownerCandidate[24u];
 if(epoch.ready!=0u){error|=STALE;}if(generation==0u){error|=GENERATION;}
 if(ownerCandidate[28u]!=OWNER_READY||ownerCandidate[23u]!=0u||frontier[6u]!=1u||frontier[9u]!=0u){error|=REJECTED;}
 if(frontier[8u]!=generation||descriptor[7u]!=generation||structured[4u]!=generation
   ||boundary[4u]!=generation||spgrid[10u]!=generation){error|=GENERATION;}
 if(count==0u){error|=EMPTY;}if(count>p.rowCapacity||descriptor[0u]>p.rowCapacity||structured[3u]>p.slotCapacity
   ||owner[1u]>owner[3u]){error|=CAPACITY;}
 if(owner[12u]!=0u||ownerCandidate[23u]!=0u){error|=SUPPORT;}
 if(descriptor[2u]!=0u||descriptor[4u]!=0u||descriptor[1u]!=count){error|=GRADING;}
 if(topology[0u]!=0u||topology[2u]!=0u||topology[3u]!=count){error|=CATALOG;}
 if(topology[4u]!=p.catalogVersion){error|=CATALOG;}
 if(structured[0u]!=STRUCTURED_READY||structured[2u]!=count){error|=RECIPROCITY;}
 if(boundary[0u]!=0u||boundary[2u]!=count||boundary[6u]!=boundary[4u]){error|=COEFFICIENT;}
 // Section 4.3's hierarchy may publish an exact zero-page delta when the
 // accepted SPGrid structure is unchanged.  That is valid GPU reuse: the
 // candidate validator proves every changed page, not every resident page.
 if(spgrid[8u]!=0u||spgrid[0u]==0u||spgrid[4u]!=spgrid[0u]
   ||spgrid[1u]==0u||spgrid[2u]!=spgrid[3u]||spgrid[2u]>spgrid[1u]){error|=HIERARCHY;}return error;}
@compute @workgroup_size(1)fn validateInactiveTopologyEpoch(){epoch.reserved[3u]=0u;let error=candidateErrors();let count=candidateCount();
 epoch.readyEpoch=epoch.activeEpoch+1u;epoch.readyGeneration=ownerCandidate[24u];epoch.error=error;epoch.rowCount=count;
 epoch.topologyHash=(count*0x9e3779b9u)^descriptor[1u]^(structured[3u]<<16u)^spgrid[1u];epoch.ready=select(1u,0u,error!=0u);
 if(error!=0u){poison(error);}}
@compute @workgroup_size(1)fn beginReadyTopologyCommit(){var error=epoch.error;
 epoch.reserved[0u]=0u;
 let age=epoch.readyGeneration-epoch.activeGeneration;
 if(epoch.ready==0u){error|=REJECTED;}if(epoch.readyGeneration==0u
   ||age==0u||age>=0x80000000u){error|=GENERATION;}
 if(error!=0u){epoch.error=error;epoch.ready=0u;poison(error);return;}epoch.activeEpoch=epoch.readyEpoch;
 epoch.activeGeneration=epoch.readyGeneration;epoch.ready=0u;epoch.reserved[0u]=1u;}
@compute @workgroup_size(1)fn prepareCandidateRowCommitDispatch(){
 // A rejected candidate still dispatches lane zero so commitCandidateRows can
 // restore the retained accepted row count. All candidate row copies remain
 // gated by the successful commit token.
 commitRowsDispatch[0u]=select(1u,(epoch.rowCount+63u)/64u,epoch.reserved[0u]==1u);
 commitRowsDispatch[1u]=1u;commitRowsDispatch[2u]=1u;
}
@compute @workgroup_size(64)fn commitCandidateRows(@builtin(global_invocation_id)gid:vec3u){
 let row=gid.x;let commit=epoch.reserved[0u]==1u;
 // A rejected candidate retains the accepted Section 4 row authority. Read
 // its row count directly: reserved words are rejection forensics and may be
 // overwritten repeatedly while a failed candidate is retried.
 if(row==0u){let retained=select(0u,acceptedStructured[2u],arrayLength(&acceptedStructured)>=6u
   &&acceptedStructured[0u]==0u&&acceptedStructured[3u]!=0u);
   rowCountControl[0u]=select(retained,epoch.rowCount,commit);}
 if(!commit||row>=epoch.rowCount){return;}
 let base=row*12u;for(var word=0u;word<12u;word+=1u){acceptedHeaders[base+word]=candidateHeaders[base+word];}
 let seed=candidatePressure[row];pressureA[row]=seed;pressureB[row]=seed;
}
`;

export function octreeTopologyEpochHistoryShader(enabled: boolean): string {
  if (!enabled) return octreeTopologyEpochWGSL;
  return octreeTopologyEpochWGSL
    .replace(
      "@group(0)@binding(17)var<storage,read>acceptedStructured:array<u32>;",
      "@group(0)@binding(17)var<storage,read>acceptedStructured:array<u32>;\n"
        + "@group(0)@binding(18)var<storage,read>candidatePressureHistory:array<f32>;\n"
        + "@group(0)@binding(19)var<storage,read_write>acceptedPressureHistory:array<f32>;",
    )
    .replace(
      "let seed=candidatePressure[row];pressureA[row]=seed;pressureB[row]=seed;",
      "let seed=candidatePressure[row];pressureA[row]=seed;pressureB[row]=seed;\n"
        + " acceptedPressureHistory[row]=candidatePressureHistory[row];",
    );
}

export interface OctreeTopologyCandidateReport {
  readonly epoch: number;
  readonly generation: number;
  readonly errorFlags: number;
  readonly count: number;
  readonly capacity: number;
  /** Maximum displacement/halo/grading support encoded by the candidate. */
  readonly encodedSupportCells: number;
  /** Support required by the command graph that will consume the candidate. */
  readonly requiredSupportCells: number;
  readonly topologyHash: number;
  readonly catalogVersion: number;
}

export interface OctreeTopologyEpochState {
  readonly activeEpoch: number;
  readonly activeTable: 0 | 1;
  readonly activeGeneration: number;
  readonly candidateEpoch: number;
  readonly candidateTable: 0 | 1;
  readonly candidateGeneration: number;
  readonly candidateCount: number;
  readonly candidateError: number;
  readonly flipReady: boolean;
  readonly topologyHash: number;
  readonly catalogVersion: number;
}

export interface OctreeTopologyCandidateDecision {
  readonly state: OctreeTopologyEpochState;
  readonly accepted: boolean;
  readonly error: number;
}

function u32(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError(`${label} must fit uint32`);
  }
  return value >>> 0;
}

/** CPU oracle for the GPU validation kernel. It never mutates the active side. */
export function validateOctreeTopologyCandidate(
  current: OctreeTopologyEpochState,
  candidate: OctreeTopologyCandidateReport,
  expectedGeneration: number,
  expectedCatalogVersion: number,
): OctreeTopologyCandidateDecision {
  const activeEpoch = u32(current.activeEpoch, "Active topology epoch");
  const epoch = u32(candidate.epoch, "Candidate topology epoch");
  const generation = u32(candidate.generation, "Candidate topology generation");
  const expected = u32(expectedGeneration, "Expected topology generation");
  const catalogVersion = u32(candidate.catalogVersion, "Candidate catalog version");
  const requiredCatalog = u32(expectedCatalogVersion, "Expected catalog version");
  const count = u32(candidate.count, "Candidate topology count");
  const capacity = u32(candidate.capacity, "Candidate topology capacity");
  let error = u32(candidate.errorFlags, "Candidate topology error flags") === 0
    ? 0
    : OCTREE_TOPOLOGY_EPOCH_ERROR.publisherRejected;
  if (!isOctreeEpochNewer(epoch, activeEpoch)) {
    error |= OCTREE_TOPOLOGY_EPOCH_ERROR.staleEpoch;
  }
  if (generation !== expected) {
    error |= OCTREE_TOPOLOGY_EPOCH_ERROR.generationMismatch;
  }
  if (count === 0) error |= OCTREE_TOPOLOGY_EPOCH_ERROR.emptyPublication;
  if (count > capacity) error |= OCTREE_TOPOLOGY_EPOCH_ERROR.capacity;
  if (u32(candidate.encodedSupportCells, "Encoded topology support")
    < u32(candidate.requiredSupportCells, "Required topology support")) {
    error |= OCTREE_TOPOLOGY_EPOCH_ERROR.supportClosure;
  }
  if (catalogVersion !== requiredCatalog) {
    error |= OCTREE_TOPOLOGY_EPOCH_ERROR.catalogVersion;
  }
  return {
    accepted: error === 0,
    error,
    state: {
      ...current,
      candidateEpoch: epoch,
      candidateTable: (current.activeTable ^ 1) as 0 | 1,
      candidateGeneration: generation,
      candidateCount: count,
      candidateError: error,
      flipReady: error === 0,
      topologyHash: u32(candidate.topologyHash, "Candidate topology hash"),
      catalogVersion,
    },
  };
}

/** CPU mirror of the next-substep flip. Rejected candidates retain authority. */
export function publishReadyOctreeTopology(
  state: OctreeTopologyEpochState,
): OctreeTopologyEpochState {
  if (!state.flipReady || state.candidateError !== 0) return state;
  return {
    ...state,
    activeEpoch: state.candidateEpoch,
    activeTable: state.candidateTable,
    activeGeneration: state.candidateGeneration,
    flipReady: false,
  };
}
