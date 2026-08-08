import type { GPUInitializationTask } from "./gpu-initialization";
import {
  OCTREE_OWNER_PAGE_VOXELS,
  octreeOwnerPageLookupWgsl,
  type OctreeOwnerPagePlan,
} from "./webgpu-octree-owner-pages";
import { PassBroker } from "./webgpu-pass-broker";

const WORKGROUP_SIZE = 256;
const CONTROL_WORDS = 12;
const PUBLISHED = 0x8000_0000;

export interface OctreeAdaptiveNodeSource {
  /** count, owner generation, validity, errors, capacity, indirect xyz, launch limit */
  readonly control: GPUBuffer;
  /** Dense-node linear indices, compacted exactly once each. */
  readonly nodes: GPUBuffer;
  readonly nodeCapacity: number;
  readonly dispatchOffsetBytes: number;
}

/**
 * Exact shared-node worklist of the accepted octree.
 *
 * A dense node invocation examines only the eight cells incident on that node.
 * It publishes itself when one of their resident owners has the node as a
 * corner. Because the invocation identity is the node identity, hanging-node
 * sharing and leaf duplicates disappear without a leaf hash table or a sort.
 */
export class WebGPUOctreeAdaptiveNodes {
  readonly source: OctreeAdaptiveNodeSource;
  readonly allocatedBytes: number;
  private readonly params: GPUBuffer;
  private readonly pipelines: Partial<Record<"resetAdaptiveNodes" |
    "publishAdaptiveNodes" | "finishAdaptiveNodes", GPUComputePipeline>> = {};
  private readonly bindGroups = new Map<GPUComputePipeline, GPUBindGroup>();
  private module?: GPUShaderModule;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly ownerArena: GPUBuffer,
    plan: OctreeOwnerPagePlan,
    maximumLeafSize: number,
  ) {
    const nodeCapacity = (plan.dimensions[0] + 1)
      * (plan.dimensions[1] + 1) * (plan.dimensions[2] + 1);
    const maximumBinding = Math.min(device.limits.maxStorageBufferBindingSize,
      device.limits.maxBufferSize);
    if (!Number.isSafeInteger(nodeCapacity) || nodeCapacity < 1
      || nodeCapacity * Uint32Array.BYTES_PER_ELEMENT > maximumBinding) {
      throw new RangeError("Adaptive octree node worklist exceeds the device storage binding limit");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const control = device.createBuffer({
      label: "Accepted octree adaptive-node control",
      size: CONTROL_WORDS * Uint32Array.BYTES_PER_ELEMENT,
      usage: storage | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    const nodes = device.createBuffer({
      label: "Accepted octree shared-node worklist",
      size: nodeCapacity * Uint32Array.BYTES_PER_ELEMENT,
      usage: storage,
    });
    this.params = device.createBuffer({
      label: "Accepted octree adaptive-node owner lookup parameters",
      size: 3 * 4 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.source = Object.freeze({
      control, nodes, nodeCapacity,
      dispatchOffsetBytes: 5 * Uint32Array.BYTES_PER_ELEMENT,
    });
    this.allocatedBytes = control.size + nodes.size + this.params.size;
    device.queue.writeBuffer(control, 8 * Uint32Array.BYTES_PER_ELEMENT,
      new Uint32Array([device.limits.maxComputeWorkgroupsPerDimension]));
    device.queue.writeBuffer(this.params, 0, new Uint32Array([
      ...plan.dimensions, maximumLeafSize,
      ...plan.brickDimensions, plan.logicalBrickCount,
      plan.ownerDirectoryOffsetWords, plan.ownerPagesOffsetWords,
      plan.capacity, OCTREE_OWNER_PAGE_VOXELS,
    ]));
  }

  private descriptor(entryPoint: "resetAdaptiveNodes" | "publishAdaptiveNodes" |
    "finishAdaptiveNodes"): GPUComputePipelineDescriptor {
    this.module ??= this.device.createShaderModule({
      label: "Accepted octree adaptive-node publication",
      code: adaptiveNodeWGSL,
    });
    return { label: `accepted octree ${entryPoint}`, layout: "auto",
      compute: { module: this.module, entryPoint } };
  }

  initializationTasks(): GPUInitializationTask[] {
    if (Object.keys(this.pipelines).length !== 0) return [];
    return (["resetAdaptiveNodes", "publishAdaptiveNodes", "finishAdaptiveNodes"] as const)
      .map((entryPoint) => ({
        id: `octree.adaptive-nodes.pipeline.${entryPoint}`,
        phase: "adaptive-topology" as const,
        label: `Compile accepted adaptive nodes · ${entryPoint}`,
        run: async () => { this.pipelines[entryPoint] =
          await this.device.createComputePipelineAsync(this.descriptor(entryPoint)); },
      }));
  }

  encode(broker: PassBroker): void {
    if (this.destroyed) throw new Error("Adaptive octree node worklist is destroyed");
    const dispatch = (entryPoint: "resetAdaptiveNodes" | "publishAdaptiveNodes" |
      "finishAdaptiveNodes", items: number) => {
      const pipeline = this.pipelines[entryPoint];
      if (!pipeline) throw new Error(`Adaptive-node pipeline ${entryPoint} is not initialized`);
      const group = this.bindGroups.get(pipeline) ?? this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          ...(entryPoint === "publishAdaptiveNodes"
            ? [{ binding: 0, resource: { buffer: this.params } }] : []),
          { binding: 1, resource: { buffer: this.ownerArena } },
          { binding: 2, resource: { buffer: this.source.control } },
          { binding: 3, resource: { buffer: this.source.nodes } },
        ],
      });
      if (!this.bindGroups.has(pipeline)) this.bindGroups.set(pipeline, group);
      const pass = broker.compute({ label: `Publish accepted adaptive nodes · ${entryPoint}` });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group);
      const groups = Math.ceil(Math.max(1, items) / WORKGROUP_SIZE);
      const width = Math.min(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
      pass.dispatchWorkgroups(width, Math.ceil(groups / width));
    };
    dispatch("resetAdaptiveNodes", 1);
    dispatch("publishAdaptiveNodes", this.source.nodeCapacity);
    dispatch("finishAdaptiveNodes", 1);
  }

  async readReceipt(): Promise<Readonly<{ count: number; generation: number;
    published: boolean; errors: number; capacity: number;
    dispatch: readonly [number, number, number] }>> {
    const readback = this.device.createBuffer({
      label: "Accepted octree adaptive-node receipt",
      size: CONTROL_WORDS * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder({ label: "Read accepted adaptive-node receipt" });
    encoder.copyBufferToBuffer(this.source.control, 0, readback, 0, readback.size);
    this.device.queue.submit([encoder.finish()]);
    try {
      await readback.mapAsync(GPUMapMode.READ);
      const words = Uint32Array.from(new Uint32Array(readback.getMappedRange()));
      return Object.freeze({
        count: words[0]!, generation: words[1]!, published: words[2] === PUBLISHED,
        errors: words[3]!, capacity: words[4]!,
        dispatch: Object.freeze([words[5]!, words[6]!, words[7]!] as const),
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true; this.bindGroups.clear();
    this.params.destroy(); this.source.control.destroy(); this.source.nodes.destroy();
  }
}

const adaptiveNodeWGSL = /* wgsl */ `
${octreeOwnerPageLookupWgsl}
@group(0) @binding(0) var<uniform> ownerPageLookupParams: OctreeOwnerPageLookupParams;
@group(0) @binding(1) var<storage, read> ownerPageArena: array<u32>;
@group(0) @binding(2) var<storage, read_write> nodeControl: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> adaptiveNodes: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn resetAdaptiveNodes(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x != 0u) { return; }
  atomicStore(&nodeControl[0], 0u);
  atomicStore(&nodeControl[1], ownerPageArena[7]);
  atomicStore(&nodeControl[2], 0u);
  atomicStore(&nodeControl[3], 0u);
  atomicStore(&nodeControl[4], arrayLength(&adaptiveNodes));
  atomicStore(&nodeControl[5], 0u);
  atomicStore(&nodeControl[6], 1u);
  atomicStore(&nodeControl[7], 1u);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn publishAdaptiveNodes(@builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) workgroups: vec3u) {
  let dimensions = ownerPageLookupParams.dimensionsMaximumLeaf.xyz;
  let nodeDimensions = dimensions + vec3u(1u);
  let nodeCount = nodeDimensions.x * nodeDimensions.y * nodeDimensions.z;
  let item = lid.x + ${WORKGROUP_SIZE}u * (wid.x + wid.y * workgroups.x);
  if (item >= nodeCount) { return; }
  let layer = nodeDimensions.x * nodeDimensions.y;
  let node = vec3u(item % nodeDimensions.x, (item / nodeDimensions.x) % nodeDimensions.y,
    item / layer);
  var isOwnerCorner = false;
  for (var corner = 0u; corner < 8u; corner += 1u) {
    let offset = vec3i(i32(corner & 1u), i32((corner >> 1u) & 1u),
      i32((corner >> 2u) & 1u));
    let cell = vec3i(node) - offset;
    if (any(cell < vec3i(0)) || any(vec3u(cell) >= dimensions)) { continue; }
    let owner = octreeOwnerPageLookup(cell);
    if ((owner.status & OWNER_PAGE_LOOKUP_INVALID) != 0u) {
      atomicAdd(&nodeControl[3], 1u);
      continue;
    }
    if ((owner.status & OWNER_PAGE_LOOKUP_MISSING) != 0u) { continue; }
    let high = owner.origin + vec3u(owner.size);
    let xCorner = node.x == owner.origin.x || node.x == high.x;
    let yCorner = node.y == owner.origin.y || node.y == high.y;
    let zCorner = node.z == owner.origin.z || node.z == high.z;
    if (xCorner && yCorner && zCorner) { isOwnerCorner = true; break; }
  }
  if (!isOwnerCorner) { return; }
  let slot = atomicAdd(&nodeControl[0], 1u);
  if (slot < arrayLength(&adaptiveNodes)) {
    adaptiveNodes[slot] = item;
  } else {
    atomicAdd(&nodeControl[3], 1u);
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn finishAdaptiveNodes(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x != 0u) { return; }
  let count = atomicLoad(&nodeControl[0]);
  let generation = atomicLoad(&nodeControl[1]);
  let stable = generation == ownerPageArena[7];
  let valid = stable && atomicLoad(&nodeControl[3]) == 0u
    && count <= arrayLength(&adaptiveNodes);
  atomicStore(&nodeControl[2], select(0u, ${PUBLISHED}u, valid));
  atomicStore(&nodeControl[4], arrayLength(&adaptiveNodes));
  let groups = (count + ${WORKGROUP_SIZE - 1}u) / ${WORKGROUP_SIZE}u;
  let maximumWidth = max(1u, atomicLoad(&nodeControl[8]));
  let width = min(groups, maximumWidth);
  var height = 0u;
  if (width > 0u) { height = (groups + width - 1u) / width; }
  atomicStore(&nodeControl[5], width);
  atomicStore(&nodeControl[6], height);
  atomicStore(&nodeControl[7], 1u);
}
`;
