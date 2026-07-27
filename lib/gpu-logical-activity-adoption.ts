import {
  GPU_LOGICAL_ACTIVITY_UNKNOWN_U32,
  GPULogicalActivityRecorder,
  buildGPULogicalActivityShaderVariant,
  gpuLogicalActivitySubgroupCallWGSL,
  gpuLogicalActivityWGSL,
  gpuLogicalActivityWorkgroupCallWGSL,
  type GPULogicalActivityMode,
  type GPULogicalActivityRecorderReadResult,
  type GPULogicalActivityShaderConfig,
} from "./gpu-logical-activity";

/** Reserved globally for opt-in shader activity variants. Production has no such binding. */
export const GPU_LOGICAL_ACTIVITY_BIND_GROUP = 3;
export const GPU_LOGICAL_ACTIVITY_BINDING = 0;
/** Bounded logical-slot sample used by the initial matrix adoption. */
export const GPU_LOGICAL_ACTIVITY_DEFAULT_WORKGROUP_SAMPLE_LIMIT = 128;

export interface GPULogicalActivityPipelineSnapshot {
  /** Snapshot `shaderActivityEnabled`, rather than reading a live store later. */
  enabled: boolean;
  /** Snapshot `shaderGeneration`; enabled cache keys include it. */
  generation: number;
}

export interface GPULogicalActivityAdoptionOptions {
  moduleId: string;
  profile: GPULogicalActivityPipelineSnapshot;
  /** Storage-saturated shaders remain byte-identical and use pass timestamps only. */
  support?: "dedicated-group" | "timestamp-only";
  identity?: "workgroup" | "subgroup";
  subgroupsAlreadyEnabled?: boolean;
}

export interface GPULogicalActivityTaskDescription {
  /** Stable UI/data identity; independent of the numeric shader ABI ID. */
  readonly id: string;
  readonly label: string;
  readonly color?: string;
  readonly parentId?: string;
  /** Measured host timestamp phase used to place clockless shader progress. */
  readonly phaseId?: string;
  /** Numeric checkpoint roles are populated by modules that can bracket work honestly. */
  readonly checkpoints?: Readonly<{ enter: number; exit: number }>;
}

interface RegisteredTaskDescription {
  readonly identity: string;
  readonly descriptor: GPULogicalActivityTaskDescription;
}

function sameTaskDescription(
  left: GPULogicalActivityTaskDescription,
  right: GPULogicalActivityTaskDescription,
): boolean {
  return left.id === right.id
    && left.label === right.label
    && left.color === right.color
    && left.parentId === right.parentId
    && left.phaseId === right.phaseId
    && left.checkpoints?.enter === right.checkpoints?.enter
    && left.checkpoints?.exit === right.checkpoints?.exit;
}

/** Shader variants are generation-scoped, so stale compiled modules cannot
 * silently lend their labels to a later capture generation. */
const taskDescriptionsByGeneration = new Map<number, Map<number, RegisteredTaskDescription>>();

export function gpuLogicalActivityTaskDescriptions(
  generation: number,
): Readonly<Record<number, GPULogicalActivityTaskDescription>> {
  const descriptions = taskDescriptionsByGeneration.get(checkedGeneration(generation));
  if (!descriptions) return Object.freeze({});
  return Object.freeze(Object.fromEntries(
    [...descriptions.entries()].map(([taskId, registered]) => [taskId, registered.descriptor]),
  ));
}

export interface GPULogicalActivityWorkgroupSite {
  tick?: string;
  workgroupId?: string;
  localInvocationIndex?: string;
  workgroupLaneCount: number | string;
  /** Uniform WGSL predicate used to bound recording without changing dispatched work. */
  recordWhen?: string;
}

export interface GPULogicalActivitySubgroupSite {
  tick?: string;
  workgroupId?: string;
  subgroupId?: string;
  subgroupIdEvidence?: "measured" | "reconstructed";
  subgroupLane?: string;
  subgroupSize?: string;
  active?: string;
}

function checkedGeneration(generation: number): number {
  if (!Number.isInteger(generation) || generation < 0 || generation > 0xffffffff) {
    throw new RangeError("GPU logical activity shader generation must be a u32");
  }
  return generation;
}

function checkedName(value: string, kind: string): string {
  if (!value) throw new RangeError(`GPU logical activity ${kind} must be non-empty`);
  return value;
}

/** Stable FNV-1a over UTF-16 code-unit bytes, with the ABI's unknown value excluded. */
export function stableGPULogicalActivityId(...parts: readonly string[]): number {
  let hash = 0x811c9dc5;
  const text = parts.join("\0");
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index);
    hash = Math.imul(hash ^ (code & 0xff), 0x01000193) >>> 0;
    hash = Math.imul(hash ^ (code >>> 8), 0x01000193) >>> 0;
  }
  return hash === GPU_LOGICAL_ACTIVITY_UNKNOWN_U32 ? 0xfffffffe : hash;
}

const u32WGSL = (value: number) => `${value >>> 0}u`;
const expression = (value: number | string) => typeof value === "number" ? u32WGSL(value) : value;

interface GPULogicalActivityPipelineRegistration {
  readonly moduleId: string;
  readonly generation: number;
  readonly config: Readonly<GPULogicalActivityShaderConfig>;
}

/** Weak ownership means retired pipelines disappear without registry cleanup. */
const instrumentedComputePipelines = new WeakMap<GPUComputePipeline, GPULogicalActivityPipelineRegistration>();

/** Primarily useful for capture diagnostics and focused validation. */
export function gpuLogicalActivityPipelineRegistration(pipeline: GPUComputePipeline) {
  return instrumentedComputePipelines.get(pipeline);
}

export interface GPULogicalActivityBindingSessionOptions {
  /** Required for an owned, readable capture; omit when sharing a recorder. */
  capacity?: number;
  /** Required for an owned, readable capture; omit when sharing a recorder. */
  captureId?: number;
  label?: string;
  /**
   * Non-owned recorder used only to satisfy instrumented pipeline bindings.
   * Shared sessions never stage, read, or destroy this recorder.
   */
  sharedRecorder?: GPULogicalActivityRecorder;
}

/**
 * Frame-scoped compute-pass binding proxy.
 *
 * It composes with command-encoder proxies (including timestamp recorders) by
 * invoking `beginComputePass` on the supplied encoder and wrapping only the
 * returned pass. Explicit pipeline layouts remain an adoption outlier: group 3
 * must be appended to those layouts before their pipelines are registered.
 */
export class GPULogicalActivityBindingSession {
  readonly encoder: GPUCommandEncoder;
  readonly recorder?: GPULogicalActivityRecorder;
  private readonly ownsRecorder: boolean;
  private readonly bindGroups = new WeakMap<GPUComputePipeline, GPUBindGroup>();

  constructor(
    private readonly activity: GPULogicalActivityAdoptionContext,
    device: GPUDevice,
    encoder: GPUCommandEncoder,
    options: GPULogicalActivityBindingSessionOptions,
  ) {
    if (!activity.enabled) {
      this.recorder = undefined;
      this.ownsRecorder = false;
    } else if (options.sharedRecorder) {
      this.recorder = options.sharedRecorder;
      this.ownsRecorder = false;
    } else {
      if (options.capacity === undefined || options.captureId === undefined) {
        throw new Error("Owned GPU logical activity binding sessions require capacity and captureId");
      }
      this.recorder = activity.recorder(device, {
        capacity: options.capacity,
        captureId: options.captureId,
        label: options.label,
      });
      this.ownsRecorder = this.recorder !== undefined;
    }
    this.encoder = this.recorder ? this.wrapEncoder(encoder) : encoder;
  }

  private wrapEncoder(encoder: GPUCommandEncoder): GPUCommandEncoder {
    const wrapPass = (pass: GPUComputePassEncoder) => {
      let activityBindGroup: GPUBindGroup | undefined;
      const bindActivity = () => {
        if (activityBindGroup) {
          pass.setBindGroup(GPU_LOGICAL_ACTIVITY_BIND_GROUP, activityBindGroup);
        }
      };
      return new Proxy(pass, {
      get: (target, property) => {
        if (property === "setPipeline") {
          return (pipeline: GPUComputePipeline) => {
            target.setPipeline(pipeline);
            const registration = instrumentedComputePipelines.get(pipeline);
            activityBindGroup = undefined;
            if (!registration) return;
            if (registration.generation !== this.activity.generation) {
              throw new Error(
                `GPU logical activity pipeline generation ${registration.generation} used in capture generation ${this.activity.generation}`,
              );
            }
            let bindGroup = this.bindGroups.get(pipeline);
            if (!bindGroup) {
              bindGroup = this.recorder!.createBindGroupForPipeline(pipeline, registration.config);
              this.bindGroups.set(pipeline, bindGroup);
            }
            activityBindGroup = bindGroup;
          };
        }
        if (property === "dispatchWorkgroups") {
          return (workgroupCountX: GPUSize32, workgroupCountY?: GPUSize32, workgroupCountZ?: GPUSize32) => {
            bindActivity();
            // Preserve the caller's arity. Dawn's native binding does not
            // accept explicit `undefined` values for omitted WebIDL arguments.
            if (workgroupCountZ !== undefined) {
              target.dispatchWorkgroups(workgroupCountX, workgroupCountY!, workgroupCountZ);
            } else if (workgroupCountY !== undefined) {
              target.dispatchWorkgroups(workgroupCountX, workgroupCountY);
            } else {
              target.dispatchWorkgroups(workgroupCountX);
            }
          };
        }
        if (property === "dispatchWorkgroupsIndirect") {
          return (indirectBuffer: GPUBuffer, indirectOffset: GPUSize64) => {
            bindActivity();
            target.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
          };
        }
        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
      }) as GPUComputePassEncoder;
    };
    return new Proxy(encoder, {
      get(target, property) {
        if (property === "beginComputePass") {
          return (descriptor?: GPUComputePassDescriptor) => wrapPass(
            descriptor === undefined ? target.beginComputePass() : target.beginComputePass(descriptor),
          );
        }
        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    }) as GPUCommandEncoder;
  }

  /** Append readback to this session's encoder before its command buffer is finished. */
  finish(): void { if (this.ownsRecorder) this.recorder!.finish(this.encoder); }

  read(): Promise<GPULogicalActivityRecorderReadResult | undefined> {
    return this.ownsRecorder ? this.recorder!.read() : Promise.resolve(undefined);
  }

  destroy(): void { if (this.ownsRecorder) this.recorder!.destroy(); }
}

/**
 * Immutable, low-bloat facade for gradually adopting one shader module.
 *
 * Typical use:
 * `const activity = createGPULogicalActivityAdoptionContext({ moduleId, profile });`
 * `const variant = activity.module(templateWith(activity.workgroup(...)), baseKey);`
 *
 * `module` prefixes `preludeWGSL` exactly once. In disabled mode the prelude and
 * calls are empty, module source is byte-identical, and `recorder` returns
 * undefined without touching the device.
 */
export class GPULogicalActivityAdoptionContext {
  readonly moduleId: string;
  readonly generation: number;
  readonly requested: boolean;
  readonly support: "dedicated-group" | "timestamp-only";
  readonly mode: GPULogicalActivityMode;
  readonly config: Readonly<GPULogicalActivityShaderConfig>;
  readonly preludeWGSL: string;
  private readonly ids = new Map<number, string>();

  constructor(options: GPULogicalActivityAdoptionOptions) {
    this.moduleId = checkedName(options.moduleId, "module ID");
    this.generation = checkedGeneration(options.profile.generation);
    this.requested = options.profile.enabled;
    this.support = options.support ?? "dedicated-group";
    this.mode = this.requested && this.support === "dedicated-group"
      ? options.identity ?? "workgroup"
      : "disabled";
    this.config = Object.freeze({
      mode: this.mode,
      group: this.mode === "disabled" ? undefined : GPU_LOGICAL_ACTIVITY_BIND_GROUP,
      binding: this.mode === "disabled" ? undefined : GPU_LOGICAL_ACTIVITY_BINDING,
      subgroupsAlreadyEnabled: options.subgroupsAlreadyEnabled,
    });
    this.preludeWGSL = gpuLogicalActivityWGSL(this.config);
  }

  get enabled(): boolean { return this.mode !== "disabled"; }
  get timestampOnly(): boolean { return this.requested && this.support === "timestamp-only"; }

  private id(kind: "task" | "checkpoint", names: readonly string[]): number {
    names.forEach((name) => checkedName(name, kind));
    const identity = [kind, this.moduleId, ...names].join("\0");
    const id = stableGPULogicalActivityId(identity);
    const existing = this.ids.get(id);
    if (existing !== undefined && existing !== identity) {
      throw new Error(`GPU logical activity ID collision between ${existing} and ${identity}`);
    }
    this.ids.set(id, identity);
    return id;
  }

  taskId(task: string): number { return this.id("task", [task]); }
  checkpointId(task: string, checkpoint: string): number {
    return this.id("checkpoint", [task, checkpoint]);
  }

  /** Register presentation metadata next to the module that owns the shader.
   * Repeated solver construction is allowed only when the semantic descriptor
   * is identical; numeric ID or label drift fails before a capture is decoded. */
  describeTask(task: string, descriptor: GPULogicalActivityTaskDescription): number {
    checkedName(descriptor.id, "task descriptor ID");
    checkedName(descriptor.label, "task descriptor label");
    const taskId = this.taskId(task);
    const identity = `${this.moduleId}\0${task}`;
    let generation = taskDescriptionsByGeneration.get(this.generation);
    if (!generation) {
      generation = new Map();
      taskDescriptionsByGeneration.set(this.generation, generation);
    }
    const normalized = Object.freeze({
      ...descriptor,
      ...(descriptor.checkpoints
        ? { checkpoints: Object.freeze({ ...descriptor.checkpoints }) }
        : {}),
    });
    const existing = generation.get(taskId);
    if (existing && (existing.identity !== identity
      || !sameTaskDescription(existing.descriptor, normalized))) {
      throw new Error(`GPU logical activity task ${taskId} conflicts with ${existing.identity}`);
    }
    generation.set(taskId, { identity, descriptor: normalized });
    return taskId;
  }

  workgroup(task: string, checkpoint: string, site: GPULogicalActivityWorkgroupSite): string {
    const call = gpuLogicalActivityWorkgroupCallWGSL(this.config, {
      taskId: u32WGSL(this.taskId(task)),
      checkpointId: u32WGSL(this.checkpointId(task, checkpoint)),
      tick: site.tick,
      workgroupId: site.workgroupId ?? "workgroupId",
      localInvocationIndex: site.localInvocationIndex ?? "localInvocationIndex",
      workgroupLaneCount: expression(site.workgroupLaneCount),
    });
    return call && site.recordWhen ? `if (${site.recordWhen}) { ${call} }` : call;
  }

  subgroup(task: string, checkpoint: string, site: GPULogicalActivitySubgroupSite = {}): string {
    return gpuLogicalActivitySubgroupCallWGSL(this.config, {
      taskId: u32WGSL(this.taskId(task)),
      checkpointId: u32WGSL(this.checkpointId(task, checkpoint)),
      tick: site.tick,
      workgroupId: site.workgroupId ?? "workgroupId",
      subgroupId: site.subgroupId ?? "subgroupId",
      subgroupIdEvidence: site.subgroupIdEvidence ?? "measured",
      subgroupLane: site.subgroupLane ?? "subgroupLane",
      subgroupSize: site.subgroupSize ?? "subgroupSize",
      active: site.active ?? "true",
    });
  }

  module(source: string, baseKey = this.moduleId) {
    const generationKey = this.enabled
      ? `${baseKey}|activity-generation:${this.generation}`
      : `${baseKey}|production`;
    return buildGPULogicalActivityShaderVariant({ source, baseKey: generationKey, config: this.config });
  }

  recorder(device: GPUDevice, input: { capacity: number; captureId: number; label?: string }) {
    return GPULogicalActivityRecorder.create(device, {
      mode: this.mode,
      capacity: input.capacity,
      captureId: input.captureId,
      label: input.label ?? `${this.moduleId} logical activity`,
    });
  }

  /** Register after creation; auto-layout pipelines need no other adoption code. */
  registerPipeline<T extends GPUComputePipeline>(pipeline: T): T {
    if (!this.enabled) return pipeline;
    const existing = instrumentedComputePipelines.get(pipeline);
    if (existing && (existing.generation !== this.generation || existing.moduleId !== this.moduleId)) {
      throw new Error(`GPU logical activity pipeline is already registered to ${existing.moduleId} generation ${existing.generation}`);
    }
    instrumentedComputePipelines.set(pipeline, {
      moduleId: this.moduleId,
      generation: this.generation,
      config: this.config,
    });
    return pipeline;
  }

  bindingSession(
    device: GPUDevice,
    encoder: GPUCommandEncoder,
    options: GPULogicalActivityBindingSessionOptions,
  ): GPULogicalActivityBindingSession {
    return new GPULogicalActivityBindingSession(this, device, encoder, options);
  }
}

export function createGPULogicalActivityAdoptionContext(
  options: GPULogicalActivityAdoptionOptions,
): GPULogicalActivityAdoptionContext {
  return new GPULogicalActivityAdoptionContext(options);
}
