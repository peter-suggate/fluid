import { portsMatch } from "../framework/ports";
import type { PublicationPort } from "../framework/ports";
import type { ActionDefinition, ActionSample, BehaviorPlugin, ConfigSchema, DocumentInspection, EditorMetadata, EntityKindPlugin, EntityRecord, InteractionDocument, InteractionPhase, InteractionPlan, JsonObject, PlannedBehavior, PreviewPrimitive, Transform, UnavailablePlugin } from "./contracts";
import { INTERACTION_PHASES } from "./contracts";

function requireValue(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function id(value: unknown): asserts value is string {
  requireValue(typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._:-]*$/.test(value), `Invalid ID: ${String(value)}`);
}
function version(value: unknown): asserts value is number {
  requireValue(Number.isSafeInteger(value) && (value as number) > 0, "Version must be a positive integer");
}
/** Reject non-JSON input before invoking any plugin code or copying records. */
function assertJson(value: unknown, ancestors = new Set<object>()): void {
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") { requireValue(Number.isFinite(value), "JSON numbers must be finite"); return; }
  requireValue(object(value) || Array.isArray(value), "Expected JSON data");
  requireValue(!ancestors.has(value), "Cyclic JSON data");
  requireValue(Array.isArray(value) || Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null, "Expected plain JSON object");
  ancestors.add(value);
  const keys = Reflect.ownKeys(value);
  if (Array.isArray(value)) requireValue(keys.length === value.length + 1, "Sparse or extended JSON array");
  for (const key of keys) {
    if (Array.isArray(value) && key === "length") continue;
    requireValue(typeof key === "string", "Symbol keys are not JSON data");
    if (Array.isArray(value)) requireValue(/^(0|[1-9][0-9]*)$/.test(key) && Number(key) < value.length, "Extended JSON array");
    const property = Object.getOwnPropertyDescriptor(value, key)!;
    requireValue(property.enumerable && "value" in property, "Expected enumerable JSON data property");
    assertJson(property.value, ancestors);
  }
  ancestors.delete(value);
}
function jsonObject(value: unknown): asserts value is JsonObject {
  requireValue(object(value), "Expected config object");
  assertJson(value);
}
function snapshot<T>(value: T): T {
  if (Array.isArray(value)) return Object.freeze(value.map(snapshot)) as T;
  if (object(value)) return Object.freeze(Object.fromEntries(Object.entries(value).map(([key, child]) => [key, snapshot(child)]))) as T;
  return value;
}
function unique<T>(values: readonly T[], key: (value: T) => string, label: string): Map<string, T> {
  const result = new Map<string, T>();
  for (const value of values) {
    const name = key(value); id(name);
    requireValue(!result.has(name), `Duplicate ${label}: ${name}`);
    result.set(name, value);
  }
  return result;
}
function schemaCheck(schema: ConfigSchema) {
  version(schema.version); jsonObject(schema.defaults);
  requireValue(typeof schema.validate === "function", "Missing schema validator");
  requireValue(schema.migrate === undefined || typeof schema.migrate === "function", "Invalid schema migration");
  const errors = schema.validate(snapshot(schema.defaults));
  requireValue(errors.length === 0, `Invalid schema defaults: ${errors.join(", ")}`);
}
function editorCheck(editor: EditorMetadata, schema: ConfigSchema) {
  requireValue(editor.label.trim() && editor.group.trim() && editor.icon.trim() && Number.isFinite(editor.order), "Invalid editor metadata");
  unique(editor.controls, control => control.id, "control");
  for (const control of editor.controls) {
    requireValue(control.label.trim(), "Missing control label");
    requireValue(["number", "choice", "toggle", "readout"].includes(control.kind), "Invalid control kind");
    requireValue(control.setting && Object.hasOwn(schema.defaults, control.setting), `Unknown config setting: ${control.setting}`);
    if (control.kind !== "readout") {
      requireValue(control.setting && control.update, `Editable control ${control.id} needs setting and update impact`);
    }
    const initial = schema.defaults[control.setting];
    if (control.kind === "number") requireValue(typeof initial === "number", "Number control needs numeric default");
    if (control.kind === "toggle") requireValue(typeof initial === "boolean", "Toggle control needs boolean default");
    if (control.kind === "choice") {
      requireValue(control.options?.length && new Set(control.options.map(option => option.value)).size === control.options.length, "Choice needs unique options");
      requireValue(control.options.some(option => option.value === initial), "Choice default must be an option");
    }
    if (control.update) requireValue(["live", "rebuild", "reset"].includes(control.update), "Invalid update impact");
    for (const number of [control.min, control.max, control.step]) requireValue(number === undefined || Number.isFinite(number), "Invalid control bounds");
    requireValue(control.min === undefined || control.max === undefined || control.min <= control.max, "Invalid control range");
    requireValue(control.step === undefined || control.step > 0, "Invalid control step");
  }
}
function portCheck(ports: readonly PublicationPort[]) {
  unique(ports, port => port.id, "port");
  for (const port of ports) requireValue(port.representation && ["device", "scene", "generation", "frame"].includes(port.lifetime), "Invalid publication port");
}
function transformCheck(value: unknown): asserts value is Transform {
  requireValue(object(value) && object(value.position_m) && object(value.orientation), "Invalid transform");
  const p = value.position_m, q = value.orientation;
  requireValue([p.x, p.y, p.z, q.x, q.y, q.z, q.w].every(n => typeof n === "number" && Number.isFinite(n)), "Non-finite transform");
  const length = Math.hypot(q.x as number, q.y as number, q.z as number, q.w as number);
  requireValue(Math.abs(length - 1) < 1e-5, "Orientation must be a unit quaternion");
}
function config(schema: ConfigSchema, value: JsonObject, from: number): JsonObject {
  version(from); jsonObject(value);
  requireValue(from <= schema.version, "Unsupported future config version");
  let next = snapshot(value);
  if (from < schema.version) {
    requireValue(schema.migrate, `Missing migration from ${from} to ${schema.version}`);
    next = schema.migrate(next, from);
  }
  jsonObject(next);
  const errors = schema.validate(snapshot(next));
  requireValue(errors.length === 0, `Invalid config: ${errors.join(", ")}`);
  return snapshot(next);
}
function actionValue(kind: ActionDefinition["kind"], value: unknown): boolean {
  const axis = (n: unknown) => typeof n === "number" && Number.isFinite(n) && n >= -1 && n <= 1;
  return kind === "button" ? typeof value === "boolean" : kind === "axis1" ? axis(value)
    : Array.isArray(value) && value.length === 2 && value.every(axis);
}
function previewCheck(preview: PreviewPrimitive) {
  const positive = (value: number) => Number.isFinite(value) && value > 0;
  if (preview.shape === "box") requireValue([preview.dimensions_m.x, preview.dimensions_m.y, preview.dimensions_m.z].every(positive), "Invalid box preview");
  else if (preview.shape === "sphere") requireValue(positive(preview.radius_m), "Invalid sphere preview");
  else requireValue(preview.shape === "capsule" && positive(preview.radius_m) && positive(preview.height_m) && preview.height_m >= 2 * preview.radius_m, "Invalid capsule preview");
}

export function createInteractionCatalog(input: {
  readonly entities: readonly EntityKindPlugin[];
  readonly behaviors: readonly BehaviorPlugin[];
}) {
  // Snapshot declarations so later caller mutation cannot bypass validation.
  const entities = snapshot(input.entities);
  const behaviors = snapshot(input.behaviors);
  const kinds = unique(entities, plugin => plugin.id, "entity kind");
  const types = unique(behaviors, plugin => plugin.id, "behavior type");
  for (const plugin of [...entities, ...behaviors]) {
    schemaCheck(plugin.schema); editorCheck(plugin.editor, plugin.schema);
    unique(plugin.requires ?? [], capability => capability, "required capability");
  }
  for (const plugin of entities) {
    requireValue(typeof plugin.compose === "function", "Missing entity recipe");
    requireValue(typeof plugin.preview === "function", "Missing entity preview");
    previewCheck(plugin.preview(plugin.schema.defaults));
    requireValue(["upright", "surface"].includes(plugin.placement.alignment) && ["reject", "allow"].includes(plugin.placement.overlap), "Invalid placement policy");
  }
  for (const plugin of behaviors) {
    unique(plugin.provides ?? [], capability => capability, "provided capability");
    unique(plugin.requiresHost ?? [], capability => capability, "host capability");
    unique(plugin.phases, phase => phase.phase, "phase");
    for (const step of plugin.phases) {
      requireValue(INTERACTION_PHASES.includes(step.phase), `Unknown phase ${step.phase}`);
      unique(step.after ?? [], capability => capability, "phase dependency");
    }
    unique(plugin.actions ?? [], action => action.id, "action");
    for (const action of plugin.actions ?? []) {
      requireValue(action.label.trim() && ["button", "axis1", "axis2"].includes(action.kind), "Invalid action");
      for (const binding of action.bindings) {
        if (binding.device === "pointer-motion") requireValue(action.kind === "axis2" && binding.scale.length === 2 && binding.scale.every(Number.isFinite), "Invalid pointer motion binding");
        else {
          requireValue(binding.device === "keyboard" ? binding.code.length > 0 : binding.device === "pointer-button" && [0, 1, 2].includes(binding.button), "Invalid action binding");
          requireValue(actionValue(action.kind, binding.value), "Binding value does not match action kind");
        }
      }
    }
    portCheck(plugin.inputs ?? []); portCheck(plugin.outputs ?? []);
  }

  function inspectDocument(raw: unknown): DocumentInspection {
    assertJson(raw);
    requireValue(object(raw) && raw.schemaVersion === 1 && Array.isArray(raw.entities) && Array.isArray(raw.connections), "Invalid interaction document schema");
    // Validate the structural envelope even for missing plugins.
    for (const entity of raw.entities) {
      requireValue(object(entity), "Invalid entity record");
      id(entity.id); id(entity.kind); version(entity.version); jsonObject(entity.config);
      requireValue(typeof entity.name === "string" && entity.name.trim(), "Missing entity name");
      transformCheck(entity.transform);
      requireValue(Array.isArray(entity.behaviors), "Missing behavior records");
      for (const behavior of entity.behaviors) {
        requireValue(object(behavior), "Invalid behavior record");
        id(behavior.id); id(behavior.type); version(behavior.version); jsonObject(behavior.config);
      }
      unique(entity.behaviors as { id: string }[], behavior => behavior.id, "behavior instance");
    }
    const document = raw as unknown as InteractionDocument;
    unique(document.entities, entity => entity.id, "entity");
    const unavailable: UnavailablePlugin[] = [];
    const migrated = document.entities.map(entity => {
      const kind = kinds.get(entity.kind);
      if (!kind || entity.version > kind.schema.version) unavailable.push({ entityId: entity.id, pluginId: entity.kind, reason: kind ? "Newer entity version" : "Missing entity plugin" });
      const resolved = kind && entity.version <= kind.schema.version;
      const migratedConfig = resolved ? config(kind.schema, entity.config, entity.version) : entity.config;
      if (resolved) previewCheck(kind.preview(migratedConfig));
      return {
        ...entity,
        ...(resolved ? { version: kind.schema.version, config: migratedConfig } : {}),
        behaviors: entity.behaviors.map(behavior => {
          const plugin = types.get(behavior.type);
          if (!plugin || behavior.version > plugin.schema.version) {
            unavailable.push({ entityId: entity.id, behaviorId: behavior.id, pluginId: behavior.type, reason: plugin ? "Newer behavior version" : "Missing behavior plugin" });
            return behavior;
          }
          return { ...behavior, version: plugin.schema.version, config: config(plugin.schema, behavior.config, behavior.version) };
        }),
      };
    });
    for (const connection of document.connections) {
      requireValue(object(connection), "Invalid connection"); id(connection.id);
      for (const endpoint of [connection.source, connection.target]) {
        requireValue(object(endpoint), "Invalid connection endpoint");
        id(endpoint.entity); id(endpoint.behavior); id(endpoint.port);
        const entity = migrated.find(value => value.id === endpoint.entity);
        requireValue(entity?.behaviors.some(value => value.id === endpoint.behavior), `Dangling connection ${connection.id}`);
      }
    }
    unique(document.connections, connection => connection.id, "connection");
    return snapshot({ document: { ...document, entities: migrated }, unavailable });
  }

  function planDocument(raw: unknown, hostCapabilities: readonly string[] = []): InteractionPlan {
    const inspected = inspectDocument(raw);
    requireValue(!inspected.unavailable.length, `Unavailable plugins: ${inspected.unavailable.map(item => item.pluginId).join(", ")}`);
    const document = inspected.document;
    const phases: Record<InteractionPhase, PlannedBehavior[]> = { intent: [], "before-physics": [], "after-physics": [] };
    const requiredHost = new Set<string>();
    const compare = (a: string, b: string) => a < b ? -1 : a > b ? 1 : 0;
    for (const entity of [...document.entities].sort((a, b) => compare(a.id, b.id))) {
      const owners = new Map<string, string>();
      for (const behavior of entity.behaviors) {
        const plugin = types.get(behavior.type)!;
        for (const capability of plugin.provides ?? []) {
          requireValue(!owners.has(capability), `Competing owners for ${entity.id}/${capability}`);
          owners.set(capability, behavior.id);
        }
        for (const capability of plugin.requiresHost ?? []) requiredHost.add(capability);
      }
      for (const plugin of [kinds.get(entity.kind)!, ...entity.behaviors.map(behavior => types.get(behavior.type)!)]) {
        for (const capability of plugin.requires ?? []) requireValue(owners.has(capability), `${entity.id}/${plugin.id} requires missing capability ${capability}`);
      }
      for (const phase of INTERACTION_PHASES) {
        const remaining = new Map(entity.behaviors.filter(behavior => types.get(behavior.type)!.phases.some(step => step.phase === phase)).map(behavior => [behavior.id, behavior]));
        const dependencies = new Map<string, string[]>();
        for (const [name, behavior] of remaining) {
          const step = types.get(behavior.type)!.phases.find(step => step.phase === phase)!;
          const after = (step.after ?? []).map(capability => {
            const owner = owners.get(capability);
            requireValue(owner && remaining.has(owner), `${entity.id}/${name}: missing ${phase} provider for ${capability}`);
            return owner;
          });
          dependencies.set(name, after);
        }
        while (remaining.size) {
          const ready = [...remaining.keys()].filter(name => dependencies.get(name)!.every(dep => !remaining.has(dep))).sort(compare);
          requireValue(ready.length, `Behavior dependency cycle in ${entity.id}/${phase}`);
          for (const name of ready) {
            const behavior = remaining.get(name)!;
            phases[phase].push({ entityId: entity.id, behaviorId: name, type: behavior.type });
            remaining.delete(name);
          }
        }
      }
    }
    for (const capability of requiredHost) requireValue(hostCapabilities.includes(capability), `Missing host capability: ${capability}`);
    for (const connection of document.connections) {
      const port = (endpoint: typeof connection.source, direction: "inputs" | "outputs") => {
        const entity = document.entities.find(entity => entity.id === endpoint.entity)!;
        const behavior = entity.behaviors.find(behavior => behavior.id === endpoint.behavior)!;
        return types.get(behavior.type)![direction]?.find(port => port.id === endpoint.port);
      };
      const output = port(connection.source, "outputs"), input = port(connection.target, "inputs");
      requireValue(output && input && portsMatch(output, input), `Incompatible connection ${connection.id}`);
    }
    return snapshot({ document, phases, requiredHostCapabilities: [...requiredHost].sort(compare) });
  }

  function createEntity(request: { readonly id: string; readonly kind: string; readonly name?: string; readonly transform: Transform; readonly config?: JsonObject }): EntityRecord {
    id(request.id); transformCheck(request.transform);
    const plugin = kinds.get(request.kind);
    requireValue(plugin, `Unknown entity kind: ${request.kind}`);
    const value = config(plugin.schema, request.config ?? plugin.schema.defaults, plugin.schema.version);
    const entity = { id: request.id, kind: plugin.id, name: request.name ?? plugin.editor.label, transform: request.transform, version: plugin.schema.version, config: value, behaviors: plugin.compose(value) };
    // Creation validates recipes independently of runtime host availability.
    const allHost = [...new Set(behaviors.flatMap(behavior => behavior.requiresHost ?? []))];
    return planDocument({ schemaVersion: 1, entities: [entity], connections: [] }, allHost).document.entities[0];
  }
  function validateAction(document: InteractionDocument, sample: ActionSample): void {
    requireValue(Number.isSafeInteger(sample.sequence) && sample.sequence >= 0, "Invalid input sequence");
    const entity = document.entities.find(entity => entity.id === sample.entityId);
    const behavior = entity?.behaviors.find(behavior => behavior.id === sample.behaviorId);
    const action = behavior && types.get(behavior.type)?.actions?.find(action => action.id === sample.actionId);
    requireValue(action, "Unknown input action target");
    requireValue(actionValue(action.kind, sample.value), "Input value does not match action kind");
    requireValue(sample.edge === undefined || (action.kind === "button" && (sample.edge === "pressed" || sample.edge === "released") && sample.value === (sample.edge === "pressed")), "Invalid button edge");
  }
  return Object.freeze({
    entities: Object.freeze([...entities].sort((a, b) => a.editor.order - b.editor.order || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))),
    behaviors,
    entityKind: (name: string) => kinds.get(name),
    behaviorType: (name: string) => types.get(name),
    createEntity, inspectDocument, planDocument, validateAction,
  });
}
export type InteractionCatalog = ReturnType<typeof createInteractionCatalog>;
