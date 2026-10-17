import { duplicateEntities } from "../../interaction/authoring";
import type { UpdateImpact } from "../../framework/composition";
import type { Vec3 } from "../../core/model";
import { add, length, scale, sub } from "../../core/math";
import type {
  ActionSample,
  BehaviorEvent,
  BehaviorRuntime,
  BehaviorRuntimeFactory,
  BodySnapshot,
  CommandEnvelope,
  CommandResult,
  EntityRecord,
  InteractionAvailability,
  InteractionCatalog,
  InteractionCommand,
  InteractionDocument,
  InteractionHost,
  InteractionPlan,
  InteractionView,
  JsonObject,
  PhysicsCommand,
  RuntimeActionOffer,
  Transform,
  WorldSnapshot,
} from "../../interaction";
import { INTERACTION_PHASES } from "../../interaction";
import { boundsOverlap, isUnitVector, previewBounds, rayPreviewDistance } from "./geometry";

/**
 * A local `InteractionHost` for UI development: the test double the handoff
 * asks the UI to build against, and nothing more.
 *
 * What it enforces is the host's side of the contract — stale and duplicate
 * envelopes are refused, every document change is validated through the
 * catalog before it lands, an authoring command is one history entry, a
 * runtime command never moves the revision, reset increments the generation,
 * and button edges are consumed once. What it does *not* have is a world:
 * there is no Box3D, no collision query and no fluid. Overlap is checked
 * between interactive entities' preview bounds only, characters walk on the
 * plane their spawn stands on, and dynamic bodies hold still. The readiness it
 * reports says so (`SIMULATED_HOST_NOTE`), so the UI can show it honestly.
 */

export const SIMULATED_HOST_NOTE = "Simulated host: no physics engine, collision queries or fluid coupling";

const FIXED_DT_S = 1 / 60;
const MAX_TICKS_PER_PUMP = 4;
const GRAVITY_M_S2 = 9.81;
const REACH_M = 2.5;
const OBSERVE_RANGE_M = 40;
const IMPACT_RANK: Readonly<Record<UpdateImpact, number>> = { live: 0, rebuild: 1, reset: 2 };
const CONTROL_CAPABILITY = "character-motor";

export interface SimulatedHostHistoryEntry { readonly revision: number; readonly label: string }

export interface SimulatedInteractionHost extends InteractionHost {
  /** Authoring transactions, oldest first. Demo-only, outside the host contract. */
  history(): readonly SimulatedHostHistoryEntry[];
  /** Replace the document with an imported one, as one history entry and a world reset. Demo-only. */
  load(raw: unknown): CommandResult;
  /** Advance fixed ticks by hand. Only meaningful with `autoTick: false`. */
  step(ticks?: number): void;
  dispose(): void;
}

interface BodyState {
  entityId: string;
  bodyId: string;
  bodyGeneration: number;
  authority: BodySnapshot["authority"];
  transform: Transform;
  linear: Vec3;
  /** Foot height a character cannot fall below: there is no ground query. */
  floor_m: number;
  grounded: boolean;
}

interface HeldAction { readonly sample: ActionSample; readonly delta: boolean }

function actionKey(sample: Pick<ActionSample, "entityId" | "behaviorId" | "actionId">): string {
  return `${sample.entityId}\u0000${sample.behaviorId}\u0000${sample.actionId}`;
}

function entityName(document: InteractionDocument, id: string): string {
  return document.entities.find((entity) => entity.id === id)?.name ?? id;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function createSimulatedInteractionHost(options: {
  readonly catalog: InteractionCatalog;
  readonly capabilities: readonly string[];
  readonly runtimes: readonly BehaviorRuntimeFactory[];
  readonly initial?: InteractionDocument;
  /** Run the fixed clock from timers. Off for scripted checks, which call `step`. */
  readonly autoTick?: boolean;
  /** How long a rebuild reports `preparing`, so the UI's gates can be seen working. */
  readonly prepare_ms?: number;
}): SimulatedInteractionHost {
  const { catalog, capabilities } = options;
  const factories = new Map(options.runtimes.map((factory) => [factory.type, factory]));
  const autoTick = options.autoTick ?? true;
  const prepare_ms = options.prepare_ms ?? 250;
  const listeners = new Set<() => void>();
  const seen = new Set<string>();
  const historyEntries: SimulatedHostHistoryEntry[] = [];

  let document: InteractionDocument = options.initial
    ? catalog.inspectDocument(options.initial).document
    : { schemaVersion: 1, entities: [], connections: [] };
  let documentRevision = 0;
  let generation = 1;
  let running = false;
  let preparingUntil = 0;
  let controlledEntityId: string | undefined;
  let aimRay: { origin: Vec3; direction: Vec3 } | undefined;
  let tick = 0;
  let time_s = 0;
  let lastSequence = -1;
  let plan: InteractionPlan | undefined;
  let blockers: string[] = [];
  let bodies = new Map<string, BodyState>();
  let runtimes = new Map<string, BehaviorRuntime>();
  let pendingEvents: BehaviorEvent[] = [];
  const heldActions = new Map<string, HeldAction>();
  let edgeQueue: ActionSample[] = [];
  let view: InteractionView;
  let timer: ReturnType<typeof setInterval> | undefined;
  let prepareTimer: ReturnType<typeof setTimeout> | undefined;
  let lastPump_ms = 0;
  let accumulated_s = 0;

  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

  function emit() {
    view = buildView();
    for (const listener of [...listeners]) listener();
  }

  function preparing(): boolean { return now() < preparingUntil; }

  function startPreparing() {
    if (prepare_ms <= 0) return;
    preparingUntil = now() + prepare_ms;
    clearTimeout(prepareTimer);
    prepareTimer = setTimeout(() => { prepareTimer = undefined; emit(); }, prepare_ms + 5);
  }

  /** Plan the document and gather every reason it cannot run. */
  function replan() {
    blockers = [];
    plan = undefined;
    const inspection = catalog.inspectDocument(document);
    for (const item of inspection.unavailable) {
      blockers.push(`${entityName(document, item.entityId)}: ${item.reason.toLowerCase()} ${item.pluginId}`);
    }
    if (blockers.length) return;
    try { plan = catalog.planDocument(document, capabilities); }
    catch (error) { blockers.push(message(error)); return; }
    for (const phase of INTERACTION_PHASES) {
      for (const item of plan.phases[phase]) {
        if (!factories.has(item.type)) blockers.push(`No runtime for ${item.type} (${entityName(document, item.entityId)})`);
      }
    }
    blockers = [...new Set(blockers)];
  }

  function authority(entity: EntityRecord): BodySnapshot["authority"] {
    const provided = entity.behaviors.flatMap((behavior) => catalog.behaviorType(behavior.type)?.provides ?? []);
    if (provided.includes(CONTROL_CAPABILITY)) return "character";
    const host = entity.behaviors.flatMap((behavior) => catalog.behaviorType(behavior.type)?.requiresHost ?? []);
    return host.includes("physics.rigid-bodies") ? "dynamic" : "kinematic";
  }

  function spawnBody(entity: EntityRecord, bodyGeneration: number): BodyState {
    const kind = catalog.entityKind(entity.kind)!;
    const bounds = previewBounds(kind.preview(entity.config), entity.transform);
    return {
      entityId: entity.id, bodyId: `${entity.id}#body`, bodyGeneration,
      authority: authority(entity), transform: entity.transform, linear: { x: 0, y: 0, z: 0 },
      floor_m: bounds.min.y, grounded: true,
    };
  }

  function disposeEntity(entityId: string) {
    for (const [key, runtime] of runtimes) {
      if (key.startsWith(`${entityId}\u0000`)) { runtime.dispose(); runtimes.delete(key); }
    }
    bodies.delete(entityId);
  }

  /** Rebuild one entity's world state from its authored record. */
  function rebuildEntity(entity: EntityRecord) {
    const previous = bodies.get(entity.id);
    disposeEntity(entity.id);
    if (!plan) return;
    bodies.set(entity.id, spawnBody(entity, (previous?.bodyGeneration ?? 0) + 1));
    for (const behavior of entity.behaviors) {
      const factory = factories.get(behavior.type);
      if (factory) runtimes.set(`${entity.id}\u0000${behavior.id}`, factory.create(entity, behavior));
    }
  }

  /** A new world: every runtime and body from the document, at tick 0. */
  function resetWorld() {
    for (const runtime of runtimes.values()) runtime.dispose();
    runtimes = new Map();
    bodies = new Map();
    pendingEvents = [];
    heldActions.clear();
    edgeQueue = [];
    lastSequence = -1;
    tick = 0; time_s = 0;
    controlledEntityId = undefined;
    running = false;
    replan();
    if (plan) for (const entity of document.entities) rebuildEntity(entity);
    startPreparing();
  }

  function readiness(): InteractionView["readiness"] {
    if (blockers.length) return { state: "unavailable", reasons: blockers };
    if (preparing()) return { state: "preparing", reasons: ["Rebuilding the simulated world"] };
    return { state: "ready", reasons: [] };
  }

  function controllable(entity: EntityRecord): boolean {
    return entity.behaviors.some((behavior) => catalog.behaviorType(behavior.type)?.provides?.includes(CONTROL_CAPABILITY));
  }

  function snapshot(): WorldSnapshot | undefined {
    if (!plan) return undefined;
    return {
      generation, tick, solidRevision: 0, time_s,
      bodies: document.entities.flatMap((entity) => {
        const body = bodies.get(entity.id);
        const kind = catalog.entityKind(entity.kind);
        if (!body || !kind) return [];
        return [{
          entityId: body.entityId, bodyId: body.bodyId, bodyGeneration: body.bodyGeneration, authority: body.authority,
          transform: body.transform, linearVelocity_m_s: body.linear, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
          shapeId: `${entity.kind}:preview`, bounds_m: previewBounds(kind.preview(entity.config), body.transform),
        }];
      }),
    };
  }

  /**
   * What the aim ray rests on and what it offers.
   *
   * Occlusion is between interactive entities only — the double cannot see the
   * scene's solids — and the offers are the button actions of entities nobody
   * controls, which is this double's policy, not the contract's.
   */
  function resolveTarget(world: WorldSnapshot | undefined): InteractionView["target"] {
    if (!aimRay || !world) return undefined;
    let best: { entity: EntityRecord; distance: number } | undefined;
    for (const entity of document.entities) {
      if (entity.id === controlledEntityId) continue;
      const kind = catalog.entityKind(entity.kind);
      const body = world.bodies.find((candidate) => candidate.entityId === entity.id);
      if (!kind || !body) continue;
      const distance = rayPreviewDistance(aimRay, kind.preview(entity.config), body.transform);
      if (distance !== undefined && (!best || distance < best.distance)) best = { entity, distance };
    }
    if (!best || controllable(best.entity)) return undefined;
    const hit = add(aimRay.origin, scale(aimRay.direction, best.distance));
    const controlled = controlledEntityId ? world.bodies.find((body) => body.entityId === controlledEntityId) : undefined;
    const reach = controlled ? length(sub(hit, controlled.transform.position_m)) : best.distance;
    const availability: InteractionAvailability = !running ? { available: false, reason: "Paused" }
      : controlled && reach > REACH_M ? { available: false, reason: "Walk closer" }
      : !controlled && reach > OBSERVE_RANGE_M ? { available: false, reason: "Out of range" }
      : { available: true };
    const actions: RuntimeActionOffer[] = best.entity.behaviors.flatMap((behavior) =>
      (catalog.behaviorType(behavior.type)?.actions ?? [])
        .filter((action) => action.kind === "button")
        .map((action) => ({ behaviorId: behavior.id, actionId: action.id, label: action.label, availability })));
    return actions.length ? { entityId: best.entity.id, actions } : undefined;
  }

  function buildView(): InteractionView {
    const ready = readiness();
    const world = snapshot();
    return {
      generation, documentRevision, document,
      readiness: ready,
      running,
      context: controlledEntityId ? "play" : running ? "observe" : "edit",
      controlledEntityId,
      controlCandidates: document.entities.map((entity) => ({
        entityId: entity.id,
        availability: !controllable(entity) ? { available: false, reason: "Nothing on it takes control" }
          : ready.state !== "ready" ? { available: false, reason: ready.reasons[0] ?? "Not ready" }
          : { available: true },
      })),
      target: resolveTarget(world),
      snapshot: world,
    };
  }

  // ---- the fixed clock ------------------------------------------------------

  function inputFrame(): ActionSample[] {
    const frame = [...[...heldActions.values()].map((held) => held.sample), ...edgeQueue];
    edgeQueue = [];
    // Deltas are a sample's worth of motion: consumed by this tick, never repeated by the next.
    for (const [key, held] of heldActions) if (held.delta) heldActions.delete(key);
    return frame;
  }

  function applyPhysics(commands: readonly PhysicsCommand[], dt_s: number) {
    const moves = new Map<string, Extract<PhysicsCommand, { kind: "character-move" }>>();
    for (const command of commands) {
      if (command.kind === "character-move") { moves.set(command.entityId, command); continue; }
      // Grabs and impulses need dynamics the double does not have; they are dropped, not faked.
      if (command.kind === "grab" || command.kind === "impulse") continue;
      const body = [...bodies.values()].find((candidate) => candidate.bodyId === command.bodyId);
      // A handle from before a rebuild names a body that no longer exists.
      if (!body || body.bodyGeneration !== command.bodyGeneration) continue;
      if (body.authority !== "dynamic") body.transform = command.transform;
    }
    for (const body of bodies.values()) {
      if (body.authority !== "character") continue;
      const move = moves.get(body.entityId);
      let vy = body.linear.y;
      if (move?.jump && body.grounded) { vy = move.velocity_m_s.y; body.grounded = false; }
      if (!body.grounded) vy -= GRAVITY_M_S2 * dt_s;
      const v = { x: move?.velocity_m_s.x ?? 0, y: vy, z: move?.velocity_m_s.z ?? 0 };
      let position = add(body.transform.position_m, scale(v, dt_s));
      const entity = document.entities.find((candidate) => candidate.id === body.entityId)!;
      const kind = catalog.entityKind(entity.kind)!;
      const bounds = previewBounds(kind.preview(entity.config), { ...body.transform, position_m: position });
      if (bounds.min.y <= body.floor_m) {
        position = { ...position, y: position.y + body.floor_m - bounds.min.y };
        v.y = 0; body.grounded = true;
      }
      body.transform = { ...body.transform, position_m: position };
      body.linear = v;
    }
  }

  function runTick() {
    if (!plan) return;
    const identity = { generation, tick, solidRevision: 0 };
    const input = { ...identity, actions: inputFrame() };
    const world = snapshot()!;
    const events = pendingEvents;
    pendingEvents = [];
    const commands: PhysicsCommand[] = [];
    const runPhase = (phase: typeof INTERACTION_PHASES[number]) => {
      for (const item of plan!.phases[phase]) {
        const entity = document.entities.find((candidate) => candidate.id === item.entityId)!;
        const behavior = entity.behaviors.find((candidate) => candidate.id === item.behaviorId)!;
        runtimes.get(`${entity.id}\u0000${behavior.id}`)?.tick(phase, {
          ...identity, dt_s: FIXED_DT_S, entity, behavior, input, snapshot: world,
          viewOrigin_m: aimRay?.origin, viewDirection: aimRay?.direction,
          events: events.filter((event) => event.target.entity === entity.id && event.target.behavior === behavior.id),
          enqueue: (command) => commands.push(command),
          emit: (portId, value) => {
            for (const connection of document.connections) {
              const source = connection.source;
              if (source.entity === entity.id && source.behavior === behavior.id && source.port === portId) {
                pendingEvents.push({ ...identity, connectionId: connection.id, source, target: connection.target, value });
              }
            }
          },
        });
      }
    };
    runPhase("intent");
    runPhase("before-physics");
    applyPhysics(commands, FIXED_DT_S);
    runPhase("after-physics");
    tick += 1;
    time_s += FIXED_DT_S;
  }

  function pump() {
    const at = now();
    if (!running || preparing() || !plan) { lastPump_ms = at; accumulated_s = 0; return; }
    accumulated_s += Math.min(0.25, (at - lastPump_ms) / 1000);
    lastPump_ms = at;
    let ticks = 0;
    while (accumulated_s >= FIXED_DT_S && ticks < MAX_TICKS_PER_PUMP) { runTick(); accumulated_s -= FIXED_DT_S; ticks += 1; }
    if (ticks >= MAX_TICKS_PER_PUMP) accumulated_s = 0;
    if (ticks > 0) emit();
  }

  function setRunning(next: boolean) {
    running = next;
    lastPump_ms = now(); accumulated_s = 0;
    if (!autoTick) return;
    if (running && !timer) timer = setInterval(pump, 1000 / 120);
    if (!running && timer) { clearInterval(timer); timer = undefined; }
  }

  // ---- authoring --------------------------------------------------------------

  function findEntity(id: string): EntityRecord | undefined {
    return document.entities.find((entity) => entity.id === id);
  }

  /** The strongest impact among the settings that changed; an undeclared change resets. */
  function configImpact(controls: readonly { setting: string; update?: UpdateImpact }[], before: JsonObject, after: JsonObject): UpdateImpact {
    let impact: UpdateImpact = "live";
    for (const key of new Set([...Object.keys(before), ...Object.keys(after)])) {
      if (JSON.stringify(before[key]) === JSON.stringify(after[key])) continue;
      const declared = controls.find((control) => control.setting === key && control.update)?.update ?? "reset";
      if (IMPACT_RANK[declared] > IMPACT_RANK[impact]) impact = declared;
    }
    return impact;
  }

  /** Validate a whole proposed document through the catalog, plus composition of what changed. */
  function validated(next: InteractionDocument, touched: readonly string[]): InteractionDocument {
    const inspected = catalog.inspectDocument(next).document;
    for (const id of touched) {
      const entity = inspected.entities.find((candidate) => candidate.id === id);
      if (!entity) continue;
      const unavailable = catalog.inspectDocument({ schemaVersion: 1, entities: [entity], connections: [] }).unavailable;
      if (unavailable.length) continue;
      // Composition (owners, requirements, phase order) checked against every capability the
      // catalog knows, so an edit is judged on its own terms rather than on this host's.
      catalog.planDocument({ schemaVersion: 1, entities: [entity], connections: [] }, capabilities);
    }
    return inspected;
  }

  function overlapReason(entity: EntityRecord): string | undefined {
    const kind = catalog.entityKind(entity.kind);
    if (!kind || kind.placement.overlap === "allow") return undefined;
    const mine = previewBounds(kind.preview(entity.config), entity.transform);
    for (const other of document.entities) {
      if (other.id === entity.id) continue;
      const otherKind = catalog.entityKind(other.kind);
      if (!otherKind || otherKind.placement.overlap === "allow") continue;
      if (boundsOverlap(mine, previewBounds(otherKind.preview(other.config), other.transform))) return `Overlaps ${other.name}`;
    }
    return undefined;
  }

  function commitDocument(next: InteractionDocument, label: string, impact: UpdateImpact, touched: readonly string[]): CommandResult {
    const before = new Map(document.entities.map((entity) => [entity.id, entity]));
    document = next;
    documentRevision += 1;
    historyEntries.push({ revision: documentRevision, label });
    if (impact === "reset") { resetWorld(); return { accepted: true, documentRevision, impact }; }
    replan();
    if (!plan) {
      for (const runtime of runtimes.values()) runtime.dispose();
      runtimes = new Map(); bodies = new Map();
      if (running) setRunning(false);
    } else if (impact === "rebuild" || bodies.size === 0) {
      for (const id of touched) {
        const entity = document.entities.find((candidate) => candidate.id === id);
        if (entity) rebuildEntity(entity); else disposeEntity(id);
      }
      if (bodies.size === 0 || touched.some((id) => !bodies.has(id) && findEntity(id))) {
        for (const entity of document.entities) if (!bodies.has(entity.id)) rebuildEntity(entity);
      }
      if (impact === "rebuild") startPreparing();
    }
    if (controlledEntityId && !findEntity(controlledEntityId)) controlledEntityId = undefined;
    return { accepted: true, documentRevision, impact };
  }

  const STRUCTURAL: ReadonlySet<InteractionCommand["kind"]> = new Set(["place", "remove", "duplicate", "import", "transform", "set-behaviors", "connect", "disconnect"]);

  function authoring(command: InteractionCommand): CommandResult {
    if (running && STRUCTURAL.has(command.kind)) return { accepted: false, reason: "Pause the simulation before changing the scene" };
    switch (command.kind) {
      case "import": {
        const next = catalog.inspectDocument(command.document).document;
        return commitDocument(next, "Imported interactive entities", "reset", next.entities.map(e => e.id));
      }
      case "duplicate": {
        const copied = duplicateEntities(document, command.entityIds, command.offset_m, command.externalConnections);
        const added = copied.entities.filter(e => !findEntity(e.id));
        const next = validated(copied, added.map(e => e.id));
        for (const entity of added) {
          const overlap = overlapReason(entity);
          if (overlap) return { accepted: false, reason: overlap };
        }
        return commitDocument(next, "Duplicated interactive entities", "reset", added.map(e => e.id));
      }
      case "place": {
        if (findEntity(command.entity.id)) return { accepted: false, reason: `ID ${command.entity.id} is already in use` };
        const next = validated({ ...document, entities: [...document.entities, command.entity] }, [command.entity.id]);
        const placed = next.entities.find((entity) => entity.id === command.entity.id)!;
        const overlap = overlapReason(placed);
        if (overlap) return { accepted: false, reason: overlap };
        return commitDocument(next, `Placed ${placed.name}`, "rebuild", [placed.id]);
      }
      case "remove": {
        const entity = findEntity(command.entityId);
        if (!entity) return { accepted: false, reason: "No such entity" };
        // Its connections go in the same transaction: a dangling one would fail validation.
        const next = validated({
          ...document,
          entities: document.entities.filter((candidate) => candidate.id !== entity.id),
          connections: document.connections.filter((connection) => connection.source.entity !== entity.id && connection.target.entity !== entity.id),
        }, []);
        return commitDocument(next, `Removed ${entity.name}`, "rebuild", [entity.id]);
      }
      case "transform": {
        const entity = findEntity(command.entityId);
        if (!entity) return { accepted: false, reason: "No such entity" };
        const moved = { ...entity, transform: command.transform };
        const next = validated({ ...document, entities: document.entities.map((candidate) => candidate.id === entity.id ? moved : candidate) }, [entity.id]);
        const overlap = overlapReason(next.entities.find((candidate) => candidate.id === entity.id)!);
        if (overlap) return { accepted: false, reason: overlap };
        return commitDocument(next, `Moved ${entity.name}`, "rebuild", [entity.id]);
      }
      case "configure": {
        const entity = findEntity(command.entityId);
        if (!entity) return { accepted: false, reason: "No such entity" };
        let next: EntityRecord, impact: UpdateImpact, label: string;
        if (command.behaviorId === undefined) {
          const kind = catalog.entityKind(entity.kind);
          if (!kind) return { accepted: false, reason: `${entity.kind} is unavailable; its settings are preserved but cannot be edited` };
          next = { ...entity, version: command.value.version, config: command.value.config };
          impact = configImpact(kind.editor.controls, entity.config, command.value.config);
          label = `Configured ${entity.name}`;
        } else {
          const behavior = entity.behaviors.find((candidate) => candidate.id === command.behaviorId);
          if (!behavior) return { accepted: false, reason: "No such behavior" };
          const plugin = catalog.behaviorType(behavior.type);
          if (!plugin) return { accepted: false, reason: `${behavior.type} is unavailable; its settings are preserved but cannot be edited` };
          next = { ...entity, behaviors: entity.behaviors.map((candidate) => candidate.id === behavior.id
            ? { ...candidate, version: command.value.version, config: command.value.config } : candidate) };
          impact = configImpact(plugin.editor.controls, behavior.config, command.value.config);
          label = `Configured ${entity.name} · ${plugin.editor.label}`;
        }
        const document_ = validated({ ...document, entities: document.entities.map((candidate) => candidate.id === entity.id ? next : candidate) }, [entity.id]);
        if (command.behaviorId === undefined) {
          const overlap = overlapReason(document_.entities.find((candidate) => candidate.id === entity.id)!);
          if (overlap) return { accepted: false, reason: overlap };
        }
        return commitDocument(document_, label, impact, [entity.id]);
      }
      case "set-behaviors": {
        const entity = findEntity(command.entityId);
        if (!entity) return { accepted: false, reason: "No such entity" };
        const removed = new Set(entity.behaviors.map((behavior) => behavior.id).filter((id) => !command.behaviors.some((behavior) => behavior.id === id)));
        const used = document.connections.filter((connection) =>
          (connection.source.entity === entity.id && removed.has(connection.source.behavior))
          || (connection.target.entity === entity.id && removed.has(connection.target.behavior)));
        if (used.length) return { accepted: false, reason: `Disconnect ${used.map((connection) => connection.id).join(", ")} first` };
        const next = validated({ ...document, entities: document.entities.map((candidate) => candidate.id === entity.id
          ? { ...entity, behaviors: command.behaviors } : candidate) }, [entity.id]);
        return commitDocument(next, `Changed ${entity.name}'s behaviors`, "rebuild", [entity.id]);
      }
      case "connect": {
        if (document.connections.some((connection) => connection.id === command.connection.id)) return { accepted: false, reason: "Connection ID in use" };
        const next = validated({ ...document, connections: [...document.connections, command.connection] }, []);
        const ends = [command.connection.source.entity, command.connection.target.entity];
        catalog.planDocument({ schemaVersion: 1, entities: next.entities.filter((entity) => ends.includes(entity.id)), connections: [command.connection] }, capabilities);
        return commitDocument(next, `Connected ${entityName(document, ends[0]!)} to ${entityName(document, ends[1]!)}`, "rebuild", []);
      }
      case "disconnect": {
        if (!document.connections.some((connection) => connection.id === command.connectionId)) return { accepted: false, reason: "No such connection" };
        const next = validated({ ...document, connections: document.connections.filter((connection) => connection.id !== command.connectionId) }, []);
        return commitDocument(next, "Removed a connection", "rebuild", []);
      }
      default:
        throw new Error(`Not an authoring command: ${command.kind}`);
    }
  }

  // ---- runtime commands -------------------------------------------------------

  function runtime(command: InteractionCommand): CommandResult {
    const unchanged = (impact: UpdateImpact = "live"): CommandResult => ({ accepted: true, documentRevision, impact });
    switch (command.kind) {
      case "possess": {
        const candidate = view.controlCandidates.find((item) => item.entityId === command.entityId);
        if (!candidate) return { accepted: false, reason: "No such entity" };
        if (!candidate.availability.available) return { accepted: false, reason: candidate.availability.reason };
        heldActions.clear(); edgeQueue = [];
        controlledEntityId = command.entityId;
        return unchanged();
      }
      case "release-control": {
        controlledEntityId = undefined;
        heldActions.clear(); edgeQueue = [];
        return unchanged();
      }
      case "aim": {
        if (!isUnitVector(command.direction)) return { accepted: false, reason: "Aim direction must be a unit vector" };
        aimRay = { origin: command.origin_m, direction: command.direction };
        return unchanged();
      }
      case "action": {
        const sample = command.action;
        try { catalog.validateAction(document, sample); }
        catch (error) { return { accepted: false, reason: message(error) }; }
        if (!running) return { accepted: false, reason: "Paused: input is suspended" };
        if (sample.sequence <= lastSequence) return { accepted: false, reason: "Out-of-order input sequence" };
        const key = actionKey(sample);
        const releasing = sample.edge === "released" || (sample.edge === undefined && heldActions.has(key));
        if (sample.entityId !== controlledEntityId && !releasing) {
          // Revalidated against the current target, not the one the prompt was drawn for.
          const offer = view.target?.entityId === sample.entityId
            ? view.target.actions.find((action) => action.behaviorId === sample.behaviorId && action.actionId === sample.actionId) : undefined;
          if (!offer) return { accepted: false, reason: "Not something you can reach" };
          if (!offer.availability.available) return { accepted: false, reason: offer.availability.reason };
        }
        lastSequence = sample.sequence;
        const definition = catalog.behaviorType(findEntity(sample.entityId)!.behaviors.find((behavior) => behavior.id === sample.behaviorId)!.type)!
          .actions!.find((action) => action.id === sample.actionId)!;
        const delta = definition.bindings.some((binding) => binding.device === "pointer-motion");
        if (sample.edge) edgeQueue.push(sample);
        const resting = sample.value === false || sample.value === 0
          || (Array.isArray(sample.value) && sample.value[0] === 0 && sample.value[1] === 0);
        const pending = heldActions.get(key);
        if (delta && pending?.delta && Array.isArray(pending.sample.value) && Array.isArray(sample.value)) {
          // Motion between ticks accumulates; the tick consumes the sum once.
          const sum: readonly [number, number] = [pending.sample.value[0] + sample.value[0], pending.sample.value[1] + sample.value[1]];
          heldActions.set(key, { sample: { ...sample, value: sum, edge: undefined }, delta });
        } else if (resting && !delta) heldActions.delete(key);
        else heldActions.set(key, { sample: { ...sample, edge: undefined }, delta });
        return unchanged();
      }
      case "transport": {
        if (command.action === "reset") { resetWorld(); generation += 1; return unchanged("reset"); }
        if (command.action === "pause") { setRunning(false); heldActions.clear(); edgeQueue = []; return unchanged(); }
        const ready = readiness();
        if (ready.state !== "ready") return { accepted: false, reason: ready.reasons[0] ?? "Not ready" };
        setRunning(true);
        return unchanged();
      }
      default:
        throw new Error(`Not a runtime command: ${command.kind}`);
    }
  }

  const RUNTIME: ReadonlySet<InteractionCommand["kind"]> = new Set(["possess", "release-control", "aim", "action", "transport"]);

  function admit(envelope: CommandEnvelope): CommandResult {
    if (seen.has(envelope.id)) return { accepted: false, reason: "Duplicate command" };
    seen.add(envelope.id);
    if (envelope.generation !== generation) return { accepted: false, reason: "The world was reset since this command was prepared" };
    if (envelope.documentRevision !== documentRevision) return { accepted: false, reason: "The scene changed since this command was prepared" };
    try {
      return RUNTIME.has(envelope.command.kind) ? runtime(envelope.command) : authoring(envelope.command);
    } catch (error) {
      return { accepted: false, reason: message(error) };
    }
  }

  resetWorld();
  generation = 1;
  view = buildView();

  return {
    getView: () => view,
    subscribe(listener) {
      listeners.add(listener);
      return () => { listeners.delete(listener); };
    },
    async dispatch(envelope) {
      const result = admit(envelope);
      if (result.accepted) emit();
      return result;
    },
    history: () => historyEntries.slice(),
    load(raw) {
      try {
        if (running) return { accepted: false, reason: "Pause the simulation before importing" };
        const inspected = catalog.inspectDocument(raw).document;
        document = inspected;
        documentRevision += 1;
        historyEntries.push({ revision: documentRevision, label: "Imported interactive entities" });
        resetWorld();
        generation += 1;
        emit();
        return { accepted: true, documentRevision, impact: "reset" };
      } catch (error) {
        return { accepted: false, reason: message(error) };
      }
    },
    step(ticks = 1) {
      if (!running || !plan) return;
      for (let i = 0; i < ticks; i += 1) runTick();
      emit();
    },
    dispose() {
      setRunning(false);
      clearTimeout(prepareTimer);
      for (const runtime of runtimes.values()) runtime.dispose();
      runtimes.clear();
      listeners.clear();
    },
  };
}
