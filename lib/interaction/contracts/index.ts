import type { FeatureControl, UpdateImpact } from "../../framework/composition";
import type { PublicationPort } from "../../framework/ports";
import type { Quaternion, Vec3 } from "../../core/model";

export type JsonValue = null | boolean | number | string | readonly JsonValue[] | JsonObject;
export interface JsonObject { readonly [key: string]: JsonValue }
export interface Transform { readonly position_m: Vec3; readonly orientation: Quaternion }
export interface VersionedConfig { readonly version: number; readonly config: JsonObject }
export interface BehaviorRecord extends VersionedConfig { readonly id: string; readonly type: string }
export interface EntityRecord extends VersionedConfig {
  readonly id: string;
  readonly kind: string;
  readonly name: string;
  readonly transform: Transform;
  /** Complete authored composition, initially supplied by the kind's recipe. */
  readonly behaviors: readonly BehaviorRecord[];
}
export interface Endpoint { readonly entity: string; readonly behavior: string; readonly port: string }
export interface Connection { readonly id: string; readonly source: Endpoint; readonly target: Endpoint }
/** Embeddable document fragment. SceneDescription integration is a separate adapter. */
export interface InteractionDocument {
  readonly schemaVersion: 1;
  readonly entities: readonly EntityRecord[];
  readonly connections: readonly Connection[];
  /** Climb assists follow authored ladders; endpoints are foot positions. */
  readonly ladders?: readonly { readonly id: string; readonly bottom_m: Vec3; readonly top_m: Vec3; readonly width_m: number }[];
}
export interface ConfigSchema {
  readonly version: number;
  readonly defaults: JsonObject;
  /** Return field errors; never coerce invalid data silently. */
  validate(config: JsonObject): readonly string[];
  /** Return current-version config. Called only for older versions. */
  migrate?(config: JsonObject, fromVersion: number): JsonObject;
}
/** Runtime action buttons come from actions; controls edit/read configuration only. */
export type InteractionControl = Omit<FeatureControl, "kind"> & {
  readonly kind: "number" | "choice" | "toggle" | "readout";
  readonly setting: string;
};
export interface EditorMetadata {
  readonly label: string;
  readonly hint: string;
  readonly icon: string;
  readonly group: string;
  readonly order: number;
  /** setting names a top-level config key; update declares its runtime impact. */
  readonly controls: readonly InteractionControl[];
}
export type ActionValue = boolean | number | readonly [number, number];
export type InputBinding =
  | { readonly device: "keyboard"; readonly code: string; readonly value: ActionValue }
  | { readonly device: "pointer-button"; readonly button: 0 | 1 | 2; readonly value: ActionValue }
  | { readonly device: "pointer-motion"; readonly scale: readonly [number, number] };
export interface ActionDefinition {
  readonly id: string;
  readonly label: string;
  readonly kind: "button" | "axis1" | "axis2";
  /** Held key contributions sum and clamp per axis; button actions combine with OR. */
  readonly bindings: readonly InputBinding[];
}
export type PreviewPrimitive =
  | { readonly shape: "box"; readonly dimensions_m: Vec3 }
  | { readonly shape: "sphere"; readonly radius_m: number }
  | { readonly shape: "capsule"; readonly radius_m: number; readonly height_m: number };
export interface EntityKindPlugin {
  readonly id: string;
  readonly schema: ConfigSchema;
  readonly editor: EditorMetadata;
  readonly placement: {
    readonly alignment: "upright" | "surface";
    readonly overlap: "reject" | "allow";
  };
  readonly requires?: readonly string[];
  /** Local-space, origin-centered preview. Capsule height includes both hemispheres. */
  preview(config: JsonObject): PreviewPrimitive;
  /** Pure recipe, called on creation only. Saved behaviors remain authoritative. */
  compose(config: JsonObject): readonly BehaviorRecord[];
}
export const INTERACTION_PHASES = ["intent", "before-physics", "after-physics"] as const;
export type InteractionPhase = typeof INTERACTION_PHASES[number];
export interface BehaviorPhase {
  readonly phase: InteractionPhase;
  /** Entity-local capabilities whose behavior must run first in this phase. */
  readonly after?: readonly string[];
}
/** Payloads cross the runtime boundary as JSON; representation identifies their schema. */
export type InteractionPort = Omit<PublicationPort, "payload">;
export interface BehaviorPlugin {
  readonly id: string;
  readonly schema: ConfigSchema;
  readonly editor: EditorMetadata;
  /** Exclusive entity-local ownership, e.g. transform or character-motor. */
  readonly provides?: readonly string[];
  readonly requires?: readonly string[];
  readonly requiresHost?: readonly string[];
  readonly phases: readonly BehaviorPhase[];
  readonly actions?: readonly ActionDefinition[];
  /** Event ports reuse framework representation, lifetime and unit matching. */
  readonly inputs?: readonly InteractionPort[];
  readonly outputs?: readonly InteractionPort[];
}
export interface PlannedBehavior {
  readonly entityId: string;
  readonly behaviorId: string;
  readonly type: string;
}
export interface InteractionPlan {
  readonly document: InteractionDocument;
  readonly phases: Readonly<Record<InteractionPhase, readonly PlannedBehavior[]>>;
  readonly requiredHostCapabilities: readonly string[];
}
export interface UnavailablePlugin { readonly entityId: string; readonly behaviorId?: string; readonly pluginId: string; readonly reason: string }
export interface DocumentInspection {
  /** Unknown/newer plugin records are retained; execution is blocked. */
  readonly document: InteractionDocument;
  readonly unavailable: readonly UnavailablePlugin[];
}
export interface TickIdentity { readonly generation: number; readonly tick: number; readonly solidRevision: number }
export interface ActionSample {
  readonly entityId: string;
  readonly behaviorId: string;
  readonly actionId: string;
  readonly value: ActionValue;
  readonly edge?: "pressed" | "released";
  readonly sequence: number;
}
export interface InputFrame extends TickIdentity { readonly actions: readonly ActionSample[] }
export interface BodySnapshot {
  readonly entityId: string;
  readonly bodyId: string;
  readonly bodyGeneration: number;
  readonly authority: "dynamic" | "kinematic" | "character" | "static";
  readonly transform: Transform;
  readonly linearVelocity_m_s: Vec3;
  readonly angularVelocity_rad_s: Vec3;
  readonly shapeId: string;
  readonly grounded?: boolean;
  readonly submergedFraction?: number;
  readonly bounds_m: { readonly min: Vec3; readonly max: Vec3 };
}
export interface WorldSnapshot extends TickIdentity {
  readonly time_s: number;
  readonly bodies: readonly BodySnapshot[];
}
export interface FluidExchange extends TickIdentity {
  readonly bodyId: string;
  readonly bodyGeneration: number;
  readonly linearImpulse_N_s: Vec3;
  readonly angularImpulse_N_m_s: Vec3;
  readonly displacedVolume_m3: number;
  readonly fluidVelocity_m_s: Vec3;
  readonly policy: "pressure-traction" | "approximate-buoyancy-drag";
}
export type PhysicsCommand =
  | { readonly kind: "grab"; readonly entityId: string; readonly active: boolean; readonly target_m: Vec3 }
  | { readonly kind: "impulse"; readonly bodyId: string; readonly bodyGeneration: number; readonly linear_N_s: Vec3; readonly angular_N_m_s: Vec3 }
  | { readonly kind: "kinematic-target"; readonly bodyId: string; readonly bodyGeneration: number; readonly transform: Transform }
  /** With `turn_rad_s` the host turns the body's facing at that rate and `velocity_m_s` is in the body's own frame (+z ahead); without it the velocity is world-space and the body turns toward it. */
  | { readonly kind: "character-move"; readonly entityId: string; readonly velocity_m_s: Vec3; readonly jump: boolean; readonly climb?: boolean; readonly somersault?: boolean; readonly turn_rad_s?: number };
export interface BehaviorEvent extends TickIdentity {
  readonly connectionId: string;
  readonly source: Endpoint;
  readonly target: Endpoint;
  readonly value: JsonValue;
}
export interface BehaviorTickContext extends TickIdentity {
  readonly dt_s: number;
  readonly viewDirection?: Vec3;
  readonly viewOrigin_m?: Vec3;
  readonly entity: EntityRecord;
  readonly behavior: BehaviorRecord;
  readonly input: InputFrame;
  readonly snapshot: WorldSnapshot;
  /** Events emitted on the previous tick, copied before producer storage is reused. */
  readonly events: readonly BehaviorEvent[];
  readonly sensors?: readonly { readonly sensorEntityId: string; readonly visitorEntityId: string; readonly entered: boolean }[];
  enqueue(command: PhysicsCommand): void;
  /** Delivery is queued for the next tick; never invokes another behavior recursively. */
  emit(portId: string, value: JsonValue): void;
}
/** Runtime factories are separate from declarative plugins and never serialized. */
export interface BehaviorRuntime {
  /** Live configuration is read from the current tick context, not captured at creation. */
  tick(phase: InteractionPhase, context: BehaviorTickContext): void;
  dispose(): void;
}
export interface BehaviorRuntimeFactory {
  readonly type: string;
  create(entity: EntityRecord, behavior: BehaviorRecord): BehaviorRuntime;
}
export type InteractionCommand =
  | { readonly kind: "place"; readonly entity: EntityRecord }
  | { readonly kind: "duplicate"; readonly entityIds: readonly string[]; readonly offset_m: Vec3; readonly externalConnections?: boolean }
  | { readonly kind: "import"; readonly document: unknown }
  | { readonly kind: "remove"; readonly entityId: string }
  | { readonly kind: "transform"; readonly entityId: string; readonly transform: Transform }
  | { readonly kind: "configure"; readonly entityId: string; readonly behaviorId?: string; readonly value: VersionedConfig }
  | { readonly kind: "set-behaviors"; readonly entityId: string; readonly behaviors: readonly BehaviorRecord[] }
  | { readonly kind: "connect"; readonly connection: Connection }
  | { readonly kind: "disconnect"; readonly connectionId: string }
  | { readonly kind: "possess"; readonly entityId: string }
  | { readonly kind: "release-control" }
  | { readonly kind: "aim"; readonly origin_m: Vec3; readonly direction: Vec3 }
  | { readonly kind: "action"; readonly action: ActionSample }
  | { readonly kind: "transport"; readonly action: "play" | "pause" | "reset" };
export interface CommandEnvelope {
  readonly id: string;
  readonly generation: number;
  readonly documentRevision: number;
  readonly command: InteractionCommand;
}
export type CommandResult =
  | { readonly accepted: true; readonly documentRevision: number; readonly impact: UpdateImpact }
  | { readonly accepted: false; readonly reason: string };
export type InteractionAvailability =
  | { readonly available: true }
  | { readonly available: false; readonly reason: string };
export interface RuntimeActionOffer {
  readonly behaviorId: string;
  readonly actionId: string;
  readonly label: string;
  readonly availability: InteractionAvailability;
}
export interface InteractionView {
  readonly generation: number;
  readonly documentRevision: number;
  readonly document: InteractionDocument;
  readonly readiness: { readonly state: "preparing" | "ready" | "unavailable" | "failed"; readonly reasons: readonly string[] };
  readonly running: boolean;
  readonly context: "edit" | "observe" | "play";
  readonly controlledEntityId?: string;
  /** Host resolves eligibility, range and occlusion; UI never switches on kind IDs. */
  readonly controlCandidates: readonly { readonly entityId: string; readonly availability: InteractionAvailability }[];
  readonly target?: { readonly entityId: string; readonly actions: readonly RuntimeActionOffer[] };
  readonly snapshot?: WorldSnapshot;
}
/** One host per pane. Mutations return only after validation/admission. */
export interface InteractionHost {
  getView(): InteractionView;
  subscribe(listener: () => void): () => void;
  dispatch(envelope: CommandEnvelope): Promise<CommandResult>;
}
