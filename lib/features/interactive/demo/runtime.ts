import type { Vec3 } from "../../../core/model";
import type {
  ActionSample,
  BehaviorRuntime,
  BehaviorRuntimeFactory,
  BehaviorTickContext,
  EntityRecord,
} from "../../../interaction";
import { add, scale } from "../../../core/math";
import { quaternionYaw, rotateVector, yawQuaternion } from "../geometry";

/**
 * Runtime factories for the demo behaviors, run by the simulated host.
 *
 * They talk to the world only through `BehaviorTickContext` — samples in,
 * typed physics commands and port emissions out — which is the shape the
 * production factories will have. What they do *not* have is physics: there
 * are no collision queries behind `character-move`, so the character walks on
 * the plane its spawn stands on.
 */

const GRAVITY_M_S2 = 9.81;

function samplesFor(context: BehaviorTickContext): readonly ActionSample[] {
  return context.input.actions.filter((sample) => sample.entityId === context.entity.id && sample.behaviorId === context.behavior.id);
}

function axis2(samples: readonly ActionSample[], actionId: string): readonly [number, number] {
  let x = 0, y = 0;
  for (const sample of samples) {
    if (sample.actionId !== actionId || !Array.isArray(sample.value)) continue;
    x += sample.value[0]!; y += sample.value[1]!;
  }
  return [x, y];
}

function held(samples: readonly ActionSample[], actionId: string): boolean {
  return samples.some((sample) => sample.actionId === actionId && sample.value === true);
}

function pressed(samples: readonly ActionSample[], actionId: string): boolean {
  return samples.some((sample) => sample.actionId === actionId && sample.edge === "pressed");
}

function bodyOf(context: BehaviorTickContext) {
  return context.snapshot.bodies.find((body) => body.entityId === context.entity.id);
}

const idle: BehaviorRuntime = { tick() {}, dispose() {} };

/**
 * Camera-relative is facing-relative here: `look` turns the character, and the
 * follow camera sits behind it, so W is always "away from the camera".
 */
const motor: BehaviorRuntimeFactory = {
  type: "demo.character-motor",
  create(): BehaviorRuntime {
    return {
      tick(phase, context) {
        if (phase !== "intent") return;
        const body = bodyOf(context);
        if (!body) return;
        const config = context.behavior.config;
        const samples = samplesFor(context);
        const [lookX] = axis2(samples, "look");
        const turn_rad = (lookX * (config.turnRate_deg as number) * Math.PI) / 180;
        const yaw = quaternionYaw(body.transform.orientation) - turn_rad;
        const orientation = yawQuaternion(yaw);
        if (turn_rad !== 0) {
          context.enqueue({ kind: "kinematic-target", bodyId: body.bodyId, bodyGeneration: body.bodyGeneration,
            transform: { position_m: body.transform.position_m, orientation } });
        }
        // The router leaves diagonals unnormalized on purpose; normalizing is the motor's call.
        let [right, forward] = axis2(samples, "move");
        const magnitude = Math.hypot(right, forward);
        if (magnitude > 1) { right /= magnitude; forward /= magnitude; }
        const speed = (held(samples, "run") ? config.runSpeed_m_s : config.walkSpeed_m_s) as number;
        const ahead = rotateVector(orientation, { x: 0, y: 0, z: 1 });
        const side = rotateVector(orientation, { x: -1, y: 0, z: 0 });
        const horizontal = add(scale(ahead, forward * speed), scale(side, right * speed));
        const jump = pressed(samples, "jump");
        const jumpSpeed = Math.sqrt(2 * GRAVITY_M_S2 * (config.jumpHeight_m as number));
        context.enqueue({ kind: "character-move", entityId: context.entity.id,
          velocity_m_s: { x: horizontal.x, y: jump ? jumpSpeed : 0, z: horizontal.z }, jump });
      },
      dispose() {},
    };
  },
};

const activator: BehaviorRuntimeFactory = {
  type: "demo.activator",
  create(_entity, behavior): BehaviorRuntime {
    let on = behavior.config.startsOn === true;
    let announced = false;
    return {
      tick(phase, context) {
        if (phase !== "intent") return;
        const samples = samplesFor(context);
        const before = on;
        if (context.behavior.config.mode === "momentary") {
          if (samples.some((sample) => sample.actionId === "use" && sample.edge === "pressed")) on = true;
          if (samples.some((sample) => sample.actionId === "use" && sample.edge === "released")) on = false;
        } else if (pressed(samples, "use")) on = !on;
        if (on !== before || !announced) { context.emit("active", on); announced = true; }
      },
      dispose() {},
    };
  },
};

const MOVER_AXES: Readonly<Record<string, Vec3>> = {
  up: { x: 0, y: 1, z: 0 }, forward: { x: 0, y: 0, z: 1 }, right: { x: -1, y: 0, z: 0 },
};

const mover: BehaviorRuntimeFactory = {
  type: "demo.kinematic-mover",
  create(entity: EntityRecord): BehaviorRuntime {
    let active = false, travelled_m = 0;
    return {
      tick(phase, context) {
        if (phase !== "before-physics") return;
        for (const event of context.events) {
          if (event.target.entity === entity.id && event.target.behavior === context.behavior.id && event.target.port === "active") {
            active = event.value === true;
          }
        }
        const body = bodyOf(context);
        if (!body) return;
        const config = context.behavior.config;
        const travel = config.travel_m as number, step = (config.speed_m_s as number) * context.dt_s;
        travelled_m = active ? Math.min(travel, travelled_m + step) : Math.max(0, travelled_m - step);
        const axis = rotateVector(context.entity.transform.orientation, MOVER_AXES[config.axis as string] ?? MOVER_AXES.up!);
        context.enqueue({ kind: "kinematic-target", bodyId: body.bodyId, bodyGeneration: body.bodyGeneration, transform: {
          position_m: add(context.entity.transform.position_m, scale(axis, travelled_m)),
          orientation: context.entity.transform.orientation,
        } });
      },
      dispose() {},
    };
  },
};

/** The behaviors whose whole job is to exist: they hold no state the demo can advance. */
const inert = ["demo.follow-camera", "demo.rigid-body", "demo.static-collider"].map((type): BehaviorRuntimeFactory => ({ type, create: () => idle }));

export const DEMO_RUNTIME_FACTORIES: readonly BehaviorRuntimeFactory[] = [motor, activator, mover, ...inert];
