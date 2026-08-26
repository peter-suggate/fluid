import {
  boxCenter,
  boxHandles,
  boxSize,
  boxResizeDrag,
  frameToWorld,
  moveHandles,
  pickSolidBox,
  positionFields,
  pickExcluded,
  type BoxExtent,
  type BoxResizePolicy,
  type EditorEntity,
  type EditorEntityContext,
  type EditorEntityDefinition,
  type EditorFrame,
} from "./editor-entity";
import type { EditorAxis } from "./editor-axis-constraint";
import { boundingRadius } from "./rigid-body";
import type { RigidBodyDescription, RigidShape, SceneDescription, Vec3 } from "./model";

/**
 * Direct manipulation of a rigid body.
 *
 * Two things make a body different from the boxes around it, and both are
 * declared here rather than special-cased by the viewport.
 *
 * The first is rotation. A body tumbles, so its handles live in its own frame:
 * grabbing the face that is currently facing you resizes the body along its own
 * axis, and an X lock constrains along the body's X. The tank and the water body
 * are the degenerate case of the same rule, with an identity frame.
 *
 * The second is that a body is *simulated*. The renderer draws it from live
 * solver state, not from the document, and `rigidBodies` is part of the solver's
 * seed key — so a document write per pointer-move would re-seed the run dozens of
 * times a second and still not move what is on screen. `simulatedBodyId` says so,
 * and the gesture previews as a runtime pose instead.
 */

export const RIGID_MINIMUM_HALF_SIZE_M = 0.01;

/**
 * The half-extents of a body's own bounding box, and which of them the schema
 * is obliged to keep equal.
 *
 * `dimensions_m` means something different for each shape — a sphere stores one
 * radius, a cylinder a radius and a height, a box three full extents — so the
 * box gizmo must never propose a state the document cannot hold. Linking the
 * axes that share a number is what keeps a dragged sphere a sphere.
 */
const SHAPE_EXTENTS: Readonly<Record<RigidShape, {
  half: (dimensions_m: Vec3) => Vec3;
  dimensions: (half: Vec3, previous: Vec3) => Vec3;
  links?: readonly (readonly EditorAxis[])[];
}>> = Object.freeze({
  sphere: {
    half: (d) => ({ x: d.x, y: d.x, z: d.x }),
    dimensions: (half) => ({ x: half.x, y: half.x, z: half.x }),
    links: [["x", "y", "z"]],
  },
  box: {
    half: (d) => ({ x: 0.5 * d.x, y: 0.5 * d.y, z: 0.5 * d.z }),
    dimensions: (half) => ({ x: 2 * half.x, y: 2 * half.y, z: 2 * half.z }),
  },
  cylinder: {
    half: (d) => ({ x: d.x, y: 0.5 * d.y, z: d.x }),
    dimensions: (half) => ({ x: half.x, y: 2 * half.y, z: half.x }),
    links: [["x", "z"]],
  },
  // A capsule's `y` is the length of its cylindrical section, so its bounding
  // half-height is that half-length plus one cap.
  //
  // Which means a drag that only widened it must keep the cylinder it had: read
  // naively, growing the radius at a fixed bounding height eats the body from
  // both ends and arrives at a sphere, so a capsule would refuse to get fatter.
  // Only a drag that actually moved the y side may set the length, and shrinking
  // that below the radius is a sphere — the floor, not an error.
  // A cup's third dimension is its wall, not an extent, so the box gizmo never
  // writes it: dragging a face changes how big the cup is, and how thick its
  // wall is stays where the user set it. The wall is clamped against the radius
  // and the height at evaluation (`cupWallThickness_m`), so shrinking a cup
  // past its own wall thins the wall rather than inverting the cavity.
  cup: {
    half: (d) => ({ x: d.x, y: 0.5 * d.y, z: d.x }),
    dimensions: (half, previous) => ({ x: half.x, y: 2 * half.y, z: previous.z }),
    links: [["x", "z"]],
  },
  capsule: {
    half: (d) => ({ x: d.x, y: 0.5 * d.y + d.x, z: d.x }),
    dimensions: (half, previous) => ({
      x: half.x,
      y: Math.abs(half.y - (0.5 * previous.y + previous.x)) < 1e-12
        ? previous.y
        : Math.max(0, 2 * (half.y - half.x)),
      z: half.x,
    }),
    links: [["x", "z"]],
  },
});

/** The body's own bounding box, centred on its origin. */
export function rigidBodyBox(description: RigidBodyDescription): BoxExtent {
  const half = SHAPE_EXTENTS[description.shape].half(description.dimensions_m);
  return { min: { x: -half.x, y: -half.y, z: -half.z }, max: { x: half.x, y: half.y, z: half.z } };
}

export function rigidBodyResizePolicy(description: RigidBodyDescription): BoxResizePolicy {
  return {
    minimum_m: [
      2 * RIGID_MINIMUM_HALF_SIZE_M,
      2 * RIGID_MINIMUM_HALF_SIZE_M,
      2 * RIGID_MINIMUM_HALF_SIZE_M,
    ],
    links: SHAPE_EXTENTS[description.shape].links,
  };
}

/**
 * Author a reshaped box back onto the body.
 *
 * A free axis holds its opposite side, which moves the box centre — and a body's
 * centre *is* its position, so the position moves with the dimensions. Without
 * this the anchored side would slide too, and a face drag would read as a
 * symmetric scale rather than as moving that face.
 */
export function rigidBodyBoxPatch(
  scene: SceneDescription,
  description: RigidBodyDescription,
  frame: EditorFrame,
  box: BoxExtent,
): Pick<SceneDescription, "rigidBodies"> {
  const half = boxSize(box);
  const dimensions_m = SHAPE_EXTENTS[description.shape].dimensions(
    { x: 0.5 * half.x, y: 0.5 * half.y, z: 0.5 * half.z },
    description.dimensions_m);
  const position_m = frameToWorld(frame, boxCenter(box));
  return rigidBodyPatch(scene, description.id, { dimensions_m, position_m });
}

export function rigidBodyPatch(
  scene: SceneDescription,
  id: string,
  patch: Partial<RigidBodyDescription>,
): Pick<SceneDescription, "rigidBodies"> {
  return {
    rigidBodies: scene.rigidBodies.map((body) => body.id === id ? { ...body, ...patch } : body),
  };
}

function rigidBodyEntityFor(
  context: EditorEntityContext,
  description: RigidBodyDescription,
): EditorEntity {
  // The live pose, not the authored one: the handles must sit on the body the
  // user can see, which during a run is nowhere near where the document says.
  const pose = context.bodies.find((body) => body.id === description.id);
  const frame: EditorFrame = {
    origin_m: pose?.position_m ?? description.position_m,
    orientation: pose?.orientation ?? description.orientation,
  };
  const box = rigidBodyBox(description);
  const size = boxSize(box);
  return {
    selection: { kind: "body", id: description.id },
    label: description.name,
    tone: "body",
    frame,
    box,
    sizeLabel: `${[size.x, size.y, size.z].map((value) => value.toFixed(3)).join(" × ")} m`,
    simulatedBodyId: description.id,
    handles: [
      ...boxHandles(box, {
        drag: boxResizeDrag(box, rigidBodyResizePolicy(description),
          (next) => rigidBodyBoxPatch(context.scene, description, frame, next)),
      }),
      ...moveHandles(frame.origin_m,
        (position_m) => rigidBodyPatch(context.scene, description.id, { position_m })),
    ],
    draftSubject: "body",
    editLabel: (handle) => handle.space === "world"
      ? `Moved ${description.name}` : `Resized ${description.name}`,
    fields: [
      ...positionFields(frame.origin_m,
        (position_m) => rigidBodyPatch(context.scene, description.id, { position_m })),
      {
        id: "size",
        label: "Size",
        unit: "m",
        value: description.dimensions_m.x,
        step: 0.005,
        min: RIGID_MINIMUM_HALF_SIZE_M,
        // One field, because a body's shape is one number for everything but a
        // box: the handles are where per-axis sizing lives.
        apply: (value) => {
          const ratio = value / Math.max(description.dimensions_m.x, 1e-9);
          return rigidBodyPatch(context.scene, description.id, {
            dimensions_m: {
              x: value,
              y: description.dimensions_m.y * ratio,
              z: description.dimensions_m.z * ratio,
            },
          });
        },
      },
      // What the body is made of, which no handle can express: the tray that
      // used to hold these was a docked list of every body in the scene, and
      // the three numbers it offered for the selected one are the three that
      // belong on the selected one.
      {
        id: "density",
        label: "Density",
        unit: "kg/m³",
        value: description.density_kg_m3,
        step: 10,
        min: 100,
        max: 4000,
        apply: (value) => rigidBodyPatch(context.scene, description.id, { density_kg_m3: value }),
      },
      {
        id: "restitution",
        label: "Bounce",
        unit: "—",
        value: description.restitution,
        step: 0.05,
        min: 0,
        max: 1,
        apply: (value) => rigidBodyPatch(context.scene, description.id, { restitution: value }),
      },
      {
        id: "friction",
        label: "Friction",
        unit: "—",
        value: description.friction,
        step: 0.05,
        min: 0,
        max: 1.2,
        apply: (value) => rigidBodyPatch(context.scene, description.id, { friction: value }),
      },
    ],
    remove: () => ({
      ...context.scene,
      rigidBodies: context.scene.rigidBodies.filter((body) => body.id !== description.id),
    }),
  };
}

/**
 * A body is picked by its bounding sphere here, which is what the hover chip has
 * always used. The exact G-buffer pick stays the authority for a click that
 * lands on a body: it is the gesture that also opens the throw, and it needs the
 * surface point, not just the fact of a hit.
 */
export const rigidBodyEntity: EditorEntityDefinition = {
  kind: "body",
  instances: (context) => context.scene.rigidBodies.map(
    (description) => rigidBodyEntityFor(context, description)),
  find: (context, id) => {
    const description = context.scene.rigidBodies.find((body) => body.id === id);
    return description && rigidBodyEntityFor(context, description);
  },
  /**
   * A body's two verbs.
   *
   * Carrying is not a resize, so it is not a handle; it is what you do with a
   * cup rather than what you do to one. Dropping is its counterpart and is here
   * rather than implied by releasing the carry, because a body that has been
   * set down inside the water still has to be *let go of* — the solver only
   * takes it back when nothing is holding it.
   */
  actions: (context, target) => {
    const description = context.scene.rigidBodies.find((body) => body.id === target.selection.id);
    if (!description) return [];
    return [
      {
        id: "carry",
        label: "Carry",
        icon: "carry",
        tone: "body",
        hint: "Bind it to the cursor: height follows the pointer, the wheel pushes it away, Q and E tilt it to pour \u00b7 click to put it down",
        effect: { kind: "carry", bodyId: description.id },
      },
      {
        id: "release",
        label: "Drop",
        icon: "drop",
        tone: "body",
        hint: `Let ${description.name} fall from above the tank`,
        effect: { kind: "release", bodyId: description.id },
      },
    ];
  },
  pick: (context, ray, exclude) => {
    let nearest: { id: string; distance_m: number } | undefined;
    for (const description of context.scene.rigidBodies) {
      if (pickExcluded(exclude, "body", description.id)) continue;
      const pose = context.bodies.find((body) => body.id === description.id);
      const centre = pose?.position_m ?? description.position_m;
      const radius = boundingRadius(description);
      const box: BoxExtent = {
        min: { x: centre.x - radius, y: centre.y - radius, z: centre.z - radius },
        max: { x: centre.x + radius, y: centre.y + radius, z: centre.z + radius },
      };
      const distance_m = pickSolidBox(ray, box);
      if (distance_m !== undefined && (!nearest || distance_m < nearest.distance_m)) {
        nearest = { id: description.id, distance_m };
      }
    }
    return nearest && { selection: { kind: "body", id: nearest.id }, distance_m: nearest.distance_m };
  },
};
