import type { Quaternion, Vec3 } from "../model";
import type { EnvironmentProxySway } from "../scenery-sway";
import {
  alongAxis,
  V,
  type EnvironmentLinearColor,
  type ProxyBuilder,
} from "./builder";

/**
 * A procedural cloud-pruned tree, grown from a seed rather than placed bead by
 * bead.
 *
 * The garden's other trees are hand-authored lollipops: a trunk cylinder and
 * two or three canopy blobs. This one is generated — a tapering trunk that
 * curves as it climbs, limbs that leave it on a spiral and bend from outward to
 * upward, and foliage pads clustered on the limb ends the way a niwaki is
 * pruned. Every part is a real analytic primitive, so it self-shadows, occludes
 * the far bank, and is reflected in the pond like anything else in the set.
 *
 * Determinism is the contract: the same spec builds the same tree on every
 * rebuild, because the environment catalog's static revision — and with it the
 * whole sparse publication cache — is keyed on the geometry produced here. The
 * only entropy is an integer avalanche hash of the branch path.
 *
 * Geometry is planned before it is published so the gust can be sized against
 * the tree's real extent rather than an authored guess; see `treeSwayFor`.
 */

export interface ProceduralTreeSpec {
  /** Key prefix. Every emitted primitive is published as `${key}/...`. */
  readonly key: string;
  /** Trunk base, normally already resolved onto the terrain under the tree. */
  readonly root_m: Vec3;
  /** Root to the top of the highest foliage pad, in metres. */
  readonly height_m: number;
  /** Trunk radius where it leaves the ground. */
  readonly rootRadius_m: number;
  /** Horizontal reach of the widest limb, in metres. */
  readonly spread_m: number;
  /** Varies branch angles, lengths and pad sizes. Any integer. */
  readonly seed: number;
  /** Direction the crown leans toward, in XZ. Need not be normalized. */
  readonly leanXZ: readonly [number, number];
  /** Trunk and limb colour, by value. */
  readonly bark: (value: number) => EnvironmentLinearColor;
  /** Foliage colour, by value. */
  readonly leaf: (value: number) => EnvironmentLinearColor;
}

export type ProceduralTreeShape =
  | { readonly kind: "cone"; readonly baseRadius_m: number; readonly topRadius_m: number; readonly halfHeight_m: number }
  | { readonly kind: "ellipsoid"; readonly radii_m: Vec3 };

export interface ProceduralTreePart {
  /** Unprefixed; the caller's spec key is prepended at publication. */
  readonly key: string;
  readonly group: string;
  readonly role: "trunk" | "limb" | "foliage";
  readonly center_m: Vec3;
  readonly orientation: Quaternion;
  readonly shape: ProceduralTreeShape;
  readonly colorLinear: EnvironmentLinearColor;
  /** Distance from the trunk base: the lever the gust acts on. */
  readonly lever_m: number;
  /** Phase offset within its own limb, so limbs do not move in lockstep. */
  readonly limbPhase_rad: number;
}

export interface ProceduralTreePlan {
  readonly spec: ProceduralTreeSpec;
  readonly parts: readonly ProceduralTreePart[];
  /** Largest `lever_m` in the tree — what the excursion budget is set by. */
  readonly maximumLever_m: number;
}

/** How much motion this tree may spend, decided by the caller's lattice. */
export interface ProceduralTreeSwaySpec {
  /**
   * Peak surface excursion any one part may reach, in metres. The bend and the
   * roll are apportioned out of this one number so the total stays inside the
   * caller's budget without the caller having to convert metres into angles.
   */
  readonly excursion_m: number;
}

/** Share of the excursion spent on travel; the rest turns normals in place. */
const SWAY_BEND_SHARE = .62;
/** A nearly round part would need an absurd angle to spend its share. Cap it. */
const SWAY_MAXIMUM_TWIST_RAD = .30;

const add = (a: Vec3, b: Vec3): Vec3 => V(a.x + b.x, a.y + b.y, a.z + b.z);
const scale = (a: Vec3, k: number): Vec3 => V(a.x * k, a.y * k, a.z * k);
const mid = (a: Vec3, b: Vec3): Vec3 => V(.5 * (a.x + b.x), .5 * (a.y + b.y), .5 * (a.z + b.z));
const difference = (a: Vec3, b: Vec3): Vec3 => V(a.x - b.x, a.y - b.y, a.z - b.z);
const length = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);

function normalize(a: Vec3): Vec3 {
  const magnitude = length(a);
  if (!(magnitude > 0)) throw new Error("Procedural tree direction must have nonzero length");
  return scale(a, 1 / magnitude);
}

/** Integer avalanche hash in [0, 1). The same tree on every rebuild. */
function hash01(n: number): number {
  let h = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
}

/** Hash mapped to [-1, 1). */
const hashSigned = (n: number): number => 2 * hash01(n) - 1;

/**
 * Blend a direction toward vertical. Limbs leave the trunk almost horizontally
 * and finish nearly upright; this is the one control that makes the silhouette
 * read as a pruned garden tree rather than as a shrub.
 */
function towardVertical(direction: Vec3, amount: number): Vec3 {
  return normalize(add(scale(direction, 1 - amount), V(0, amount, 0)));
}

/**
 * Grow one tree's geometry. Pure: it allocates no GPU resources, touches no
 * builder, and can be inspected on its own by a test that wants to know how far
 * the crown reaches before anything is published.
 */
export function planProceduralTree(spec: ProceduralTreeSpec): ProceduralTreePlan {
  const { root_m: root, height_m: height, rootRadius_m: rootRadius, spread_m: spread, seed, bark, leaf } = spec;
  if (!(height > 0) || !(rootRadius > 0) || !(spread > 0)) throw new RangeError("Procedural tree needs a positive height, root radius and spread");
  const parts: ProceduralTreePart[] = [];
  const lean = (() => {
    const [x, z] = spec.leanXZ;
    const magnitude = Math.hypot(x, z);
    return magnitude > 0 ? V(x / magnitude, 0, z / magnitude) : V(1, 0, 0);
  })();
  const across = V(-lean.z, 0, lean.x);

  /** A tapered run from `from` to `to`, published as a truncated cone. */
  const segment = (
    key: string, group: string, role: ProceduralTreePart["role"],
    from: Vec3, to: Vec3, fromRadius: number, toRadius: number, value: number, limbPhase_rad: number,
  ): void => {
    const axis = difference(to, from);
    const halfHeight_m = .5 * length(axis);
    if (!(halfHeight_m > 0)) throw new Error(`Procedural tree segment ${key} has zero length`);
    const center_m = mid(from, to);
    parts.push({
      key, group, role, center_m, orientation: alongAxis(axis),
      shape: { kind: "cone", baseRadius_m: fromRadius, topRadius_m: toRadius, halfHeight_m },
      colorLinear: bark(value), lever_m: length(difference(center_m, root)), limbPhase_rad,
    });
  };

  // ---- Root flare -------------------------------------------------------
  // Three short buttresses splayed off the base. They are what stops the trunk
  // reading as a dowel pushed into the lawn, and they hold the contact shadow
  // where the eye expects the tree to actually meet the ground.
  for (let index = 0; index < 3; index += 1) {
    const angle = 2 * Math.PI * (index / 3 + .17 * hash01(seed + 11 * index));
    const reach = rootRadius * (1.45 + .35 * hash01(seed + 11 * index + 1));
    const direction = add(scale(lean, Math.cos(angle)), scale(across, Math.sin(angle)));
    const foot = add(root, add(scale(direction, reach), V(0, -.35 * rootRadius, 0)));
    segment(`flare-${index}`, "tree-trunk", "trunk", foot, add(root, V(0, .95 * rootRadius, 0)),
      .34 * rootRadius, .62 * rootRadius, .44 + .04 * hash01(seed + 11 * index + 2), 0);
  }

  // ---- Trunk ------------------------------------------------------------
  // Four tapering segments on a path that curves into the lean, so the crown
  // sits off the root rather than directly over it. Node positions are kept:
  // limbs leave from them, and a limb that starts anywhere else visibly floats.
  const trunkSegments = 4;
  const trunkTop = root.y + .46 * height;
  const trunkNodes: Vec3[] = [root];
  for (let index = 1; index <= trunkSegments; index += 1) {
    const t = index / trunkSegments;
    // Quadratic drift into the lean: nothing at the base, most near the crown.
    const drift = .16 * spread * t * t + .012 * spread * hashSigned(seed + 29 * index);
    const wander = .05 * spread * hashSigned(seed + 29 * index + 1);
    trunkNodes.push(add(root, add(
      V(0, t * (trunkTop - root.y), 0),
      add(scale(lean, drift), scale(across, wander)),
    )));
  }
  const trunkRadiusAt = (t: number): number => rootRadius * (1 - .58 * t);
  for (let index = 0; index < trunkSegments; index += 1) {
    segment(`trunk-${index}`, "tree-trunk", "trunk", trunkNodes[index], trunkNodes[index + 1],
      trunkRadiusAt(index / trunkSegments), trunkRadiusAt((index + 1) / trunkSegments),
      .47 + .03 * hash01(seed + 41 * index), 0);
  }

  // ---- Limbs ------------------------------------------------------------
  // Five primary limbs leave the upper trunk on a golden-angle spiral so no two
  // stack in the same silhouette, and each is a four-segment arc that starts
  // out and finishes nearly upright. Every limb carries one two-segment
  // sub-limb off its elbow, which is where the crown gets an inner layer —
  // foliage with visible structure under it, rather than a canopy floating on
  // nothing.
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const limbCount = 5;
  const tips: { at: Vec3; direction: Vec3; order: 0 | 1; limb: number; phase_rad: number; attachment: number }[] = [];
  for (let limb = 0; limb < limbCount; limb += 1) {
    const attachment = .34 + .66 * (limb / (limbCount - 1));
    const limbPhase = .22 * hashSigned(seed + 103 * limb);
    const node = add(
      trunkNodes[Math.min(trunkSegments, 1 + Math.floor(attachment * (trunkSegments - 1)))],
      V(0, .04 * height * hashSigned(seed + 53 * limb), 0),
    );
    const azimuth = goldenAngle * limb + .45 * hashSigned(seed + 53 * limb + 1);
    const outward = add(scale(lean, Math.cos(azimuth)), scale(across, Math.sin(azimuth)));
    // The one control that decides the silhouette: the lowest limb leaves
    // almost horizontally and reaches furthest, the highest climbs and stays
    // short. That is what stacks the crown into distinct tiers with sky between
    // them, instead of the single cauliflower a uniform rise produces — and the
    // gaps matter twice over here, because they are what lets the lantern throw
    // light through the canopy rather than around it.
    const rise = .10 + .92 * attachment * attachment + .10 * hash01(seed + 53 * limb + 2);
    const reach = spread * (.66 + .52 * (1 - attachment) + .18 * hash01(seed + 53 * limb + 3));
    let at = node;
    let direction = normalize(add(outward, V(0, rise, 0)));
    let radius = trunkRadiusAt(attachment) * (.60 + .12 * hash01(seed + 53 * limb + 4));
    for (let index = 0; index < 4; index += 1) {
      const next = add(at, scale(direction, reach / 4 * (1.30 - .20 * index)));
      const nextRadius = radius * (.70 + .06 * hash01(seed + 67 * limb + index));
      segment(`limb-${limb}-${index}`, "tree-branch", "limb", at, next, radius, nextRadius,
        .50 + .05 * hash01(seed + 67 * limb + index + 4), limbPhase);
      at = next;
      radius = nextRadius;
      // How hard a limb turns upward is itself a function of where it left the
      // trunk. Without this every limb converges on vertical within a couple of
      // segments and the tiers collapse back into one mass.
      direction = towardVertical(
        normalize(add(direction, scale(across, .12 * hashSigned(seed + 71 * limb + index)))),
        .05 + .34 * attachment,
      );
    }
    tips.push({ at, direction, order: 0, limb, phase_rad: limbPhase, attachment });

    const elbow = add(node, scale(difference(at, node), .50));
    const sideAzimuth = azimuth + (hash01(seed + 79 * limb) > .5 ? 1 : -1) * (.7 + .4 * hash01(seed + 79 * limb + 1));
    const sideOutward = add(scale(lean, Math.cos(sideAzimuth)), scale(across, Math.sin(sideAzimuth)));
    const sidePhase = limbPhase + .16 * hashSigned(seed + 107 * limb);
    let sideAt = elbow;
    let sideDirection = normalize(add(sideOutward, V(0, .70 + .22 * hash01(seed + 79 * limb + 2), 0)));
    let sideRadius = radius * (1.15 + .18 * hash01(seed + 79 * limb + 3));
    for (let index = 0; index < 2; index += 1) {
      const next = add(sideAt, scale(sideDirection, spread * (.36 - .09 * index) * (.85 + .3 * hash01(seed + 83 * limb + index))));
      const nextRadius = sideRadius * .74;
      segment(`branch-${limb}-${index}`, "tree-branch", "limb", sideAt, next, sideRadius, nextRadius,
        .52 + .04 * hash01(seed + 83 * limb + index + 2), sidePhase);
      sideAt = next;
      sideRadius = nextRadius;
      sideDirection = towardVertical(sideDirection, .10 + .34 * attachment);
    }
    tips.push({ at: sideAt, direction: sideDirection, order: 1, limb, phase_rad: sidePhase, attachment });
  }

  // ---- Foliage pads -----------------------------------------------------
  // Flattened domes rather than discs, and deliberately overlapping: a cluster
  // has to fuse into one lobe of cloud, because a ring of separate plates is
  // exactly what a hand-built lollipop canopy already looked like. A primary
  // tip carries a fused pair, a sub-limb tip one. Flattening still earns its
  // keep under the gust — a pad's roll turns its normals through a wide arc
  // while moving its surface only by its own anisotropy.
  let padIndex = 0;
  for (const tip of tips) {
    const cluster = tip.order === 0 ? 2 : 1;
    // A low tier is a wide flat plate and a high one a small dome, which is how
    // a pruned tree is actually read: mass at the bottom, taper to the leader.
    const base = spread * (tip.order === 0 ? .56 - .20 * tip.attachment : .38 - .12 * tip.attachment);
    const flatten = .42 + .16 * tip.attachment;
    for (let pad = 0; pad < cluster; pad += 1) {
      const jitter = seed + 101 * padIndex + 7 * pad;
      const swing = 2 * Math.PI * (pad / cluster + .21 * hash01(jitter + 1));
      const lateral = add(scale(across, Math.cos(swing)), scale(lean, Math.sin(swing)));
      const center_m = add(tip.at, add(
        scale(tip.direction, base * (pad === 0 ? .30 : .10 + .20 * hash01(jitter))),
        add(scale(lateral, pad === 0 ? 0 : base * (.46 + .20 * hash01(jitter + 2))),
          V(0, base * .18 * hashSigned(jitter + 3), 0)),
      ));
      const width = base * (pad === 0 ? 1 : .74 + .22 * hash01(jitter + 4));
      // Tilt each pad off horizontal, which both breaks the stacked-plate look
      // and gives the roll something to reveal.
      const tilt = alongAxis(normalize(add(V(0, 1, 0), scale(lateral, .22 + .20 * hash01(jitter + 7)))));
      parts.push({
        key: `pad-${padIndex}`, group: "leaf-foliage", role: "foliage",
        center_m, orientation: tilt,
        shape: { kind: "ellipsoid", radii_m: V(width, width * (flatten + .08 * hash01(jitter + 5)), width * (.86 + .16 * hash01(jitter + 6))) },
        colorLinear: leaf(.80 + .16 * hash01(seed + 97 * padIndex)),
        lever_m: length(difference(center_m, root)), limbPhase_rad: tip.phase_rad,
      });
      padIndex += 1;
    }
  }

  return { spec, parts, maximumLever_m: Math.max(...parts.map(({ lever_m }) => lever_m)) };
}

/** Widest minus narrowest local half-extent: what a roll actually moves. */
function partAnisotropy_m(shape: ProceduralTreeShape): number {
  const extents = shape.kind === "cone"
    ? [Math.max(shape.baseRadius_m, shape.topRadius_m), shape.halfHeight_m]
    : [shape.radii_m.x, shape.radii_m.y, shape.radii_m.z];
  return Math.max(...extents) - Math.min(...extents);
}

/**
 * The gust one part takes.
 *
 * The whole tree leans about its own root, so the bend alone already produces
 * the shape everyone expects — a base that stays put and tips that travel. The
 * square-root profile on top of that lets the crown flex a little further than
 * a rigid pole would, and because it is applied to the *normalized* lever the
 * outermost part is still the one that spends exactly its share of the
 * excursion: `bend * lever = share * (lever / maximumLever)^1.5 <= share`.
 *
 * The roll gets the remainder, converted from metres into an angle through the
 * part's own anisotropy, because that is the factor by which a rotation moves a
 * convex surface. Foliage takes all of its share and bare limbs a fraction of
 * theirs: a pad turning in the light is the point, a barely-tapered limb
 * rolling about its own axis is invisible.
 *
 * Phase trails the lever so the motion travels out from the trunk instead of
 * the tree snapping as one plate, and neighbouring parts — which have similar
 * levers — stay near enough in step that they never move relative to one
 * another by more than a fraction of the excursion.
 */
export function treeSwayFor(plan: ProceduralTreePlan, part: ProceduralTreePart, sway: ProceduralTreeSwaySpec): EnvironmentProxySway {
  if (!(sway.excursion_m > 0)) throw new RangeError("Procedural tree sway excursion must be positive");
  const normalizedLever = plan.maximumLever_m > 0 ? Math.min(1, part.lever_m / plan.maximumLever_m) : 0;
  const twistShare = (part.role === "foliage" ? 1 : .3) * normalizedLever * (1 - SWAY_BEND_SHARE) * sway.excursion_m;
  const anisotropy_m = partAnisotropy_m(part.shape);
  return {
    pivot_m: plan.spec.root_m,
    bendAmplitude_rad: SWAY_BEND_SHARE * sway.excursion_m / plan.maximumLever_m * Math.sqrt(normalizedLever),
    twistAmplitude_rad: anisotropy_m > 0 ? Math.min(SWAY_MAXIMUM_TWIST_RAD, twistShare / anisotropy_m) : 0,
    phase_rad: part.limbPhase_rad - .6 * normalizedLever,
  };
}

/**
 * Publish a planned tree. Parts are emitted in plan order, so the tree occupies
 * one contiguous run of owner indices — which is what lets the renderer re-pose
 * it every frame with a single buffer write.
 */
export function emitProceduralTree(builder: ProxyBuilder, plan: ProceduralTreePlan, sway?: ProceduralTreeSwaySpec): void {
  const publish = () => {
    for (const part of plan.parts) {
      const key = `${plan.spec.key}/${part.key}`;
      const tags = ["tree", part.role];
      if (part.shape.kind === "cone") {
        builder.cone(key, part.group, part.center_m, part.shape.baseRadius_m, part.shape.topRadius_m, part.shape.halfHeight_m,
          part.colorLinear, 0, tags, part.orientation);
      } else {
        builder.ellipsoid(key, part.group, part.center_m, part.shape.radii_m, part.colorLinear, 0, tags, part.orientation);
      }
    }
  };
  if (!sway) { publish(); return; }
  // The builder calls the resolver once per primitive in emission order, and
  // `publish` emits in plan order, so the counter and the plan stay in step.
  let index = 0;
  builder.sway(() => treeSwayFor(plan, plan.parts[index++], sway), publish);
}
