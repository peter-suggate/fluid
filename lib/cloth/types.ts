/** SI contracts of the triangle cloth: authoring, editing, the GPU solver and
 * its publications. Positions are world metres in documents and
 * publications; the GPU solver works in a cloth-local frame (see
 * `ClothFrame`). */
export type ClothVector = readonly [number, number, number];

/** Orthotropic StVK membrane, discrete-shell bending, Rayleigh damping.
 * Stiffnesses are per unit rest area (modulus x thickness), so a material
 * is independent of mesh resolution. Warp is the material u axis, weft v. */
export interface ClothMaterial {
  readonly id: string;
  readonly arealDensity_kg_m2: number;
  /** Membrane tensile stiffness along warp and weft (Young's modulus x thickness). */
  readonly stretchWarp_N_m: number;
  readonly stretchWeft_N_m: number;
  /** In-plane shear stiffness (shear modulus x thickness). */
  readonly shear_N_m: number;
  /** Warp-weft Poisson coupling, |nu| < sqrt(weft/warp) keeps the law convex. */
  readonly poisson: number;
  /** Ratio of compressive to tensile fibre stiffness in (0, 1]: a woven
   * sheet buckles instead of carrying in-plane compression. */
  readonly compressionRatio: number;
  /** Discrete-shell bending stiffness (N m), per unit hinge length. */
  readonly bending_N_m: number;
  /** Stiffness-proportional Rayleigh coefficient (s): f = -(k_d/h) H (x - x^).  */
  readonly damping_s: number;
  /** Physical thickness, used by cuts and (lane C2) the contact offset. */
  readonly thickness_m: number;
  /** Optional AVBD hard strain limit per edge, |e| <= (1 + strainLimit) L. */
  readonly strainLimit?: number;
  readonly baseColor?: ClothVector;
}

/** Saved solver state beside a pose: enough to continue a run exactly. */
export interface ClothSolverStateData {
  /** Velocity of the previous step (adaptive initialization, BDF2). */
  readonly previousVelocities_m_s: readonly number[];
  /** Displacement of the previous step, x_t - x_{t-1} (BDF2). */
  readonly previousDisplacements_m: readonly number[];
  /** Length of the previous step, s; 0 before the first step. */
  readonly previousStep_s: number;
  /** Scene time of the cloth clock, s; attachment scripts are evaluated at it. */
  readonly time_s: number;
  /** Warm-started AVBD duals, one (lambda xyz, k) per pinned vertex in vertex order. */
  readonly pinDuals: readonly number[];
  /** One (lambda, k) per edge of the canonical edge list (`prepareClothTopology`). */
  readonly edgeDuals: readonly number[];
  /** The contact detector's state carried from step to step (C2's OgcCarriedState), so a run continued from this
   *  state steps on as the straight run does. Omitted: the detector starts as before any step. */
  readonly contact?: ClothContactStateData;
}

/** OgcCarriedState beside a pose, in world metres (the detector holds it in its cloth frame). */
export interface ClothContactStateData {
  /** Per vertex: the reference predicted position (xyz, world metres) and the residual rho_v (w; -1 before any step). */
  readonly reference: readonly number[];
  /** Per vertex: the held flags (bit 0: held in the step that wrote it). */
  readonly held: readonly number[];
}

export interface ClothTopologyData {
  /** Constitutive rest shape, world metres. */
  readonly restPositions_m: readonly number[];
  /** Current pose after an edit or snapshot; omitted means the rest shape. */
  readonly initialPositions_m?: readonly number[];
  readonly initialVelocities_m_s?: readonly number[];
  readonly solverState?: ClothSolverStateData;
  readonly triangles: readonly number[];
  readonly triangleMaterials?: readonly number[];
  /** Stable authored identity of each face, carried through cuts. */
  readonly triangleIds?: readonly number[];
  /** Material (warp, weft) coordinates per vertex, metres. Omitted: each
   * face's own rest frame with warp along the projected world x axis. */
  readonly uv?: readonly number[];
  /** Per vertex attachment group; 0 is free cloth, g > 0 is a hard AVBD pin
   * following the scripts of group g. */
  readonly groups?: readonly number[];
  /** Authored mass per vertex (kg), replacing the area lumping rho A / 3 (e.g. Bridson 2002's equal particle
   * masses). A cut carries it as a per-vertex areal density m_v / A_v, so removed cloth takes its mass along. */
  readonly vertexMass_kg?: readonly number[];
}

export type ClothGeometryReference = { readonly kind: "inline"; readonly data: ClothTopologyData };

/** Prescribed attachment motion. Every script of a group composes in order:
 * rotate, then translate/oscillate offsets. */
export interface ClothAttachmentScript {
  readonly kind: "translate" | "rotate" | "oscillate";
  readonly group: number;
  readonly start_s: number;
  readonly end_s: number;
  /** translate: total displacement; rotate: unit axis; oscillate: amplitude vector. */
  readonly vector: ClothVector;
  readonly angle_rad?: number;
  readonly origin_m?: ClothVector;
  readonly period_s?: number;
  readonly easing?: "linear" | "cosine";
}

export type ClothIntegrator = "implicit-euler" | "bdf2";

/** Caps and tolerances of the Newton-PCG step (all decided on the GPU; the caps bound the encoded work). One step per
 * frame: Newton and PCG iterations and contact re-detection rounds stay inside it and never advance time. */
export interface ClothNewtonSettings {
  /** Newton iterations per step (cap). */
  readonly iterations: number;
  /** PCG iterations per Newton iteration (cap). */
  readonly pcgIterations: number;
  /** PCG stops when r.z <= tol^2 r0.z0 (default 1e-3). */
  readonly pcgTolerance?: number;
  /** Newton stops when |g| <= tol |m g| over free vertices (default 1e-4). */
  readonly residualTolerance?: number;
  /** Newton also stops after an accepted step that moves no vertex faster than this (|alpha d_v| <= tol h, m/s;
   * default 0, off): the step's velocities are then settled to this tolerance. */
  readonly velocityTolerance_m_s?: number;
  /** Eisenstat-Walker forcing (choice 2, gamma 0.9, safeguarded): the PCG solve of Newton iteration n stops at
   * eta_n = 0.9 (|g_n| / |g_n-1|)^2 (in place of pcgTolerance), capped at this and floored at pcgTolerance; the first
   * iteration of a step solves to this. In [0, 1); default 0, off (pcgTolerance in every iteration). */
  readonly forcingMax?: number;
}

/** Per-scene solver authoring. The cloth takes exactly one implicit step per
 * advance, at the frame/fluid dt: there is no substepping of any kind; the
 * Newton and PCG caps are the only budget. */
export interface ClothSolverSettings {
  /** BDF2 by default: implicit Euler visibly damps a swing at 1/60 s. */
  readonly integrator?: ClothIntegrator;
  /** Newton-PCG caps and tolerances (default CLOTH_NEWTON_SCENE_DEFAULTS in system.ts). */
  readonly newton?: ClothNewtonSettings;
  /** Coulomb friction of cloth on cloth (self contact); omitted means frictionless. */
  readonly selfFriction?: number;
  /**
   * INTERIM (demand bug, C2; to be removed with the relative-motion pair test): the plane records per
   * vertex the contact line-search filter may hold for its projected arc, authored from the scene's
   * measured peak (default 16). Past it the filter holds the vertex where it is and reports
   * planeRecordOverflow, which is how 16 per vertex left the free-fall sheet in the air (ca462bbb).
   */
  readonly contactPlaneRecordsPerVertex?: number;
}

/** A solid the cloth would meet that the scene cannot carry yet (a moving
 * solid needs a scene rigid body with a prescribed pose and cloth contact
 * against it: lane C2, M6). The cloth system halts with an explicit failure
 * before its clock passes `start_s`; it never runs on without the solid. */
export interface ClothPendingSolid {
  readonly id: string;
  /** Scene time up to which the cloth runs correctly without it. */
  readonly start_s: number;
}

export interface ClothDocument {
  readonly schemaVersion: "2.0.0";
  readonly id: string;
  readonly geometry: ClothGeometryReference;
  readonly materials: readonly ClothMaterial[];
  readonly scripts?: readonly ClothAttachmentScript[];
  readonly solver?: ClothSolverSettings;
  readonly pendingSolids?: readonly ClothPendingSolid[];
}

/**
 * A mouse grab of a region of one cloth document (the viewport's cloth-grab gesture, `ClothSystem.setGrab`). Each
 * listed vertex is pulled by an implicit spring, of stiffness weight x the vertex's reference stiffness, towards where it
 * was at the grab's first step plus `displacement_m`. Transient input: never saved, published or undone, and a
 * rebuild of the document (a cut, an undo) lets go.
 */
export interface ClothGrab {
  /** Names one press: a new id starts a grab (its anchors taken from the live pose), the same id moves it. A press
   *  starts at zero displacement, so a grab first seen displaced (pressed before this cloth system was built) is
   *  refused for the whole press rather than started mid-drag with a jump. */
  readonly id: number;
  readonly documentId: string;
  /** The topology the vertices index (`ClothSurfacePublication.topologyRevision`); a grab of another is refused. */
  readonly topologyRevision: number;
  readonly vertices: readonly number[];
  /** Spring weight per vertex, in (0, 1]: 1 at the grab point, falling to 0 at the region's edge. */
  readonly weights: readonly number[];
  /** Where the cursor holds the grab point, minus where the press picked it (world metres). */
  readonly displacement_m: ClothVector;
}

export interface ClothFrameIdentity {
  readonly generation: number;
  readonly tick: number;
  readonly topologyRevision: number;
  readonly time_s: number;
}

/** The accepted surface every consumer (render, picking, cuts, history)
 * reads. Positions are world metres. */
export interface ClothSurfacePublication extends ClothFrameIdentity {
  readonly documentId: string;
  readonly topology: ClothTopologyData;
  readonly positions_m: Float32Array | Float64Array;
  readonly velocities_m_s?: Float32Array | Float64Array;
  readonly solverState?: ClothSolverStateData;
}
