/** SI contracts shared by the cloth host, authoring, import and fluid plugins. */
export type ClothVector = readonly [number, number, number];

export interface ClothMaterial {
  readonly id: string;
  /** Intrinsic fiber density, before multiplication by solidFraction. */
  readonly density_kg_m3: number;
  readonly youngModulus_Pa: number;
  readonly poissonRatio: number;
  /** Full shell thickness. Independent of the microstructure fiber diameter. */
  readonly thickness_m: number;
  readonly rodRadius_m: number;
  /** Second material-axis radius; omitted means circular. */
  readonly rodRadiusB_m?: number;
  readonly solidFraction: number;
  readonly solidFractionExponent: number;
  readonly fiberDiameter_m: number;
  readonly contactAngle_rad: number;
  readonly surfaceTension_N_m: number;
  readonly waterDensity_kg_m3: number;
  readonly waterViscosity_Pa_s: number;
  readonly friction: number;
  /** Kelvin-Voigt relaxation time: viscous modulus / elastic modulus. */
  readonly mechanicalDamping_s: number;
  /** Source finite attachment multiplier; undefined retains authored kinematic pins. */
  readonly attachmentMultiplier?: number;
  /** Per-mode relaxation times; omitted values retain mechanicalDamping_s. */
  readonly membraneDamping_s?: number;
  readonly bendingDamping_s?: number;
  readonly rodStretchDamping_s?: number;
  readonly rodBendingDamping_s?: number;
  readonly rodTwistDamping_s?: number;
  readonly bendingMode?: "theta" | "tan-half" | "sin-half";
  readonly baseColor?: ClothVector;
}

/** Shared vertices allow pile roots and woven yarns to remain attached. */
export interface ClothTopologyData {
  readonly restPositions_m: readonly number[];
  /** Saved deformed pose after an edit; constitutive rest shape remains above. */
  readonly initialPositions_m?: readonly number[];
  readonly initialSupportPositions_m?: readonly number[];
  readonly initialAttachmentTwist_rad?: readonly number[];
  readonly initialVelocities_m_s?: readonly number[];
  readonly triangles: readonly number[];
  readonly triangleMaterials?: readonly number[];
  readonly triangleIds?: readonly number[];
  readonly uv?: readonly number[];
  /** CSR offsets, including terminal offset; [0] when there are no rods. */
  readonly rodOffsets: readonly number[];
  readonly rodIndices: readonly number[];
  readonly rodMaterials?: readonly number[];
  readonly rodIds?: readonly number[];
  /** Cut-plane outward normals in the rod material frame (d1,d2,tangent).
   * Zero triples mean an original rounded end. */
  readonly rodStartClipNormals?: readonly number[];
  readonly rodEndClipNormals?: readonly number[];
  /** Accepted material-frame state, persisted when editing a live rod. */
  readonly initialTwist_rad?: readonly number[];
  readonly initialReferenceTwist_rad?: readonly number[];
  readonly initialAngularVelocity_rad_s?: readonly number[];
  readonly initialRodDirectors?: readonly number[];
  readonly initialRodTangents?: readonly number[];
  readonly restRodDirectors?: readonly number[];
  /** Immediate pre-edit rod identity and segment for material-frame remap. */
  readonly rodSegmentOrigins?: readonly {readonly rodId:number;readonly segment:number}[];
  /** Per vertex: bit 1 fixes translation, bit 2 fixes twist. */
  readonly fixed?: readonly number[];
  readonly groups?: readonly number[];
  readonly solidFraction?: readonly number[];
  readonly initialWaterVolume_m3?: readonly number[];
}

export type ClothGeometryReference =
  | { readonly kind: "inline"; readonly data: ClothTopologyData }
  | { readonly kind: "asset"; readonly url: string; readonly sha256: string; readonly vertexCount: number };

/** Prescribed supports only; free cloth motion is always computed by mechanics. */
export interface ClothSupportScript {
  readonly kind: "translate" | "rotate";
  readonly group: number;
  readonly start_s: number;
  readonly end_s: number;
  /** Total displacement for translate, unit axis for rotate. */
  readonly vector: ClothVector;
  readonly angle_rad?: number;
  readonly origin_m?: ClothVector;
  readonly global?: boolean;
  readonly easing?: "cubic" | "cosine";
  readonly easeIn_s?: number;
  readonly easeOut_s?: number;
}

export interface ClothPhysicsOptions {
  readonly nonlinearDrag?:boolean;
  readonly dragExponent?:number;
  readonly pressureOnSolid?:boolean;
  readonly varyingSolidFraction?:boolean;
  readonly porePressure?:boolean;
  readonly manifoldTransport?:boolean;
  readonly pressureOnManifold?:boolean;
  readonly manifoldSubsteps?:number;
  /** Source CGS discretization floor: one cm³, expressed explicitly in SI. */
  readonly manifoldDragFloorReferenceVolume_m3?:number;
  readonly captureRate?:number;
  readonly absorption?:boolean;
  readonly solveSolid?:boolean;
  /** GPU Newton contact pool (pairs), authored from the scene's measured peak.
   * It never grows at run time: a step that fills it fails with the counts. */
  readonly contactCapacity?:number;
  /** Its swept-primitive grid entries, authored the same way. */
  readonly contactGridEntries?:number;
}

export interface ClothDocument {
  readonly schemaVersion: "1.0.0";
  readonly id: string;
  readonly geometry: ClothGeometryReference;
  readonly materials: readonly ClothMaterial[];
  /** Source preview palette is distinct from the paper's Houdini materials. */
  readonly appearance?: "material" | "source-preview";
  readonly physics?:ClothPhysicsOptions;
  readonly scripts?: readonly ClothSupportScript[];
  readonly source?: { readonly url: string; readonly sha256: string; readonly configuration: "source" | "paper-corrected" | "interactive";
    /** Original preview palette also counts initial free/solid sample groups. */
    readonly previewParticleGroupCount?:number };
}

export interface ClothState {
  readonly positions_m: Float64Array;
  readonly velocities_m_s: Float64Array;
  readonly dryMass_kg: Float64Array;
  readonly waterVolume_m3: Float64Array;
  readonly restControlVolume_m3: Float64Array;
  /** Conserved intrinsic fiber volume, independent of geometric deformation. */
  readonly solidVolume_m3: Float64Array;
  readonly controlVolume_m3: Float64Array;
  readonly solidFraction: Float64Array;
  /** One material angle and angular velocity per rod segment. */
  readonly twist_rad: Float64Array;
  readonly rodReferenceTwist_rad: Float64Array;
  readonly attachmentTwistTarget_rad: Float64Array;
  readonly angularVelocity_rad_s: Float64Array;
  /** Untwisted reference directors and accepted unit tangents, xyz per segment. */
  readonly rodDirectors: Float64Array;
  readonly rodTangents: Float64Array;
  time_s: number;
  generation: number;
  topologyRevision: number;
}

export interface ClothFrameIdentity {
  readonly generation: number;
  readonly tick: number;
  readonly topologyRevision: number;
  readonly time_s: number;
}

export interface ClothSurfacePublication extends ClothFrameIdentity {
  readonly documentId: string;
  readonly topology: ClothTopologyData;
  readonly positions_m: Float32Array | Float64Array;
  readonly supportPositions_m?: Float64Array;
  readonly attachmentTwistTarget_rad?: Float64Array;
  readonly rodReferenceTwist_rad?: Float64Array;
  readonly velocities_m_s?: Float32Array | Float64Array;
  readonly waterVolume_m3: Float32Array | Float64Array;
  readonly controlVolume_m3?: Float32Array | Float64Array;
  readonly solidFraction?: Float32Array | Float64Array;
  readonly saturation?: Float32Array;
  readonly rodTwist_rad?: Float64Array;
  readonly rodAngularVelocity_rad_s?: Float64Array;
  readonly rodDirectors?: Float64Array;
  readonly rodTangents?: Float64Array;
}
