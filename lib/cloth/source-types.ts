import type { ClothVector } from './types';

export interface WetClothSourceVariant {
 readonly id:string;
 readonly sourceId:string;
 readonly kind:'paper-corrected'|'exploratory';
 readonly label:string;
 readonly reason:string;
 readonly evidence:readonly string[];
 readonly overrides:readonly {readonly path:string;readonly sourceValue:number;readonly value:number}[];
 readonly reproductionValidated:false;
}

/** Versioned SI view of the released XML. Original settings remain available
 * because an imported input is not a claim that every source feature runs. */
export interface WetClothSourceField {
  readonly index: number;
  readonly kind: 'sphere' | 'box' | 'cylinder' | 'intersect' | 'file';
  readonly usage: 'solid' | 'source' | 'terminator';
  readonly group: number;
  readonly inside: boolean;
  readonly sampled: boolean;
  /** CSG children are not also emitted as independent root sources. */
  readonly root: boolean;
  readonly center_m?: ClothVector;
  readonly axis?: ClothVector;
  readonly angle_rad?: number;
  readonly radius_m?: number;
  readonly halfExtents_m?: ClothVector;
  readonly roundingRadius_m?: number;
  readonly halfLength_m?: number;
  readonly children?: readonly number[];
  readonly filename?: string;
  readonly scale?: number;
  readonly sampleSpacing_m?: number;
  readonly emission?: readonly {
    readonly start_s: number;
    readonly end_s: number;
    readonly velocity_m_s: ClothVector;
  }[];
  readonly sourceAttributes: Readonly<Record<string, string>>;
}

export interface WetClothSourceScript {
  readonly kind: 'translate' | 'rotate';
  readonly group: number;
  readonly start_s: number;
  readonly end_s: number;
  readonly easing: 'cubic';
  readonly easeStart_s: number;
  readonly easeEnd_s: number;
  readonly global: boolean;
  readonly displacement_m?: ClothVector;
  readonly axis?: ClothVector;
  readonly angle_rad?: number;
  readonly origin_m?: ClothVector;
  readonly sourceAttributes: Readonly<Record<string, string>>;
}

export interface WetClothSourceMaterial {
  readonly id: string;
  readonly radius_m: number;
  readonly radiusB_m: number;
  readonly thickness_m: number;
  readonly youngModulus_Pa: number;
  readonly poissonRatio: number;
  readonly density_kg_m3: number;
  readonly solidViscosity_Pa_s: number;
  readonly sourceSettings: Readonly<Record<string, number | string>>;
}

/** An authored dx and dt replacing a released XML's (scenes/discretization.ts). */
export interface WetClothSourceDiscretization {
 readonly sourceCellSize_m:number;
 readonly sourceDt_s:number;
 readonly cellSize_m:number;
 readonly dt_s:number;
 readonly reason:string;
 readonly evidence:readonly string[];
}

export interface WetClothSourceDefinition {
  readonly id: string;
  readonly variant?:WetClothSourceVariant;
  readonly discretization?:WetClothSourceDiscretization;
  readonly name: string;
  readonly family: string;
  readonly source: { readonly commit: string; readonly path: string; readonly url: string; readonly sha256: string };
  readonly asset: { readonly kind: 'asset'; readonly url: string; readonly sha256: string; readonly vertexCount: number };
  readonly counts: { readonly vertices: number; readonly triangles: number; readonly rods: number };
  readonly duration_s: number;
  readonly dt_s: number;
  readonly cellSize_m: number;
  readonly gravity_m_s2: ClothVector;
  readonly materials: readonly WetClothSourceMaterial[];
  readonly cloths: readonly { readonly firstTriangle: number; readonly triangleCount: number; readonly material: number }[];
  readonly fields: readonly WetClothSourceField[];
  readonly scripts: readonly WetClothSourceScript[];
  readonly camera: null | {
    readonly distance_m: number;
    readonly radius_m: number;
    readonly fov_deg: number;
    readonly center_m: ClothVector;
    readonly rotation: { readonly x: number; readonly y: number; readonly z: number; readonly w: number };
  };
  readonly liquid: {
    readonly density_kg_m3: number;
    readonly dynamicViscosity_Pa_s: number;
    readonly surfaceTension_N_m: number;
    readonly contactAngle_rad: number;
    readonly fiberDiameter_m: number;
    readonly solidFraction: number;
  };
  readonly sourceSettings: {
    readonly liquid: Readonly<Record<string, number | string>>;
    readonly integrator: Readonly<Record<string, string>>;
    readonly bucket: Readonly<Record<string, string>>;
    readonly collision: Readonly<Record<string, string>>;
  };
  readonly auditNotes: readonly string[];
  readonly externalDependencies: readonly { readonly path: string; readonly available_in_release: boolean;
    readonly recovered_from?: string; readonly recovered_sha256?: string; readonly license?: string;
    readonly upstream_byte_identity_verified?: boolean }[];
  readonly reproductionStatus: 'source-imported-unverified'|'paper-corrected-unverified'|'exploratory-unverified';
}

export interface WetClothSourceCatalog {
  readonly schemaVersion: 1;
  readonly upstreamCommit: string;
  readonly sourceUnits: 'CGS';
  readonly runtimeUnits: 'SI';
  readonly scope: string;
  readonly scenes: readonly WetClothSourceDefinition[];
}

/** Versioned physics input, not display-only scene decoration. */
export interface WetClothSourceProgram {
  readonly schemaVersion: 1;
  readonly sourceId: string;
  readonly variant?:WetClothSourceVariant;
  readonly fields: readonly WetClothSourceField[];
  readonly scripts: readonly WetClothSourceScript[];
  /** Rigid translation applied to all imported positions and rotation pivots. */
  readonly sourceOffset_m: ClothVector;
  readonly timeStep_s: number;
  readonly requiredCapabilities: readonly string[];
  /** Time-zero events already represented by fluid.initialLiquidVolumes. */
  readonly initializedFieldIndices?: readonly number[];
  /** Exact initial SDFs seeded by the common initial-liquid host path. */
  readonly initialSdfFieldIndices?: readonly number[];
  /** Delayed zero-velocity sphere events represented by fluid.scheduledDrops. */
  readonly scheduledFieldIndices?: readonly number[];
  /** Static roots represented by the authoritative procedural solid layer. */
  readonly bakedSolidFieldIndices?: readonly number[];
  /** Moving roots represented by exact native rigid primitives. */
  readonly movingSolidFieldIndices?: readonly number[];
}
