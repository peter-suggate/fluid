import { clothMaterialErrors } from "./material";
import type { ClothDocument, ClothMaterial, ClothState, ClothTopologyData, ClothVector } from "./types";

export const sub3 = (a: ClothVector, b: ClothVector): [number, number, number] => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const dot3 = (a: ClothVector, b: ClothVector) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross3 = (a: ClothVector, b: ClothVector): [number, number, number] => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm3 = (a: ClothVector) => Math.hypot(...a);
export const scale3 = (a: ClothVector, s: number): [number, number, number] => [a[0] * s, a[1] * s, a[2] * s];
export const point3 = (p: ArrayLike<number>, i: number): [number, number, number] => [p[3 * i]!, p[3 * i + 1]!, p[3 * i + 2]!];

export interface ClothTriangle {
  readonly vertices: readonly [number, number, number];
  readonly material: number;
  readonly restArea_m2: number;
  readonly ru: ClothVector;
  readonly rv: ClothVector;
}
export interface ClothHinge {
  readonly vertices: readonly [number, number, number, number];
  readonly faces: readonly [number, number];
  readonly restAngle_rad: number;
  readonly shapeFactor: number;
}
export interface ClothRod {
  readonly vertices: readonly number[];
  readonly material: number;
  readonly segmentOffset: number;
  readonly restLengths_m: Float64Array;
  readonly restCurvature: Float64Array;
  readonly restTwist: Float64Array;
  readonly restDirectors: Float64Array;
}

/** Minimal rotation between tangents, followed by orthogonalization. */
export function transportDirector(d: ClothVector, from: ClothVector, to: ClothVector): [number, number, number] {
  const axis = cross3(from, to), cosine = Math.max(-1, Math.min(1, dot3(from, to)));
  let rotated: ClothVector = d;
  if (cosine > -1 + 1e-10) {
    const a = cross3(axis, d), b = cross3(axis, a);
    rotated = [d[0] + a[0] + b[0] / (1 + cosine), d[1] + a[1] + b[1] / (1 + cosine), d[2] + a[2] + b[2] / (1 + cosine)];
  }
  const normal = sub3(rotated, scale3(to, dot3(rotated, to))), length = norm3(normal);
  if (length < 1e-12) {
    const basis: ClothVector = Math.abs(to[0]) < .8 ? [1, 0, 0] : [0, 1, 0], fallback = cross3(to, basis);
    return scale3(fallback, 1 / norm3(fallback));
  }
  return scale3(normal, 1 / length);
}
export interface PreparedClothTopology {
  readonly data: ClothTopologyData;
  readonly materials: readonly ClothMaterial[];
  readonly vertexCount: number;
  readonly triangles: readonly ClothTriangle[];
  readonly hinges: readonly ClothHinge[];
  readonly rods: readonly ClothRod[];
  readonly segmentCount: number;
  readonly vertexMaterial: Uint32Array;
  readonly kinematicFixed: Uint8Array;
  readonly attachmentTranslation_N_m: Float64Array;
  readonly attachmentTwist_N_m: Float64Array;
  readonly incidentTriangles: readonly number[][];
  readonly incidentHinges: readonly number[][];
  readonly incidentRods: readonly number[][];
}

export function clothTopologyErrors(data: ClothTopologyData, materials: readonly ClothMaterial[]): string[] {
  const errors = materials.flatMap(clothMaterialErrors);
  if (!materials.length) errors.push("Cloth requires at least one material");
  const n = data.restPositions_m.length / 3;
  if (!Number.isInteger(n) || n < 1) errors.push("Cloth rest positions must contain complete xyz vertices");
  if (!data.restPositions_m.every(Number.isFinite)) errors.push("Cloth positions must be finite");
  if (data.triangles.length % 3) errors.push("Cloth triangles must contain complete triples");
  for (const [label, indices] of [["triangle", data.triangles], ["rod", data.rodIndices]] as const)
    if (!indices.every(i => Number.isInteger(i) && i >= 0 && i < n)) errors.push(`Cloth ${label} index out of range`);
  const offsets = data.rodOffsets;
  const rodCount=offsets.length-1,segmentCount=data.rodIndices.length-rodCount;
  if (!offsets.length || offsets[0] !== 0 || offsets.at(-1) !== data.rodIndices.length
    || !offsets.every((x, i) => Number.isInteger(x) && x >= 0 && (i === 0 || x >= offsets[i - 1]! + 2)))
    errors.push("Cloth rod offsets must delimit ordered strands of at least two vertices");
  for (const [key, length] of [["initialPositions_m", 3 * n], ["initialSupportPositions_m",3*n], ["initialVelocities_m_s", 3 * n], ["fixed", n], ["groups", n], ["solidFraction", n], ["initialWaterVolume_m3", n], ["triangleMaterials", data.triangles.length / 3], ["triangleIds", data.triangles.length / 3], ["uv", 2 * n], ["rodMaterials", offsets.length - 1], ["rodIds", offsets.length - 1]] as const) {
    const values = data[key];
    if (values && (values.length !== length || !values.every(Number.isFinite))) errors.push(`Cloth ${key} has an invalid length or nonfinite value`);
  }
  if (data.solidFraction?.some(v => !(v > 0 && v <= 1))) errors.push("Cloth vertex solid fraction must be in (0, 1]");
  if (data.initialWaterVolume_m3?.some(v => v < 0)) errors.push("Cloth water volume must be nonnegative");
  if (data.fixed?.some(v => !Number.isInteger(v) || v < 0 || v > 3)) errors.push("Cloth fixed mask uses only bits 1 and 2");
  for (const list of [data.triangleMaterials, data.rodMaterials])
    if (list?.some(i => !Number.isInteger(i) || i < 0 || i >= materials.length)) errors.push("Cloth material index out of range");
  for(const [key,length]of [["initialReferenceTwist_rad",segmentCount],["initialAttachmentTwist_rad",segmentCount],["rodStartClipNormals",3*rodCount],["rodEndClipNormals",3*rodCount],["initialTwist_rad",segmentCount],["initialAngularVelocity_rad_s",segmentCount],["initialRodDirectors",3*segmentCount],["initialRodTangents",3*segmentCount],["restRodDirectors",3*segmentCount]] as const){const values=data[key];if(values&&(values.length!==length||!values.every(Number.isFinite)))errors.push(`Cloth ${key} has an invalid length or nonfinite value`);}
  for(const normals of [data.rodStartClipNormals,data.rodEndClipNormals])if(normals)for(let i=0;i<normals.length;i+=3){const length=Math.hypot(normals[i]!,normals[i+1]!,normals[i+2]!);if(length!==0&&Math.abs(length-1)>1e-5)errors.push("Cloth rod clipping normals must be unit vectors or zero");}
  if(data.rodSegmentOrigins&&(data.rodSegmentOrigins.length!==segmentCount||data.rodSegmentOrigins.some(o=>!Number.isInteger(o.rodId)||o.rodId<0||!Number.isInteger(o.segment)||o.segment<0)))errors.push("Invalid cloth rod segment provenance");
  return errors;
}

export function validateClothDocument(document: ClothDocument): string[] {
  if (!document || typeof document !== "object") return ["Cloth document must be an object"];
  const errors: string[] = [];
  if (document.schemaVersion !== "1.0.0") errors.push("Unsupported cloth document schema");
  if (typeof document.id !== "string" || !document.id) errors.push("Cloth document requires an id");
  if (document.appearance !== undefined && document.appearance !== 'material' && document.appearance !== 'source-preview') errors.push('Unsupported cloth appearance');
  if(document.source?.previewParticleGroupCount!==undefined&&(!Number.isSafeInteger(document.source.previewParticleGroupCount)||document.source.previewParticleGroupCount<1))errors.push('Invalid source preview group count');
  if (!Array.isArray(document.materials) || !document.materials.length) return [...errors, "Cloth document requires materials"];
  errors.push(...document.materials.flatMap(clothMaterialErrors));
  if (!document.geometry) return [...errors, "Cloth document requires geometry"];
  if (document.geometry.kind === "inline") {
    try { errors.push(...clothTopologyErrors(document.geometry.data, document.materials)); }
    catch { errors.push("Invalid inline cloth topology"); }
  } else if (document.geometry.kind === "asset") {
    if (!document.geometry.url || !/^[a-f0-9]{64}$/i.test(document.geometry.sha256)
      || !Number.isInteger(document.geometry.vertexCount) || document.geometry.vertexCount < 1) errors.push("Invalid cloth asset reference");
  } else errors.push(`Unsupported cloth geometry kind: ${(document.geometry as { kind: unknown }).kind}`);
  for (const script of document.scripts ?? []) {
    if (script.kind !== "translate" && script.kind !== "rotate") errors.push("Unsupported cloth support script");
    if (!Number.isInteger(script.group) || !Number.isFinite(script.start_s) || !(script.end_s > script.start_s)) errors.push("Invalid cloth support script interval or group");
    if (script.vector?.length !== 3 || !script.vector.every(Number.isFinite)) errors.push("Invalid cloth support script vector");
    if (script.kind === "rotate" && (!Number.isFinite(script.angle_rad) || Math.hypot(...script.vector) < 1e-12)) errors.push("Invalid cloth support rotation");
  }
  return errors;
}

export function hingeAngle(positions: ArrayLike<number>, v: readonly [number, number, number, number]): number {
  const [a, b, c, d] = v.map(i => point3(positions, i)) as unknown as [ClothVector, ClothVector, ClothVector, ClothVector];
  const edge = sub3(b, a), length = norm3(edge), n0 = cross3(edge, sub3(c, a)), n1 = cross3(sub3(d, a), edge);
  if (!(length > 0 && norm3(n0) > 0 && norm3(n1) > 0)) throw new Error("Degenerate cloth hinge");
  return Math.atan2(dot3(scale3(edge, 1 / length), cross3(n0, n1)), dot3(n0, n1));
}

/** Exact released DER first-frame seed followed by spatial parallel transport.
 * Persist these frames on source asset decode so picking and mechanics agree. */
export function sourceRodReferenceDirectors(data:ClothTopologyData):readonly number[]{
  const result:number[]=[];
  for(let r=0;r<data.rodOffsets.length-1;r++){
    let previousT:ClothVector=[1,0,0],previousD:ClothVector=[0,1,0];
    for(let j=data.rodOffsets[r]!;j<data.rodOffsets[r+1]!-1;j++){
      const edge=sub3(point3(data.restPositions_m,data.rodIndices[j+1]!),point3(data.restPositions_m,data.rodIndices[j]!)),t=scale3(edge,1/norm3(edge));
      let d:ClothVector;
      if(j===data.rodOffsets[r]){
        const seed:[number,number,number]=[0,0,0],zero=t.findIndex(v=>Math.abs(v)<1e-12);
        if(zero>=0)seed[zero]=1;
        else{let largest=0;for(let k=1;k<3;k++)if(Math.abs(t[k]!)>Math.abs(t[largest]!))largest=k;const other=(largest+1)%3;seed[other]=t[largest]!;seed[largest]=-t[other]!;}
        d=scale3(seed,1/norm3(seed));
      }else d=transportDirector(previousD,previousT,t);
      result.push(...d);previousT=t;previousD=d;
    }
  }
  return result;
}

export function prepareClothTopology(data: ClothTopologyData, materials: readonly ClothMaterial[]): PreparedClothTopology {
  const errors = clothTopologyErrors(data, materials);
  if (errors.length) throw new Error(errors.join("; "));
  const n = data.restPositions_m.length / 3, triangles: ClothTriangle[] = [], hinges: ClothHinge[] = [], rods: ClothRod[] = [];
  const incidentTriangles = Array.from({ length: n }, () => [] as number[]), incidentHinges = Array.from({ length: n }, () => [] as number[]), incidentRods = Array.from({ length: n }, () => [] as number[]);
  const vertexMaterial = new Uint32Array(n), materialWeight = new Float64Array(n);
  const edges = new Map<string, { a: number; b: number; c: number; face: number; count: number }>();
  for (let f = 0; f < data.triangles.length / 3; f++) {
    const vertices = data.triangles.slice(3 * f, 3 * f + 3) as unknown as [number, number, number];
    const [a, b, c] = vertices.map(i => point3(data.restPositions_m, i)) as unknown as [ClothVector, ClothVector, ClothVector];
    const e = sub3(b, a), d = sub3(c, a), l = norm3(e), normal = cross3(e, d), twiceArea = norm3(normal);
    if (!(l > 1e-14 && twiceArea > 1e-20) || new Set(vertices).size !== 3) throw new Error(`Degenerate cloth triangle ${f}`);
    const u = dot3(d, e) / l, v = twiceArea / l, area = twiceArea / 2, material = data.triangleMaterials?.[f] ?? 0;
    triangles.push({ vertices, material, restArea_m2: area, ru: [-1 / l, 1 / l, 0], rv: [(u / l - 1) / v, -u / (l * v), 1 / v] });
    for (const vertex of vertices) { incidentTriangles[vertex]!.push(f); if (area > materialWeight[vertex]!) { vertexMaterial[vertex] = material; materialWeight[vertex] = area; } }
    for (let k = 0; k < 3; k++) {
      const a = vertices[k]!, b = vertices[(k + 1) % 3]!, c = vertices[(k + 2) % 3]!;
      const key = `${Math.min(a, b)}:${Math.max(a, b)}`, previous = edges.get(key);
      if (!previous) edges.set(key, { a, b, c, face: f, count: 1 });
      else {
        if (++previous.count > 2) throw new Error(`Nonmanifold cloth edge ${key}`);
        if (previous.a !== b || previous.b !== a) throw new Error(`Inconsistent cloth winding at edge ${key}`);
        const ids: [number, number, number, number] = [previous.a, previous.b, previous.c, c];
        const hinge = hinges.length, edgeLength = norm3(sub3(point3(data.restPositions_m, a), point3(data.restPositions_m, b)));
        hinges.push({ vertices: ids, faces: [previous.face, f], restAngle_rad: hingeAngle(data.restPositions_m, ids), shapeFactor: 3 * edgeLength ** 2 / (area + triangles[previous.face]!.restArea_m2) });
        for (const vertex of ids) incidentHinges[vertex]!.push(hinge);
      }
    }
  }
  let segmentCount = 0;
  for (let r = 0; r < data.rodOffsets.length - 1; r++) {
    const vertices = data.rodIndices.slice(data.rodOffsets[r], data.rodOffsets[r + 1]), restLengths_m = new Float64Array(vertices.length - 1), material = data.rodMaterials?.[r] ?? 0;
    for (let i = 0; i < restLengths_m.length; i++) {
      const l = norm3(sub3(point3(data.restPositions_m, vertices[i + 1]!), point3(data.restPositions_m, vertices[i]!)));
      if (!(l > 1e-14)) throw new Error(`Zero-length cloth rod ${r} segment ${i}`);
      restLengths_m[i] = l;
    }
    const restDirectors = new Float64Array(restLengths_m.length * 3), restCurvature = new Float64Array(Math.max(0, restLengths_m.length - 1) * 4),restTwist=new Float64Array(Math.max(0,restLengths_m.length-1));
    let previousT: ClothVector = [1, 0, 0], previousD: ClothVector = [0, 1, 0];
    for (let i = 0; i < restLengths_m.length; i++) {
      const t = scale3(sub3(point3(data.restPositions_m, vertices[i + 1]!), point3(data.restPositions_m, vertices[i]!)), 1 / restLengths_m[i]!);
      const d = data.restRodDirectors?point3(data.restRodDirectors,segmentCount+i):transportDirector(previousD, previousT, t); restDirectors.set(d, i * 3);
      if (i) {
        const denom = 1 + dot3(previousT, t);
        if (denom < 1e-10) throw new Error(`Antiparallel segments in cloth rod ${r}`);
        const kb = scale3(cross3(previousT, t), 2 / denom), m2a = cross3(previousT, previousD), m2b = cross3(t, d);
        restCurvature.set([dot3(kb,m2a),-dot3(kb,previousD),dot3(kb,m2b),-dot3(kb,d)],4*(i-1));
        const transported=transportDirector(previousD,previousT,t);restTwist[i-1]=Math.atan2(dot3(t,cross3(transported,d)),dot3(transported,d));
      }
      previousT = t; previousD = d;
    }
    rods.push({ vertices, material, segmentOffset: segmentCount, restLengths_m, restDirectors, restCurvature,restTwist }); segmentCount += restLengths_m.length;
    for (const vertex of new Set(vertices)) { incidentRods[vertex]!.push(r); if (materialWeight[vertex] === 0) vertexMaterial[vertex] = material; }
  }
  for (let i = 0; i < n; i++) if (!incidentTriangles[i]!.length && !incidentRods[i]!.length) throw new Error(`Unreferenced cloth vertex ${i}`);
  const finiteNodes=Uint8Array.from({length:n},(_,i)=>{
    const modes=new Set([...incidentTriangles[i]!.map(f=>triangles[f]!.material),...incidentRods[i]!.map(r=>rods[r]!.material)].map(m=>materials[m]!.attachmentMultiplier!==undefined));
    if(modes.size>1)throw new Error(`Cloth vertex ${i} mixes finite source attachments and kinematic materials`);
    return modes.has(true)?1:0;
  });
  const finiteAttachments=finiteNodes.some(Boolean),kinematicFixed=Uint8Array.from({length:n},(_,i)=>finiteNodes[i]?0:data.fixed?.[i]??0),attachmentTranslation_N_m=new Float64Array(n),attachmentTwist_N_m=new Float64Array(segmentCount);
  if(finiteAttachments){
    const volume=new Float64Array(n),weight=new Float64Array(n),young=new Float64Array(n),shear=new Float64Array(n),length=new Float64Array(n);
    const accumulate=(i:number,m:ClothMaterial,w:number)=>{weight[i]!+=w;young[i]!+=w*m.youngModulus_Pa*(m.attachmentMultiplier??0);shear[i]!+=w*m.youngModulus_Pa/(2*(1+m.poissonRatio))*(m.attachmentMultiplier??0);};
    for(const face of triangles){const m=materials[face.material]!,v=face.restArea_m2*m.thickness_m,points=face.vertices.map(i=>point3(data.restPositions_m,i));for(let j=0;j<3;j++){const i=face.vertices[j]!,a=sub3(points[(j+1)%3]!,points[j]!),b=sub3(points[(j+2)%3]!,points[j]!),fraction=Math.atan2(norm3(cross3(a,b)),dot3(a,b))/Math.PI;volume[i]!+=v/3;accumulate(i,m,fraction*v);}}
    for(const rod of rods){const m=materials[rod.material]!;for(let j=0;j<rod.restLengths_m.length;j++){const l=rod.restLengths_m[j]!;for(const i of[rod.vertices[j]!,rod.vertices[j+1]!]){const v=Math.PI*m.rodRadius_m*(m.rodRadiusB_m??m.rodRadius_m)*l/2;volume[i]!+=v;length[i]!+=l/2;accumulate(i,m,v);}}}
    for(let i=0;i<n;i++)if(((data.fixed?.[i]??0)&1)&&weight[i]!>0)attachmentTranslation_N_m[i]=young[i]!/weight[i]!*Math.cbrt(volume[i]!);
    // Source twist at vertex j belongs to outgoing segment j; the final tip's
    // extra scalar is uncoupled and must never clamp the preceding segment.
    for(const rod of rods){const m=materials[rod.material]!;for(let j=0;j<rod.restLengths_m.length;j++){const i=rod.vertices[j]!;if(((data.fixed?.[i]??0)&2)&&weight[i]!>0&&length[i]!>0)attachmentTwist_N_m[rod.segmentOffset+j]=shear[i]!/weight[i]!*Math.PI*.25*m.rodRadius_m*(m.rodRadiusB_m??m.rodRadius_m)*(m.rodRadius_m**2+(m.rodRadiusB_m??m.rodRadius_m)**2)/length[i]!;}}
  }
  return { data, materials, vertexCount: n, triangles, hinges, rods, segmentCount, vertexMaterial, kinematicFixed, attachmentTranslation_N_m, attachmentTwist_N_m, incidentTriangles, incidentHinges, incidentRods };
}

export function createClothState(topology: PreparedClothTopology): ClothState {
  const { data, materials, vertexCount: n } = topology, volume = new Float64Array(n), mass = new Float64Array(n), fraction = new Float64Array(n);
  for (let i = 0; i < n; i++) fraction[i] = data.solidFraction?.[i] ?? materials[topology.vertexMaterial[i]!]!.solidFraction;
  for (const face of topology.triangles) {
    const m = materials[face.material]!, share = face.restArea_m2 * m.thickness_m / 3;
    for (const i of face.vertices) { volume[i]! += share; mass[i]! += share * (m.attachmentMultiplier===undefined?fraction[i]!:m.solidFraction) * m.density_kg_m3; }
  }
  for (const rod of topology.rods) {
    const m = materials[rod.material]!, area = Math.PI * m.rodRadius_m * (m.rodRadiusB_m??m.rodRadius_m);
    for (let j = 0; j < rod.restLengths_m.length; j++) for (const i of [rod.vertices[j]!, rod.vertices[j + 1]!]) {
      const share = area * rod.restLengths_m[j]! / 2; volume[i]! += share; mass[i]! += share * (m.attachmentMultiplier===undefined?fraction[i]!:m.solidFraction) * m.density_kg_m3;
    }
  }
  const rodDirectors = new Float64Array(3 * topology.segmentCount), rodTangents = new Float64Array(3 * topology.segmentCount);
  for (const rod of topology.rods) for (let j = 0; j < rod.restLengths_m.length; j++) {
    rodDirectors.set(point3(rod.restDirectors, j), 3 * (rod.segmentOffset + j));
    rodTangents.set(scale3(sub3(point3(data.restPositions_m, rod.vertices[j + 1]!), point3(data.restPositions_m, rod.vertices[j]!)), 1 / rod.restLengths_m[j]!), 3 * (rod.segmentOffset + j));
  }
  if(data.initialRodDirectors)rodDirectors.set(data.initialRodDirectors);
  if(data.initialRodTangents)rodTangents.set(data.initialRodTangents);
  const referenceTwist=new Float64Array(topology.segmentCount);for(const rod of topology.rods)for(let j=1;j<rod.restLengths_m.length;j++)referenceTwist[rod.segmentOffset+j]=rod.restTwist[j-1]!;
  if(data.initialReferenceTwist_rad)referenceTwist.set(data.initialReferenceTwist_rad);
  const state: ClothState = { positions_m: Float64Array.from(data.initialPositions_m ?? data.restPositions_m), velocities_m_s: data.initialVelocities_m_s ? Float64Array.from(data.initialVelocities_m_s) : new Float64Array(3 * n), dryMass_kg: mass,
    waterVolume_m3: data.initialWaterVolume_m3 ? Float64Array.from(data.initialWaterVolume_m3) : new Float64Array(n), restControlVolume_m3: volume,
    solidVolume_m3: Float64Array.from(volume, (v, i) => v * fraction[i]!), controlVolume_m3: volume.slice(), solidFraction: fraction, twist_rad: data.initialTwist_rad?Float64Array.from(data.initialTwist_rad):new Float64Array(topology.segmentCount), rodReferenceTwist_rad:referenceTwist, angularVelocity_rad_s: data.initialAngularVelocity_rad_s?Float64Array.from(data.initialAngularVelocity_rad_s):new Float64Array(topology.segmentCount), rodDirectors, rodTangents, attachmentTwistTarget_rad:data.initialAttachmentTwist_rad?Float64Array.from(data.initialAttachmentTwist_rad):new Float64Array(topology.segmentCount), time_s: 0, generation: 1, topologyRevision: 0 };
  if(data.initialPositions_m) updateClothControlVolumes(topology,state);
  return state;
}

/** Geometric capacity changes with area/length; intrinsic fiber volume remains conserved. */
export function updateClothControlVolumes(topology:PreparedClothTopology,state:ClothState):void {
  const x=state.positions_m;state.controlVolume_m3.fill(0);
  for(const face of topology.triangles) {
    const [a,b,c]=face.vertices.map(i=>point3(x,i)) as unknown as [ClothVector,ClothVector,ClothVector];
    const share=.5*norm3(cross3(sub3(b,a),sub3(c,a)))*topology.materials[face.material]!.thickness_m/3;
    for(const i of face.vertices)state.controlVolume_m3[i]!+=share;
  }
  for(const rod of topology.rods)for(let j=0;j<rod.restLengths_m.length;j++){
    const a=rod.vertices[j]!,b=rod.vertices[j+1]!,share=Math.PI*topology.materials[rod.material]!.rodRadius_m*(topology.materials[rod.material]!.rodRadiusB_m??topology.materials[rod.material]!.rodRadius_m)*norm3(sub3(point3(x,b),point3(x,a)))/2;
    state.controlVolume_m3[a]!+=share;state.controlVolume_m3[b]!+=share;
  }
  for(let i=0;i<topology.vertexCount;i++)state.solidFraction[i]=state.solidVolume_m3[i]!/state.controlVolume_m3[i]!;
}
