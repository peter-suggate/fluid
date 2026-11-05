import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_CLOTH_MATERIAL, clothMaterialErrors, membraneModulus } from "../lib/cloth/material";
import { createClothState, prepareClothTopology, validateClothDocument } from "../lib/cloth/topology";
import { clothElasticEnergy, stepCloth } from "../lib/cloth/dynamics";
import { resolveClothContacts } from "../lib/cloth/contact";
import { composeClothFeatures } from "../lib/cloth/definition";
import type { ClothMaterial, ClothTopologyData } from "../lib/cloth/types";

const material: ClothMaterial = { ...DEFAULT_CLOTH_MATERIAL, mechanicalDamping_s: 0 };
const square = (): ClothTopologyData => ({ restPositions_m: [0, 1, 0, .1, 1, 0, 0, 1, .1, .1, 1, .1], triangles: [0, 2, 1, 1, 2, 3], rodOffsets: [0], rodIndices: [] });
const close = (a: number, b: number, tolerance = 1e-10) => assert.ok(Math.abs(a - b) < tolerance, `${a} differs from ${b}`);

test("cloth material mass uses intrinsic density and solid fraction, including shared pile vertices", () => {
  const data = { ...square(), rodOffsets: [0, 2], rodIndices: [0, 1] }, topology = prepareClothTopology(data, [material]), state = createClothState(topology);
  const volume = .01 * material.thickness_m + Math.PI * material.rodRadius_m ** 2 * .1;
  close(state.restControlVolume_m3.reduce((a, b) => a + b), volume, 1e-16);
  close(state.dryMass_kg.reduce((a, b) => a + b), volume * material.density_kg_m3 * material.solidFraction, 1e-14);
  close(membraneModulus(material, .2) / membraneModulus(material, .4), .25);
});

test("shell Green strain and dihedral energies are rigid-motion invariant and resist stretch/bend", () => {
  const topology = prepareClothTopology(square(), [material]), state = createClothState(topology);
  close(clothElasticEnergy(topology, state), 0);
  for (let i = 0; i < 4; i++) { const x = state.positions_m[3 * i]!, y = state.positions_m[3 * i + 1]!; state.positions_m[3 * i] = -y + .3; state.positions_m[3 * i + 1] = x + 2; }
  close(clothElasticEnergy(topology, state), 0, 1e-18);
  state.positions_m[9]! += .02;
  assert.ok(clothElasticEnergy(topology, state) > .0001);
});

test("implicit cloth translates in free fall without artificial strain or damping", () => {
  const topology = prepareClothTopology(square(), [material]), state = createClothState(topology);
  const result = stepCloth(topology, state, .001, { iterations: 6 });
  assert.ok(result.converged);
  for (let i = 0; i < 4; i++) { close(state.positions_m[3 * i + 1]!, 1 - 9.81e-6); close(state.velocities_m_s[3 * i + 1]!, -.00981, 1e-9); }
  assert.ok(result.elasticEnergy_J < 1e-16);
});

test("implicit shell supports pins and decreases released stretch energy", () => {
  const topology = prepareClothTopology({ ...square(), fixed: [1, 1, 0, 0] }, [material]), state = createClothState(topology);
  state.positions_m[8]! += .02; state.positions_m[11]! += .02;
  const before = clothElasticEnergy(topology, state);
  for (let i = 0; i < 8; i++) stepCloth(topology, state, .0002, { gravity_m_s2: [0, 0, 0], iterations: 20 });
  assert.ok(clothElasticEnergy(topology, state) < before);
  assert.deepEqual([...state.positions_m.slice(0, 6)], square().restPositions_m.slice(0, 6));
});

test("discrete rods possess independent bending and material twist energy", () => {
  const topology = prepareClothTopology({ restPositions_m: [0, 0, 0, .01, 0, 0, .02, 0, 0], triangles: [], rodOffsets: [0, 3], rodIndices: [0, 1, 2] }, [material]), state = createClothState(topology);
  close(clothElasticEnergy(topology, state), 0);
  state.twist_rad[1] = .5; const twistEnergy = clothElasticEnergy(topology, state); assert.ok(twistEnergy > 0);
  state.twist_rad[1] = 0; state.positions_m[7] = .002; assert.ok(clothElasticEnergy(topology, state) > twistEnergy);
  const result = stepCloth(topology, state, .00001, { gravity_m_s2: [0, 0, 0], iterations: 8 });
  assert.ok(Number.isFinite(result.elasticEnergy_J));
});

test("swept cloth sphere contact catches tunnelling and returns reciprocal body impulse", () => {
  const data: ClothTopologyData = { restPositions_m: [0, 1, 0, .01, 1, 0, 0, 1, .01], triangles: [0, 2, 1], rodOffsets: [0], rodIndices: [] };
  const topology = prepareClothTopology(data, [material]), state = createClothState(topology), old = state.positions_m.slice();
  for (let i = 0; i < 3; i++) { state.positions_m[3 * i + 1] = -1; state.velocities_m_s[3 * i + 1] = -200; }
  const beforeMomentum = state.dryMass_kg.reduce((s, m) => s - 200 * m, 0);
  const report = resolveClothContacts(topology, state, old, .01, [{ kind: "sphere", center_m: [0, 0, 0], radius_m: .1, bodyId: "sphere", friction: 0 }], false);
  assert.ok(report.contacts >= 3); assert.ok(state.positions_m[1]! >= .1);
  const afterMomentum = state.dryMass_kg.reduce((s, m, i) => s + m * state.velocities_m_s[3 * i + 1]!, 0);
  close(report.reactions[0]!.linearImpulse_N_s[1], beforeMomentum - afterMomentum);
});

test("cloth self collision catches a triangle crossing between accepted frames", () => {
  const data: ClothTopologyData = { restPositions_m: [-1, 0, -1, 1, 0, -1, 0, 0, 1, -.1, .2, -.1, .1, .2, -.1, 0, .2, .1], triangles: [0, 2, 1, 3, 5, 4], rodOffsets: [0], rodIndices: [], fixed: [1, 1, 1, 0, 0, 0] };
  const topology = prepareClothTopology(data, [material]), state = createClothState(topology), old = state.positions_m.slice();
  for (let i = 3; i < 6; i++) { state.positions_m[3 * i + 1] = -.2; state.velocities_m_s[3 * i + 1] = -40; }
  const report = resolveClothContacts(topology, state, old, .01, [], true);
  assert.ok(report.selfContacts >= 3);
  for (let i = 3; i < 6; i++) assert.ok(state.positions_m[3 * i + 1]! >= material.thickness_m - 1e-8);
});

test("cloth plugin and document contracts reject unsupported geometry and invalid materials", () => {
  assert.ok(composeClothFeatures().capabilities.includes("cloth.physical-contact"));
  assert.ok(clothMaterialErrors({ ...material, thickness_m: 0 }).length);
  assert.ok(validateClothDocument({ schemaVersion: "1.0.0", id: "test", geometry: { kind: "mystery" } as never, materials: [material] }).some(s => s.includes("Unsupported cloth geometry")));
  assert.throws(() => prepareClothTopology({ ...square(), triangles: [0, 0, 1] }, [material]), /Degenerate/);
});

test("rod edge CCD catches interior crossing with both endpoints outside", () => {
  const topology = prepareClothTopology({ restPositions_m: [-.1, 0, 0, .1, 0, 0, 0, .1, -.1, 0, .1, .1], triangles: [], rodOffsets: [0, 2, 4], rodIndices: [0, 1, 2, 3], fixed: [1, 1, 0, 0] }, [material]);
  const state = createClothState(topology), old = state.positions_m.slice();
  for (const i of [2, 3]) { state.positions_m[3 * i + 1] = -.1; state.velocities_m_s[3 * i + 1] = -20; }
  const report = resolveClothContacts(topology, state, old, .01, [], true);
  assert.ok(report.selfContacts > 0);
  for (const i of [2, 3]) assert.ok(state.positions_m[3 * i + 1]! >= 2 * material.rodRadius_m - 1e-8);
});

test('analytic shell strain and hinge derivatives agree with independent energy differences',async()=>{
  const {membraneCoordinateDerivatives,hingeCoordinateDerivatives}=await import('../lib/cloth/dynamics');
  const topology=prepareClothTopology(square(),[material]),state=createClothState(topology);
  state.positions_m[9]! += .008;state.positions_m[10]! += .015;state.positions_m[5]! -= .004;
  for(let i=0;i<4;i++)for(let axis=0;axis<3;axis++) {
    const k=i*3+axis,value=state.positions_m[k]!,h=1e-6,e=clothElasticEnergy(topology,state);
    const derivatives=[...topology.incidentTriangles[i]!.map(f=>membraneCoordinateDerivatives(topology,state,f,i,axis)),...topology.incidentHinges[i]!.map(f=>hingeCoordinateDerivatives(topology,state,f,i,axis))];
    state.positions_m[k]=value+h;const plus=clothElasticEnergy(topology,state);state.positions_m[k]=value-h;const minus=clothElasticEnergy(topology,state);state.positions_m[k]=value;
    close(derivatives.reduce((sum,d)=>sum+d.gradient,0),(plus-minus)/(2*h),1e-7);
    close(derivatives.reduce((sum,d)=>sum+d.diagonal,0),(plus+minus-2*e)/(h*h),1e-4);
  }
});

test('refitted swept BVH preserves all overlaps after large nonrigid deformation',async()=>{
  const {ClothContactBVH}=await import('../lib/cloth/contact-bvh');
  const bounds=new Float64Array(1200);
  for(let i=0;i<200;i++)for(let a=0;a<3;a++){const value=Math.sin(i*7+a*11);bounds[i*6+a]=value-.1;bounds[i*6+a+3]=value+.1;}
  const tree=new ClothContactBVH(bounds);
  for(let frame=0;frame<3;frame++){
    if(frame){for(let i=0;i<200;i++)for(let a=0;a<3;a++){const shift=Math.cos(i*9+a+frame);bounds[i*6+a]!+=shift;bounds[i*6+a+3]!+=shift;}tree.refit();}
    for(let q=0;q<30;q++){const min=[Math.sin(q),Math.cos(q),-.2],max=min.map(v=>v+.4);const expected=[];for(let i=0;i<200;i++)if(min.every((v,a)=>bounds[i*6+a+3]!>=v&&bounds[i*6+a]!<=max[a]!))expected.push(i);assert.deepEqual(tree.query(min,max),expected);}
  }
});

test('rod material-point contact Jacobian preserves axial twist and transverse angular momentum',async()=>{
 const {rodContactDOF}=await import('../lib/cloth/rod-contact-dof');
 const topology=prepareClothTopology({restPositions_m:[-.5,0,0,.5,0,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1]},[material]),state=createClothState(topology),radius=material.rodRadius_m,point=[0,radius,0] as const,dof=rodContactDOF(topology,state,0,0,point),impulse=[.2,0,.3] as const;
 dof.applyImpulse(impulse);const m=state.dryMass_kg[0]!,inertia=.5*material.density_kg_m3*Math.PI*radius**4*material.solidFraction;
 for(let a=0;a<3;a++)close(m*(state.velocities_m_s[a]!+state.velocities_m_s[3+a]!),impulse[a]!,1e-10);
 close(inertia*state.angularVelocity_rad_s[0]!,radius*impulse[2],1e-10);
 close(.5*m*(state.velocities_m_s[4]!-state.velocities_m_s[1]!),-radius*impulse[0],1e-10);
 const v=dof.velocity();close(v[0],dof.inverseMass([1,0,0])*impulse[0],1e-8);close(v[2],dof.inverseMass([0,0,1])*impulse[2],1e-8);
});

test('swept full rod catches a rigid sphere between both distant endpoints',()=>{
 const topology=prepareClothTopology({restPositions_m:[-.5,.2,0,.5,.2,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1]},[material]),state=createClothState(topology),before=state.positions_m.slice();for(const i of[0,1]){state.positions_m[3*i+1]=-.2;state.velocities_m_s[3*i+1]=-4;}
 const result=resolveClothContacts(topology,state,before,.1,[{kind:'sphere',center_m:[0,0,0],radius_m:.05}],false);assert.ok(result.contacts>0);for(const i of[0,1])assert.ok(state.positions_m[3*i+1]!>=.05+material.rodRadius_m-1e-7);
});

test('rod plane contact converges both endpoint constraints and produces physical friction spin',()=>{
 const r=material.rodRadius_m,topology=prepareClothTopology({restPositions_m:[-.5,r+.1,0,.5,r+.1,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1]},[{...material,friction:.5}]),state=createClothState(topology),old=state.positions_m.slice();for(const i of[0,1]){state.positions_m[3*i+1]=r-.1;state.velocities_m_s[3*i+1]=-2;state.velocities_m_s[3*i+2]=1;}
 const result=resolveClothContacts(topology,state,old,.1,[{kind:'plane',normal:[0,1,0],offset_m:0,friction:.5,bodyId:'floor'}],false);for(const i of[0,1])assert.ok(state.positions_m[3*i+1]!>=r-1e-8);assert.ok(result.residual_m!<1e-8);assert.ok(Math.abs(state.angularVelocity_rad_s[0]!)>0,'radial friction must excite material twist');
 const impulse=result.reactions[0]!.linearImpulse_N_s;close(state.dryMass_kg[0]!*(state.velocities_m_s[2]!+state.velocities_m_s[5]!-2),-impulse[2],1e-9);
});

test('Float64 refinement keeps the original implicit inertial target from a GPU candidate',()=>{
 const topology=prepareClothTopology(square(),[material]),cold=createClothState(topology),refined=createClothState(topology);cold.velocities_m_s.fill(.02);refined.velocities_m_s.fill(.02);
 const guess={positions_m:Float64Array.from(refined.positions_m,(x,i)=>x+(i%3===1?.00001:0)),twist_rad:refined.twist_rad.slice()},options={iterations:128,tolerance_m:1e-10};
 const a=stepCloth(topology,cold,.001,options),b=stepCloth(topology,refined,.001,options,guess);assert.ok(a.converged&&b.converged);for(let i=0;i<cold.positions_m.length;i++)close(cold.positions_m[i]!,refined.positions_m[i]!,1e-9);close(refined.time_s,.001);
});

test('sampled collision fields share exact full-rod contact and conservative spatial bounds',()=>{
 const r=material.rodRadius_m,topology=prepareClothTopology({restPositions_m:[-.5,r+.1,0,.5,r+.1,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1]},[material]),state=createClothState(topology),old=state.positions_m.slice();for(const i of[0,1]){state.positions_m[3*i+1]=r-.1;state.velocities_m_s[3*i+1]=-2;}
 const result=resolveClothContacts(topology,state,old,.1,[{kind:'sampled',sample:p=>({distance_m:p[1],normal:[0,1,0]}),velocityAt:()=>[0,0,0],maximumSpeed_m_s:0,spatialLipschitz:1,distanceLowerBound:minimum=>minimum[1]}],false);assert.ok(result.contacts>0);for(const i of[0,1])assert.ok(state.positions_m[3*i+1]!>=r-1e-8);
});

test('finite yarn crosses a shell interior with angular friction response',()=>{
 const topology=prepareClothTopology({restPositions_m:[-.2,0,-.2,.2,0,-.2,0,0,.2,-.5,.2,0,.5,.2,0],triangles:[0,2,1],rodOffsets:[0,2],rodIndices:[3,4],fixed:[1,1,1,0,0]},[{...material,friction:.5}]),state=createClothState(topology),old=state.positions_m.slice();for(const i of[3,4]){state.positions_m[3*i+1]=-.2;state.velocities_m_s[3*i+1]=-4;state.velocities_m_s[3*i+2]=.5;}
 const report=resolveClothContacts(topology,state,old,.1,[],true);assert.ok(report.selfContacts>0);assert.ok(report.residual_m!<1e-8);assert.ok(.5*(state.positions_m[10]!+state.positions_m[13]!)>=material.rodRadius_m+.5*material.thickness_m-1e-8);assert.ok(Math.abs(state.angularVelocity_rad_s[0]!)>0);
});
