import type { FluidInflow, SceneDescription, Vec3 } from "./model";
import { inflowStrength } from "./initial-fluid";

/** Shared by the cubic and tall-cell shaders. Each shader supplies
 * inflowGridDims(), while Params supplies the common scene/grid fields.
 *
 * The outlet is a one-sided open boundary embedded at the nozzle face. Its
 * virtual upstream reservoir contributes flux to the receiver without owning
 * a simulation cell, so the prescribed flow enters through VOF transport
 * instead of periodically overfilling a cell with a post-projection source.
 */
export const inflowBoundaryWGSL = /* wgsl */ `
fn inflowStrength()->f32{return clamp(params.inflowTiming.x,0.0,1.0);}
fn inflowAxis()->u32{let v=abs(params.inflowVelocityLength.xyz);if(v.y>v.x&&v.y>=v.z){return 1u;}if(v.z>v.x){return 2u;}return 0u;}
fn inflowFaceIndex(axis:u32)->i32{let h=params.cellGravity.xyz;let minimum=vec3f(-0.5*params.container.x,0.0,-0.5*params.container.z);let d=inflowGridDims();return clamp(i32(round((params.inflowPositionRadius[axis]-minimum[axis])/h[axis]))-1,0,d[axis]-2);}
fn inflowReceiverIndex(axis:u32)->i32{let face=inflowFaceIndex(axis);return select(face,face+1,params.inflowVelocityLength[axis]>=0.0);}
fn inflowApertureFraction(q:vec3i)->f32{
  let velocity=params.inflowVelocityLength.xyz;let speed=length(velocity);if(speed<=1e-6){return 0.0;}let direction=velocity/speed;let axis=inflowAxis();let h=params.cellGravity.xyz;let minimum=vec3f(-0.5*params.container.x,0.0,-0.5*params.container.z);let face=inflowFaceIndex(axis);var tangentA=0u;var tangentB=1u;if(axis==0u){tangentA=1u;tangentB=2u;}else if(axis==1u){tangentA=0u;tangentB=2u;}
  var point=vec3f(0.0);point[axis]=minimum[axis]+f32(face+1)*h[axis];point[tangentA]=minimum[tangentA]+(f32(q[tangentA])+0.5)*h[tangentA];point[tangentB]=minimum[tangentB]+(f32(q[tangentB])+0.5)*h[tangentB];let relative=point-params.inflowPositionRadius.xyz;let axial=dot(relative,direction);let radialDistance=length(relative-axial*direction);let edgeWidth=max(0.5*length(vec2f(h[tangentA],h[tangentB])),1e-6);let coverage=clamp(0.5+0.5*(params.inflowPositionRadius.w-radialDistance)/edgeWidth,0.0,1.0);return coverage*coverage*(3.0-2.0*coverage);
}
fn isInflowBoundaryFace(q:vec3i,axis:u32)->bool{
  return inflowStrength()>0.0&&axis==inflowAxis()&&q[axis]==inflowFaceIndex(axis)&&inflowApertureFraction(q)>0.0;
}
fn inflowBoundaryFlux(q:vec3i,axis:u32,dt:f32)->f32{
  let strength=inflowStrength();if(strength<=0.0||axis!=inflowAxis()||q[axis]!=inflowFaceIndex(axis)){return 0.0;}let fraction=inflowApertureFraction(q);if(fraction<=0.0){return 0.0;}let normalVelocity=params.inflowVelocityLength[axis]*strength*params.inflowVelocityLength.w;return dt/params.cellGravity.xyz[axis]*normalVelocity*fraction;
}
fn applyInflowVelocity(q:vec3i,inputVelocity:vec3f)->vec3f{
  let strength=inflowStrength();if(strength<=0.0){return inputVelocity;}let axis=inflowAxis();let receiver=inflowReceiverIndex(axis);let donor=select(receiver+1,receiver-1,params.inflowVelocityLength[axis]>=0.0);if(q[axis]!=receiver&&q[axis]!=donor){return inputVelocity;}let fraction=inflowApertureFraction(q);if(fraction<=0.0){return inputVelocity;}var velocity=inputVelocity;let desiredVelocity=params.inflowVelocityLength.xyz*strength;if(q[axis]==receiver){velocity=desiredVelocity;}velocity[axis]=desiredVelocity[axis];return velocity;
}
`;

export interface InflowGridBoundary {
  axis: 0 | 1 | 2;
  direction: Vec3;
  outletCenter_m: Vec3;
  faceIndex: number;
  receiverIndex: number;
  apertureScale: number;
  rawProjectedArea_m2: number;
  targetProjectedArea_m2: number;
  flowRate_m3_s: number;
  cellSize_m: Vec3;
  dims: readonly [number, number, number];
}

const components = (value: Vec3) => [value.x, value.y, value.z] as const;

function dominantAxis(direction: Vec3): 0 | 1 | 2 {
  const magnitude = components(direction).map(Math.abs);
  return magnitude[1] > magnitude[0] && magnitude[1] >= magnitude[2] ? 1 : magnitude[2] > magnitude[0] ? 2 : 0;
}

export function inflowOutletCenter(inflow: FluidInflow): Vec3 {
  const speed = Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z);
  if (!(speed > 0)) return { ...inflow.center_m };
  const halfLengthOverSpeed = 0.5 * inflow.length_m / speed;
  return {
    x: inflow.center_m.x + inflow.velocity_m_s.x * halfLengthOverSpeed,
    y: inflow.center_m.y + inflow.velocity_m_s.y * halfLengthOverSpeed,
    z: inflow.center_m.z + inflow.velocity_m_s.z * halfLengthOverSpeed
  };
}

function apertureCoverage(point: readonly [number, number, number], outlet: Vec3, direction: Vec3, radius_m: number, edgeWidth_m: number) {
  const relative = [point[0] - outlet.x, point[1] - outlet.y, point[2] - outlet.z] as const;
  const axial = relative[0] * direction.x + relative[1] * direction.y + relative[2] * direction.z;
  const radialX = relative[0] - axial * direction.x;
  const radialY = relative[1] - axial * direction.y;
  const radialZ = relative[2] - axial * direction.z;
  const radialDistance = Math.hypot(radialX, radialY, radialZ);
  const coverage = Math.max(0, Math.min(1, 0.5 + 0.5 * (radius_m - radialDistance) / Math.max(edgeWidth_m, 1e-6)));
  return coverage * coverage * (3 - 2 * coverage);
}

/**
 * Converts the analytic circular outlet into one normalized Cartesian face
 * aperture. The disk projects to an ellipse on the dominant-axis face. A
 * cell-width analytic edge coverage keeps the aperture smooth without
 * shader-side quadrature; apertureScale makes its integrated flux exactly
 * pi r^2 |u| independent of grid orientation.
 */
export function createInflowGridBoundary(
  inflow: FluidInflow,
  container: SceneDescription["container"],
  dims: readonly [number, number, number]
): InflowGridBoundary {
  const size = [container.width_m, container.height_m, container.depth_m] as const;
  const minimum = [-0.5 * container.width_m, 0, -0.5 * container.depth_m] as const;
  const cell = size.map((value, axis) => value / dims[axis]) as unknown as [number, number, number];
  const velocity = components(inflow.velocity_m_s);
  const speed = Math.hypot(...velocity);
  const direction: Vec3 = speed > 0
    ? { x: velocity[0] / speed, y: velocity[1] / speed, z: velocity[2] / speed }
    : { x: 1, y: 0, z: 0 };
  const directionComponents = components(direction);
  const axis = dominantAxis(direction);
  const outlet = inflowOutletCenter(inflow);
  const outletComponents = components(outlet);
  const faceIndex = Math.max(0, Math.min(dims[axis] - 2, Math.round((outletComponents[axis] - minimum[axis]) / cell[axis]) - 1));
  const receiverIndex = directionComponents[axis] >= 0 ? faceIndex + 1 : faceIndex;
  const tangents = [0, 1, 2].filter((candidate) => candidate !== axis) as [number, number];
  const faceArea = cell[tangents[0]] * cell[tangents[1]];
  const edgeWidth = 0.5 * Math.hypot(cell[tangents[0]], cell[tangents[1]]);
  let weightSum = 0;
  for (let second = 0; second < dims[tangents[1]]; second += 1) {
    for (let first = 0; first < dims[tangents[0]]; first += 1) {
      const point: [number, number, number] = [0, 0, 0];
      point[axis] = minimum[axis] + (faceIndex + 1) * cell[axis];
      point[tangents[0]] = minimum[tangents[0]] + (first + 0.5) * cell[tangents[0]];
      point[tangents[1]] = minimum[tangents[1]] + (second + 0.5) * cell[tangents[1]];
      weightSum += apertureCoverage(point, outlet, direction, inflow.radius_m, edgeWidth);
    }
  }
  const rawProjectedArea_m2 = weightSum * faceArea;
  const targetProjectedArea_m2 = speed > 0 ? Math.PI * inflow.radius_m ** 2 / Math.max(Math.abs(directionComponents[axis]), 1e-6) : 0;
  const apertureScale = rawProjectedArea_m2 > 0 ? targetProjectedArea_m2 / rawProjectedArea_m2 : 0;
  return {
    axis, direction, outletCenter_m: outlet, faceIndex, receiverIndex, apertureScale,
    rawProjectedArea_m2, targetProjectedArea_m2,
    flowRate_m3_s: Math.PI * inflow.radius_m ** 2 * speed,
    cellSize_m: { x: cell[0], y: cell[1], z: cell[2] }, dims
  };
}

export function integratedInflowVolume(inflow: FluidInflow, from_s: number, to_s: number, steps = 4096) {
  if (!(to_s > from_s)) return 0;
  const dt = (to_s - from_s) / steps;
  const flowRate = Math.PI * inflow.radius_m ** 2 * Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z);
  let integral = 0;
  for (let step = 0; step < steps; step += 1) {
    const time = from_s + (step + 0.5) * dt;
    integral += flowRate * inflowStrength(time, inflow.start_s, inflow.end_s, inflow.ramp_s) * dt;
  }
  return integral;
}

export function averageInflowStrength(inflow: FluidInflow, from_s: number, to_s: number) {
  if (!(to_s > from_s)) return 0;
  const breakpoints = [from_s, to_s, inflow.start_s, inflow.end_s, inflow.start_s + inflow.ramp_s, inflow.end_s - inflow.ramp_s, 0.5 * (inflow.start_s + inflow.end_s)]
    .filter((time) => time >= from_s && time <= to_s)
    .sort((a, b) => a - b)
    .filter((time, index, values) => index === 0 || time !== values[index - 1]);
  let integral = 0;
  for (let index = 1; index < breakpoints.length; index += 1) {
    const a = breakpoints[index - 1], b = breakpoints[index];
    integral += 0.5 * (inflowStrength(a, inflow.start_s, inflow.end_s, inflow.ramp_s) + inflowStrength(b, inflow.start_s, inflow.end_s, inflow.ramp_s)) * (b - a);
  }
  return integral / (to_s - from_s);
}
