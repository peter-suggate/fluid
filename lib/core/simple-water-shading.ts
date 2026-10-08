/**
 * The Simple water look: a murky blue body under a softly lit surface.
 *
 * Two passes draw it and have to agree. The water composite fogs the dry scene
 * by the liquid the view ray crosses; the particle spheres are drawn after it
 * and fog themselves by the liquid in front of them, then put the same surface
 * back over the result. Both call the functions below with display-encoded
 * colour, so a sphere just under the surface and the water beside it are one
 * material rather than a sprite pasted over a tint.
 */

/**
 * Distance over which the body takes hold, as a share of the tank's footprint
 * side. Short enough that a wall a few cells behind the surface is plainly
 * under water, long enough that a three-cell particle band reads through it.
 */
export const SIMPLE_WATER_MURK_LENGTH_FRACTION = 0.09;

export const simpleWaterMurkLength_m = (container_m: readonly [number, number, number]): number =>
  SIMPLE_WATER_MURK_LENGTH_FRACTION * Math.sqrt(Math.max(container_m[0] * container_m[2], 1e-12));

/**
 * The water composite's interface targets for the frame being drawn over:
 * world positions with a validity alpha for the nearest liquid interval and
 * the one peeled behind it, and the front interface's normal.
 */
export interface SimpleWaterInterfaces {
  readonly frontPosition: GPUTextureView;
  readonly frontNormal: GPUTextureView;
  readonly backPosition: GPUTextureView;
  readonly rearFrontPosition: GPUTextureView;
  readonly rearBackPosition: GPUTextureView;
}

export const simpleWaterShadingWGSL = /* wgsl */ `
const SIMPLE_WATER_BODY:vec3f=vec3f(.235,.455,.875);
// Red leaves first, so what shows through the water goes blue before it goes.
const SIMPLE_WATER_EXTINCTION:vec3f=vec3f(1.55,1.0,.62);
// The surface's own opacity: a sheet too thin to have a body is still seen.
const SIMPLE_WATER_FILM:f32=.26;
// A fixed key, high and over the left shoulder of the default view.
const SIMPLE_WATER_KEY:vec3f=vec3f(-.4581,.8144,.3563);

fn simpleWaterTransmittance(travel_m:f32,murkLength_m:f32)->vec3f{
  return exp(-SIMPLE_WATER_EXTINCTION*(max(travel_m,0.0)/max(murkLength_m,1e-6)));
}
// Liquid a view ray crosses before reach_m, from the two intervals the
// interface raster peels: [front, back] and the next one behind it. A rear
// distance that is not past its predecessor is an interface that is not there;
// a rear interval with no exit runs to the reach.
fn simpleWaterPath(reach_m:f32,front_m:f32,back_m:f32,rearFront_m:f32,rearBack_m:f32)->f32{
  var path=max(min(back_m,reach_m)-front_m,0.0);
  if(rearFront_m>=back_m){
    path+=max(select(reach_m,min(rearBack_m,reach_m),rearBack_m>rearFront_m)-rearFront_m,0.0);
  }
  return path;
}
// Half-Lambert against the key: folds stay readable on the side away from it.
fn simpleWaterRelief(normal:vec3f)->f32{
  let wrap=.5+.5*dot(normal,SIMPLE_WATER_KEY);
  return .5+.5*wrap*wrap;
}
// behind is what the ray reaches after travel_m of liquid, normal the front
// interface turned toward the viewer, rd the view ray.
fn simpleWaterOver(behind:vec3f,travel_m:f32,normal:vec3f,rd:vec3f,murkLength_m:f32)->vec3f{
  let relief=simpleWaterRelief(normal);
  let body=SIMPLE_WATER_BODY*relief;
  // Light reaches what is under the surface through that surface.
  var color=mix(body,behind*mix(1.0,relief,.5),simpleWaterTransmittance(travel_m,murkLength_m));
  color=mix(color,1.1*body,SIMPLE_WATER_FILM);
  let cosine=clamp(dot(normal,-rd),0.0,1.0);
  let grazing=1.0-cosine;
  let fresnel=.03+.97*grazing*grazing*grazing*grazing*grazing;
  let sky=mix(vec3f(.93,.95,1.0),vec3f(.62,.76,.98),clamp(reflect(rd,normal).y,0.0,1.0));
  color=mix(color,sky,.7*fresnel);
  let glint=max(dot(normal,normalize(SIMPLE_WATER_KEY-rd)),0.0);
  let glint8=pow(glint,8.0);
  // A broad sheen on the ridges and a tight sun on top of it.
  return color+vec3f(.10*glint8*glint8+.5*pow(glint8,12.0));
}
`;
