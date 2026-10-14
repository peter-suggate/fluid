/** Serializable presentation surface for a flume rasterized on a coarser solver
 * lattice. Distances and positions are metres; curvature is inverse metres.
 * Each arc spans at most one revolution. Adjacent pieces share their endpoints.
 */
export interface SweptFlumeSurface {
  readonly kind: "swept-flume";
  readonly crossSection?: "semicircular";
  readonly halfWidth: number;
  readonly wallThickness: number;
  readonly wallHeight: number;
  readonly slab: number;
  readonly edgeRadius: number;
  readonly back: number;
  readonly stop: number;
  readonly pieces: readonly {
    readonly x: number; readonly z: number; readonly heading: number;
    readonly curvature: number; readonly length: number;
    readonly top: number; readonly fall: number;
    readonly wallEnd: number;
    readonly sideRise?: number;
    readonly hood: boolean;
  }[];
  readonly boxes: readonly { readonly min: readonly number[]; readonly max: readonly number[] }[];
}

export function sweptFlumeSurfaceErrors(s: SweptFlumeSurface): string[] {
  const errors: string[] = [];
  if (![s.halfWidth, s.wallThickness, s.wallHeight, s.slab, s.edgeRadius, s.back, s.stop].every(v => Number.isFinite(v) && v > 0)
    || s.edgeRadius > Math.min(s.slab, s.wallThickness) / 2) errors.push("Flume surface dimensions must be positive with a fitting edge radius");
  if (s.crossSection !== undefined && s.crossSection !== "semicircular") errors.push("Unknown flume cross-section");
  if (s.crossSection === "semicircular" && s.wallHeight < s.halfWidth + s.wallThickness) errors.push("Round flume sides must reach above the bowl");
  if (!Array.isArray(s.pieces) || !s.pieces.length || s.pieces.length > 64) errors.push("Flume surface needs 1..64 pieces");
  else for (const p of s.pieces) {
    if (![p.x, p.z, p.heading, p.curvature, p.length, p.top, p.fall, p.wallEnd].every(Number.isFinite)
      || (p.sideRise !== undefined && (!Number.isFinite(p.sideRise) || p.sideRise < 0))
      || p.length <= 0 || p.fall < 0 || p.wallEnd < 0 || p.wallEnd > p.length
      || Math.abs(p.curvature) * p.length > 2 * Math.PI + 1e-6 || typeof p.hood !== "boolean") errors.push("Invalid flume surface piece");
  }
  if (!Array.isArray(s.boxes) || s.boxes.length > 256) errors.push("Flume surface needs at most 256 support boxes");
  else for (const b of s.boxes) {
    if (b.min?.length !== 3 || b.max?.length !== 3 || !b.min.every((v: number, a: number) => Number.isFinite(v) && Number.isFinite(b.max[a]) && v < b.max[a]!)) errors.push("Invalid flume surface support box");
  }
  return errors;
}

function roundedRectangle(x: number, y: number, hx: number, hy: number, r: number): number {
  const qx = Math.abs(x) - hx + r, qy = Math.abs(y) - hy + r;
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r;
}

/** CPU oracle for cross-sections, seams and the generated shader. */
export function sweptFlumeDistance(s: SweptFlumeSurface, x: number, y: number, z: number): number {
  const reach = s.halfWidth + s.wallThickness, turn = 2 * Math.PI;
  let result = 1e20;
  s.pieces.forEach((p, index) => {
    const dx = x - p.x, dz = z - p.z, sin = Math.sin(p.heading), cos = Math.cos(p.heading);
    let across = -sin * dx + cos * dz, distance = cos * dx + sin * dz;
    if (p.curvature !== 0) {
      const cx = dx + sin / p.curvature, cz = dz - cos / p.curvature;
      const radius = 1 / Math.abs(p.curvature), sign = Math.sign(p.curvature);
      across = sign * (radius - Math.hypot(cx, cz));
      const angle = sign * (Math.atan2(cz, cx) - Math.atan2(-cos / p.curvature, sin / p.curvature));
      distance = ((angle % turn + turn) % turn) * radius;
    }
    for (let wrap = p.curvature ? -1 : 0; wrap <= (p.curvature ? 1 : 0); wrap++) {
      const d = distance + (p.curvature ? wrap * turn / Math.abs(p.curvature) : 0);
      const low = index === 0 ? -s.back : 0, high = p.length + (index === s.pieces.length - 1 ? s.stop : 0);
      // Tiny overlap prevents a zero-distance internal sheet where pieces meet.
      const cap = Math.max(low - d, d - high) - s.edgeRadius / 4;
      const floor = p.top - p.fall * Math.max(0, Math.min(1, d / p.length));
      let profile: number;
      if (s.crossSection === "semicircular" && !p.hood) {
        const rise = (p.sideRise ?? 0) * Math.sin(Math.PI * Math.max(0, Math.min(1, d / p.length))) ** 2;
        const cy = y - floor - s.halfWidth, radius = s.halfWidth + s.wallThickness / 2;
        // Distance to a single U-shaped centreline: circular bowl, tangent
        // sides and rounded lips, with no internal CSG seam at the join.
        const arm = (side: number) => Math.hypot(across - side * radius,
          Math.max(0, cy - (s.wallHeight + (side * p.curvature < 0 ? rise : 0) - s.halfWidth - s.wallThickness / 2)));
        profile = (cy < 0 ? Math.abs(Math.hypot(across, cy) - radius) : Math.min(arm(-1), arm(1))) - s.wallThickness / 2;
      } else {
        profile = roundedRectangle(across, y - floor + s.slab / 2, reach, s.slab / 2, s.edgeRadius);
        const wall = roundedRectangle(Math.abs(across) - s.halfWidth - s.wallThickness / 2,
          y - floor - (s.wallHeight - s.slab) / 2, s.wallThickness / 2, (s.wallHeight + s.slab) / 2, s.edgeRadius);
        profile = Math.min(profile, Math.max(wall, d - p.wallEnd));
      }
      if (index === 0) profile = Math.min(profile, Math.max(d, Math.abs(across) - reach, floor - s.slab - y, y - floor - s.wallHeight));
      if (p.hood) {
        const lid = p.top + s.wallHeight;
        profile = Math.min(profile, roundedRectangle(across, y - lid - s.wallThickness / 2, reach, s.wallThickness / 2, s.edgeRadius));
        profile = Math.min(profile, Math.max(p.length - d, Math.abs(across) - reach, floor - s.slab - y, y - lid - s.wallThickness));
      }
      result = Math.min(result, Math.max(cap, profile));
    }
  });
  for (const b of s.boxes) {
    const q = [x, y, z].map((v, a) => Math.abs(v - (b.min[a]! + b.max[a]!) / 2) - (b.max[a]! - b.min[a]!) / 2);
    result = Math.min(result, Math.hypot(...q.map(v => Math.max(v, 0))) + Math.min(Math.max(...q), 0));
  }
  return result;
}

const f = (v: number) => Number.isInteger(v) ? `${v}.0` : String(v);

/** Compile data, not scene IDs, into the existing editable authored-surface path. */
export function sweptFlumeSurfaceWGSL(s: SweptFlumeSurface): string {
  const errors = sweptFlumeSurfaceErrors(s);
  if (errors.length) throw new RangeError(errors.join("; "));
  return /* wgsl */ `
fn flumeRoundRect(p:vec2f,h:vec2f,r:f32)->f32{
  let q=abs(p)-h+vec2f(r);return length(max(q,vec2f(0.)))+min(max(q.x,q.y),0.)-r;
}
fn flumePiece(world:vec3f,start:vec4f,course:vec4f,ends:vec4f,sideRise:f32)->f32{
  let delta=world.xz-start.xy;let heading=start.z;let curvature=start.w;
  let reach=${f(s.halfWidth + s.wallThickness)};let r=${f(s.edgeRadius)};
  var across=dot(delta,vec2f(-sin(heading),cos(heading)));
  var along=dot(delta,vec2f(cos(heading),sin(heading)));
  var first=0;var last=0;var circumference=0.;
  if(curvature!=0.){
    let rel=delta+vec2f(sin(heading),-cos(heading))/curvature;
    let radius=1./abs(curvature);across=sign(curvature)*(radius-length(rel));
    let angle=sign(curvature)*(atan2(rel.y,rel.x)-atan2(-cos(heading)/curvature,sin(heading)/curvature));
    along=(angle-floor(angle/${f(2 * Math.PI)})*${f(2 * Math.PI)})*radius;
    circumference=${f(2 * Math.PI)}*radius;first=-1;last=1;
  }
  var result=1e20;
  for(var wrap=first;wrap<=last;wrap++){
    let d=along+f32(wrap)*circumference;let floorY=course.y-course.z*clamp(d/course.x,0.,1.);
    let cap=max(-ends.x-d,d-course.x-ends.y)-r*.25;
    var profile=0.;
    if(${s.crossSection === "semicircular" ? "true" : "false"} && ends.z==0.){
      let bend=sin(${f(Math.PI)}*clamp(d/course.x,0.,1.));let rise=sideRise*bend*bend;
      let cy=world.y-floorY-${f(s.halfWidth)};let radius=${f(s.halfWidth + s.wallThickness / 2)};
      let armTop=${f(s.wallHeight - s.halfWidth - s.wallThickness / 2)};
      let left=length(vec2f(across+radius,max(0.,cy-armTop-select(0.,rise,curvature>0.))));
      let right=length(vec2f(across-radius,max(0.,cy-armTop-select(0.,rise,curvature<0.))));
      profile=select(min(left,right),abs(length(vec2f(across,cy))-radius),cy<0.)-${f(s.wallThickness / 2)};
    }else{
    profile=flumeRoundRect(vec2f(across,world.y-floorY+${f(s.slab / 2)}),vec2f(reach,${f(s.slab / 2)}),r);
    let wall=flumeRoundRect(vec2f(abs(across)-${f(s.halfWidth + s.wallThickness / 2)},world.y-floorY-${f((s.wallHeight - s.slab) / 2)}),vec2f(${f(s.wallThickness / 2)},${f((s.wallHeight + s.slab) / 2)}),r);
    profile=min(profile,max(wall,d-course.w));
    }
    if(ends.x>0.){profile=min(profile,max(max(d,abs(across)-reach),max(floorY-${f(s.slab)}-world.y,world.y-floorY-${f(s.wallHeight)})));}
    if(ends.z>0.){
      let lid=course.y+${f(s.wallHeight)};
      profile=min(profile,flumeRoundRect(vec2f(across,world.y-lid-${f(s.wallThickness / 2)}),vec2f(reach,${f(s.wallThickness / 2)}),r));
      profile=min(profile,max(max(course.x-d,abs(across)-reach),max(floorY-${f(s.slab)}-world.y,world.y-lid-${f(s.wallThickness)})));
    }
    result=min(result,max(cap,profile));
  }
  return result;
}
fn swAuthoredDistance(world:vec3f)->f32{
  var result=1e20;
  ${s.pieces.map((p, i) => `result=min(result,flumePiece(world,vec4f(${[p.x, p.z, p.heading, p.curvature].map(f)}),vec4f(${[p.length, p.top, p.fall, p.wallEnd].map(f)}),vec4f(${[i === 0 ? s.back : 0, i === s.pieces.length - 1 ? s.stop : 0, p.hood ? 1 : 0, 0].map(f)}),${f(p.sideRise ?? 0)}));`).join("\n  ")}
  ${s.boxes.map(b => `{let q=abs(world-vec3f(${b.min.map((v, a) => f((v + b.max[a]!) / 2))}))-vec3f(${b.min.map((v, a) => f((b.max[a]! - v) / 2))});result=min(result,length(max(q,vec3f(0.)))+min(max(q.x,max(q.y,q.z)),0.));}`).join("\n  ")}
  return result;
}`;
}
