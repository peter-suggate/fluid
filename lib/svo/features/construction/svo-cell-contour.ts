/**
 * Native-cell Laine variant. A cube supplies the lower slab support plane;
 * the upper support is quantized outward in the packed geometry's high byte.
 * Zero is absent. The normal is the cell's separately stored oct8 normal.
 * This bounds solid volume, not an infinitesimal surface or a dual contour.
 */
export const SVO_CELL_CONTOUR = Object.freeze({ shift: 24, levels: 255, subdivisions: 4 });

export type ContourPoint = readonly [number, number, number];
const dot = (a: ContourPoint, b: ContourPoint) => a.reduce((s, v, i) => s + v * b[i], 0);
const centred = (p: ContourPoint): ContourPoint => [p[0] - .5, p[1] - .5, p[2] - .5];

/** Mirror of the outward offset encoding, including a coordinate-rounding pad. */
export function encodeCellContourSupport(support: number, radius: number): number {
  if (!Number.isFinite(support) || !(radius > 0) || !Number.isFinite(radius)) return 0;
  const q = Math.max(1, Math.min(255, Math.ceil((support / radius + 1) * 127.5) + 1));
  return q < 255 ? q : 0;
}

/** Clip a convex polygon to dot(n,p-.5) <= high. Used by the geometry oracle. */
export function clipCellContourPolygon(points: readonly ContourPoint[], normal: ContourPoint, high: number): ContourPoint[] {
  const out: ContourPoint[] = [];
  for (let i = 0; i < points.length; i++) {
    const a = points[i], b = points[(i + 1) % points.length];
    const da = dot(normal, centred(a)) - high;
    const db = dot(normal, centred(b)) - high;
    if (da <= 0) out.push(a);
    if ((da < 0 && db > 0) || (da > 0 && db < 0)) {
      const t = da / (da - db);
      out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
    }
  }
  return out;
}

/** No scene bindings: shared by the mesh builder and its GPU geometry oracle. */
export const svoCellContourWGSL = /* wgsl */ `
struct CellContour { normal:vec3f, high:f32, valid:u32 }
struct ContourPolygon { points:array<vec3f,12>, count:u32 }
fn cellContour(normal:vec3f,cellSize:vec3f,code:u32)->CellContour{
  let n=normal*cellSize;let magnitude=length(n);
  if(code==0u||code>=255u||!(magnitude>1e-12)){return CellContour(vec3f(0.0),0.0,0u);}
  let unit=n/magnitude;
  return CellContour(unit,0.5*dot(abs(unit),vec3f(1.0))*(f32(code)*(2.0/255.0)-1.0),1u);
}
fn contourCubeFace(face:u32)->ContourPolygon{
  let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let corners=array<vec2f,4>(vec2f(0,0),vec2f(1,0),vec2f(1,1),vec2f(0,1));
  var polygon:ContourPolygon;polygon.count=4u;
  for(var i=0u;i<4u;i+=1u){
    let c=select(corners[i].yx,corners[i],(face&1u)!=0u);
    var p=vec3f(0.0);p[axis]=f32(face&1u);p[u]=c.x;p[v]=c.y;polygon.points[i]=p;
  }
  return polygon;
}
fn contourClipPolygon(polygon:ContourPolygon,contour:CellContour)->ContourPolygon{
  var out:ContourPolygon;
  for(var i=0u;i<polygon.count;i+=1u){
    let a=polygon.points[i];let b=polygon.points[(i+1u)%polygon.count];
    let da=dot(contour.normal,a-vec3f(0.5))-contour.high;
    let db=dot(contour.normal,b-vec3f(0.5))-contour.high;
    if(da<=0.0){out.points[out.count]=a;out.count+=1u;}
    if((da<0.0&&db>0.0)||(da>0.0&&db<0.0)){
      out.points[out.count]=mix(a,b,da/(da-db));out.count+=1u;
    }
  }
  return out;
}
fn contourCap(contour:CellContour)->ContourPolygon{
  var out:ContourPolygon;
  // Each cube edge once. An exact vertex is deduplicated across its edges.
  for(var axis=0u;axis<3u;axis+=1u){for(var edge=0u;edge<4u;edge+=1u){
    var a=vec3f(0.0);a[(axis+1u)%3u]=f32(edge&1u);a[(axis+2u)%3u]=f32(edge>>1u);
    var b=a;b[axis]=1.0;
    let da=dot(contour.normal,a-vec3f(0.5))-contour.high;
    let db=dot(contour.normal,b-vec3f(0.5))-contour.high;
    if(da*db>0.0||abs(da-db)<1e-10){continue;}
    let p=mix(a,b,clamp(da/(da-db),0.0,1.0));var duplicate=false;
    for(var i=0u;i<out.count;i+=1u){duplicate=duplicate||distance(p,out.points[i])<1e-6;}
    if(!duplicate){out.points[out.count]=p;out.count+=1u;}
  }}
  if(out.count<3u){return out;}
  var centre=vec3f(0.0);for(var i=0u;i<out.count;i+=1u){centre+=out.points[i];}centre/=f32(out.count);
  let helper=select(vec3f(1,0,0),vec3f(0,1,0),abs(contour.normal.x)>0.8);
  let u=normalize(cross(helper,contour.normal));let v=cross(contour.normal,u);
  var angles:array<f32,12>;
  for(var i=0u;i<out.count;i+=1u){let p=out.points[i]-centre;angles[i]=atan2(dot(p,v),dot(p,u));}
  for(var i=1u;i<out.count;i+=1u){var j=i;
    while(j>0u){if(angles[j]>=angles[j-1u]){break;}
      let angle=angles[j];angles[j]=angles[j-1u];angles[j-1u]=angle;
      let p=out.points[j];out.points[j]=out.points[j-1u];out.points[j-1u]=p;j-=1u;
    }
  }
  return out;
}
fn contourPackPoint(p:vec3f)->u32{
  let q=vec3u(round(clamp(p,vec3f(0.0),vec3f(1.0))*1023.0));
  return q.x|(q.y<<10u)|(q.z<<20u);
}
fn contourUnpackPoint(word:u32)->vec3f{return vec3f(f32(word&1023u),f32((word>>10u)&1023u),f32((word>>20u)&1023u))/1023.0;}
`;
