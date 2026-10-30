/** Detached coarse mass: a coarse owner whose pressure authority is air, that
 * holds more than a twentieth of an h cell of V, and that touches no
 * pressure-liquid owner. A thin h sheet inside a 4h owner whose centre is air
 * is such mass: neither a pressure row nor beside one, so zeroing its air
 * faces froze up to half a 4h owner of liquid in place (sheets stuck to the
 * ceiling and walls). Its interior faces keep their forced velocity instead:
 * free flight, the native airborne model, without the native centre-distance
 * and wall-margin tests, which scaled by owner width exclude every owner
 * within 8 cells of a wall. Owners touching liquid stay excluded: ballistic
 * faces on a contact film are not divergence free. So is a cut owner standing
 * on solid (umTileSupported): its V rests there, and free flight would run its
 * faces into the solid. So is an owner resting on a SOLID or BULK16 page
 * (a container wall, or deep water): its face neighbour cell along
 * gravity's dominant axis lies in one. umTileSupported answers for cut tiles
 * only, and a 4h film on an uncut floor tile in free flight ran into the
 * floor at g (9.9 m/s after 60 steps, tests/uniform-coarse-free-fall.test.ts);
 * BULK16 water below is liquid that is no face neighbour. Without gravity
 * nothing rests. Unit owners keep the native rule. Needs umFace, umCellKind
 * and the solid helpers; liquid(owner) and volume(owner) are WGSL
 * expressions, support a vec4i one (uniformMixedGravitySupport). */
export function uniformMixedDetachedMassWGSL(liquid: (owner: string) => string, volume: (owner: string) => string, dust: string, support: string): string {
 return /* wgsl */ `
fn umDetachedMass(o:UMOwner)->bool{
 if(o.width<2u){return false;}
 if(${volume("o")}*f32(o.width*o.width*o.width)<=max(${dust},0.05)){return false;}
 if(${liquid("o")}||umTileSupported(o.tile)){return false;}
 let s=${support};
 if(s.w!=0&&umCellKind(vec3i(umOrigin(o))+select(s.xyz,s.xyz*i32(o.width),s.xyz>vec3i(0)))>=UM_SOLID){return false;}
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);let first=umFace(o,axis,sign,0u);
  for(var part=0u;part<first.count;part++){let n=umFace(o,axis,sign,part).neighbor;if(n.width!=0u&&${liquid("n")}){return false;}}
 }}
 return true;
}`;
}

/** The step to the cell an owner rests on: a unit step along gravity's
 * dominant axis (x before y before z on a tie), w 1; all zero without
 * gravity. Written into `out` (four i32 words) for umDetachedMass's support. */
export function uniformMixedGravitySupport(gravity: readonly [number, number, number] | ArrayLike<number>, out: Int32Array): Int32Array {
 out.fill(0);
 let axis = -1, best = 0;
 for (let a = 0; a < 3; a++) { const g = Math.abs(gravity[a]!); if (g > best) { best = g; axis = a; } }
 if (axis >= 0) { out[axis] = Math.sign(gravity[axis]!); out[3] = 1; }
 return out;
}
