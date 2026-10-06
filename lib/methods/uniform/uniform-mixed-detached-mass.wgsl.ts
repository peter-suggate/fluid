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
 * faces into the solid. Unit owners keep the native rule. Needs umFace and
 * the solid helpers; liquid(owner) and volume(owner) are WGSL expressions. */
export function uniformMixedDetachedMassWGSL(liquid: (owner: string) => string, volume: (owner: string) => string, dust: string): string {
 return /* wgsl */ `
fn umDetachedMass(o:UMOwner)->bool{
 if(o.width<2u){return false;}
 if(${volume("o")}*f32(o.width*o.width*o.width)<=max(${dust},0.05)){return false;}
 if(${liquid("o")}||umTileSupported(o.tile)){return false;}
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);let first=umFace(o,axis,sign,0u);
  for(var part=0u;part<first.count;part++){let n=umFace(o,axis,sign,part).neighbor;if(n.width!=0u&&${liquid("n")}){return false;}}
 }}
 return true;
}`;
}
