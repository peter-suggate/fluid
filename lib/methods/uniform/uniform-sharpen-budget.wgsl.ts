/** Native geometric sharpening budgets (intensive cell-volume units).
 * Compaction may offer all liquid toward deeper phi; proposal restricts other
 * directions to surplus. Orphan mode gathers sub-half-full residue toward
 * greater density, while relay reception can exclude an empty air gap. */
export const uniformSharpenBudgetWGSL = /* wgsl */ `
fn uvSharpenBudgets(own:f32,desired:f32,phi:f32,h:f32,dose:f32,band:f32,compact:bool,orphanMode:f32,open:bool)->vec2f {
 let admitted=open&&select(abs(phi)<band*h,phi<band*h,compact);
 let relay=phi>0.0&&desired<=1e-6;
 let orphan=orphanMode>1.5&&open&&phi>=band*h;
 let receptive=orphanMode<0.5||own>1e-4||phi<h;
 let give=select(0.0,dose*select(select(max(own-desired,0.0),own,compact&&phi<0.0),select(0.0,own,own<0.5),orphan),admitted||orphan);
 let take=select(0.0,dose*max(select(desired,select(0.0,1.0,receptive),relay)-own,0.0),admitted||orphan);
 return vec2f(give,take);
}
`;
