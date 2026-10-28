/** Unpolarized dielectric power reflectance at either side of an interface.
 * eta is incident IOR / transmitted IOR. Unlike Schlick's entrance approximation,
 * this approaches one continuously at a denser-to-rarer critical angle. */
export function dielectricInterfaceReflectance(cosine: number, eta: number): number {
  if (!Number.isFinite(cosine) || !Number.isFinite(eta) || eta <= 0) throw new RangeError("Finite cosine and positive IOR ratio required");
  const c = Math.min(1, Math.max(0, cosine));
  if (Math.abs(eta - 1) < 1e-7) return 0;
  const k = 1 - eta * eta * (1 - c * c);
  if (k <= 0) return 1;
  const t = Math.sqrt(k);
  const s = (eta * c - t) / Math.max(eta * c + t, 1e-7);
  const p = (c - eta * t) / Math.max(c + eta * t, 1e-7);
  return Math.min(1, Math.max(0, .5 * (s * s + p * p)));
}

export const waterDielectricInterfaceWGSL = /* wgsl */ `
fn dielectricInterfaceReflectance(cosine:f32,eta:f32)->f32{
  let c=clamp(cosine,0.0,1.0);
  if(abs(eta-1.0)<1e-7){return 0.0;}
  let k=1.0-eta*eta*(1.0-c*c);
  if(k<=0.0){return 1.0;}
  let t=sqrt(k);
  let s=(eta*c-t)/max(eta*c+t,1e-7);
  let p=(c-eta*t)/max(c+eta*t,1e-7);
  return clamp(.5*(s*s+p*p),0.0,1.0);
}
`;
