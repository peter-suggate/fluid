"use client";
import { RailRange, RailToggle } from "../../../components/rail";
import { useSession } from "../../core/session/session-context";

export function LensBloomControls(){
  const session=useSession();
  const enabled=session.ui(s=>s.lensBloomEnabled),strength=session.ui(s=>s.lensBloomStrength);
  const threshold=session.ui(s=>s.lensBloomThreshold),radius=session.ui(s=>s.lensBloomRadius);
  const set=session.ui(s=>s.setLensBloom);
  return <>
    <RailToggle label="Lens bloom" checked={enabled} onChange={lensBloomEnabled=>set({lensBloomEnabled})}
      hint="In Shaded view, spread bright scene-linear light before display tone mapping. Geometry and fluid stay unchanged." testId="lens-bloom" />
    {enabled&&<>
      <RailRange label="Bloom strength" value={strength*100} min={0} max={100} step={1} digits={0} unit="%" onChange={v=>set({lensBloomStrength:v/100})} />
      <RailRange label="Bloom threshold" value={threshold} min={0} max={64} step={.1} digits={1} onChange={lensBloomThreshold=>set({lensBloomThreshold})} hint="Scene-linear luminance above which light spreads." />
      <RailRange label="Bloom radius" value={radius*100} min={.2} max={15} step={.1} digits={1} unit="%" onChange={v=>set({lensBloomRadius:v/100})} hint="Fraction of the shorter image dimension." />
    </>}
  </>;
}
