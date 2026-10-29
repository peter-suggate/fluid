"use client";
import { RailRange, RailToggle } from "../../../../components/rail";
import { useSession } from "../../../core/session/session-context";
export function VoxelSpecularControls(){
  const session=useSession();
  const enabled=session.ui(s=>s.svoVoxelSpecularReflectionsEnabled);
  const cones=session.ui(s=>s.svoConeTracingMode==="cones"),gi=session.ui(s=>s.svoGlobalIlluminationEnabled);
  const bodies=session.scene(s=>s.scene.rigidBodies.length>0);
  const available=cones&&gi&&!bodies;
  const steps=session.ui(s=>s.svoVoxelSpecularSteps);
  const setEnabled=session.ui(s=>s.setSvoVoxelSpecularReflectionsEnabled);
  const setSteps=session.ui(s=>s.setSvoVoxelSpecularSteps);
  return <>
    <RailToggle label="Voxel reflections" checked={enabled} onChange={setEnabled} testId="voxel-specular"
      hint="Reflect scene radiance on opaque voxel surfaces. Enabling selects Cones and GI; both must remain enabled. Supports static voxel scenes; scenes with rigid bodies retain sky reflections. Fine mirror detail is limited by the radiance voxel grid." />
    {enabled&&!available&&<p className="tool-options-note" role="status">{bodies?"Withheld: rigid-body scenes retain sky reflections.":"Withheld: select Cones and enable Global illumination."}</p>}
    {enabled&&<RailRange disabled={!available} label="Reflection steps" value={steps} min={1} max={512} step={1} digits={0} onChange={setSteps}
      hint="Maximum samples per reflection cone. Narrow smooth reflections travel farther with higher budgets; exhausted traces retain partial radiance and remaining sky." />}
  </>;
}
