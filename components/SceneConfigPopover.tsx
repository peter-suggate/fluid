"use client";

import { useRef, useState } from "react";
import { NumberField, RangeControl, Segmented } from "./controls";
import { MethodPanel } from "./MethodPanel";
import { SceneLibraryPanel } from "./SceneLibraryPanel";
import { simulation } from "@/lib/simulation/controller";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";
import type { FluidInflow } from "@/lib/model";

const defaultInflow: FluidInflow = {
  center_m: { x: -0.4, y: 0.55, z: 0 }, radius_m: 0.08, length_m: 0.12,
  velocity_m_s: { x: 0.5, y: 0, z: 0 }, start_s: 0, end_s: 12, ramp_s: 0.3
};

const sections = [
  { id: "scene", label: "Scene", hint: "Identity and seed" },
  { id: "method", label: "Method", hint: "Solver, quality, and parameters" },
  { id: "container", label: "Container", hint: "Extents, walls, and the voxel domain" },
  { id: "fluid", label: "Fluid", hint: "Material properties and the inflow jet" },
  { id: "numerics", label: "Numerics", hint: "Timestep rates and solver budgets" },
  { id: "library", label: "My scenes", hint: "Saved authored scenes" },
] as const;
type SectionId = (typeof sections)[number]["id"];

const degrees = (radians: number) => Math.round(radians * (180 / Math.PI) * 10) / 10;
const radians = (deg: number) => deg * (Math.PI / 180);

/**
 * Non-blocking scene configuration. Edits patch the scene store immediately:
 * the GPU solver rebuilds live from its config key, so the viewport shows the
 * effect while the popover stays open. "Apply & reset" restarts the clock and
 * re-initializes rigid bodies and the CPU reference for a clean run.
 *
 * A section rail keeps every authored knob one click away without turning the
 * body into one long scroll, which is what it became once the method controls
 * moved here from the old sidebar.
 */
export function SceneConfigPopover() {
  const open = useUIStore((state) => state.sceneModalOpen);
  const setOpen = useUIStore((state) => state.setSceneModalOpen);
  const scene = useSceneStore((state) => state.scene);
  const patchScene = useSceneStore((state) => state.patchScene);
  const patchContainer = useSceneStore((state) => state.patchContainer);
  const patchFluid = useSceneStore((state) => state.patchFluid);
  const patchNumerics = useSceneStore((state) => state.patchNumerics);
  const fixedRate_hz = 1 / scene.numerics.fixedDt_s;
  const gpuStepRate_hz = 1 / scene.numerics.maxDt_s;
  const [section, setSection] = useState<SectionId>("scene");
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const dragRef = useRef<{ pointerId: number; startX: number; startY: number; baseX: number; baseY: number } | null>(null);
  if (!open) return null;

  const inflow = scene.fluid.inflow;
  const voxelDomain = scene.voxelDomain;
  const fluidEnabled = scene.systems?.fluid !== false;
  const voxelDimensions = [scene.container.width_m, scene.container.height_m, scene.container.depth_m]
    .map((extent) => Math.max(8, Math.round(extent / voxelDomain.finestCellSize_m)));
  const patchVoxelDomain = (patch: Partial<typeof voxelDomain>) => patchScene({ voxelDomain: { ...voxelDomain, ...patch } });
  const patchInflow = (patch: Partial<FluidInflow>) => patchFluid({ inflow: { ...(inflow ?? defaultInflow), ...patch } });
  // The jet's direction is carried by its velocity vector (the solvers orient
  // the injection cylinder along it), edited here as speed + pitch + yaw.
  const jetVelocity = inflow?.velocity_m_s ?? defaultInflow.velocity_m_s;
  const jetSpeed = Math.hypot(jetVelocity.x, jetVelocity.y, jetVelocity.z);
  const jetPitchDeg = jetSpeed > 1e-9 ? degrees(Math.asin(Math.max(-1, Math.min(1, jetVelocity.y / jetSpeed)))) : 0;
  const jetYawDeg = jetSpeed > 1e-9 ? degrees(Math.atan2(jetVelocity.z, jetVelocity.x)) : 0;
  const patchJet = (speed: number, pitchDeg: number, yawDeg: number) => {
    const pitch = radians(Math.max(-90, Math.min(90, pitchDeg))), yaw = radians(yawDeg), magnitude = Math.max(0, speed);
    patchInflow({ velocity_m_s: { x: magnitude * Math.cos(pitch) * Math.cos(yaw), y: magnitude * Math.sin(pitch), z: magnitude * Math.cos(pitch) * Math.sin(yaw) } });
  };
  const headerDown = (event: React.PointerEvent<HTMLElement>) => {
    if ((event.target as HTMLElement).closest("button")) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, baseX: offset.x, baseY: offset.y };
  };
  const headerMove = (event: React.PointerEvent<HTMLElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    setOffset({ x: drag.baseX + event.clientX - drag.startX, y: drag.baseY + event.clientY - drag.startY });
  };
  const headerUp = (event: React.PointerEvent<HTMLElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
  };

  return (
    <section className="scene-popover" role="dialog" aria-label="Scene configuration" data-testid="scene-settings-modal" style={{ transform: `translate(${offset.x}px, ${offset.y}px)` }}>
      <header onPointerDown={headerDown} onPointerMove={headerMove} onPointerUp={headerUp} onPointerCancel={headerUp} title="Drag to move">
        <div><p className="eyebrow">SCENE CONFIGURATION · LIVE</p><h2>{scene.sceneId}</h2></div>
        <button className="icon-button" onClick={() => setOpen(false)} aria-label="Close scene configuration">×</button>
      </header>
      <div className="popover-shell">
        <nav className="popover-tabs" aria-label="Configuration sections">
          {sections.map((entry) => (
            <button
              key={entry.id}
              type="button"
              title={entry.hint}
              className={section === entry.id ? "active" : ""}
              aria-current={section === entry.id}
              onClick={() => setSection(entry.id)}
            >{entry.label}</button>
          ))}
        </nav>
        <div className="popover-body">
          {section === "scene" && <section>
            <h3>Identity</h3>
            <label className="text-field"><span>Scene ID</span><input aria-label="Scene name" value={scene.sceneId} onChange={(event) => patchScene({ sceneId: event.target.value })} /></label>
            <NumberField label="Random seed" value={scene.randomSeed} step={1} min={0} onChange={(value) => patchScene({ randomSeed: Math.max(0, Math.round(value)) })} />
          </section>}
          {section === "method" && <MethodPanel />}
          {section === "container" && <>
            <section>
              <h3>Container</h3>
              <RangeControl label="Width" unit="m" value={scene.container.width_m} min={0.4} max={2.5} step={0.05} onChange={(value) => patchContainer({ width_m: value })} displayDigits={2} />
              <RangeControl label="Height" unit="m" value={scene.container.height_m} min={0.4} max={20} step={0.05} onChange={(value) => patchContainer({ height_m: value })} displayDigits={2} />
              <RangeControl label="Depth" unit="m" value={scene.container.depth_m} min={0.4} max={2} step={0.05} onChange={(value) => patchContainer({ depth_m: value })} displayDigits={2} />
              <RangeControl label="Water fill" unit="%" value={scene.container.fillFraction * 100} min={5} max={90} step={1} onChange={(value) => patchContainer({ fillFraction: value / 100 })} displayDigits={0} />
              <div className="field-grid">
                <Segmented ariaLabel="Container top" value={scene.container.top} options={[{ value: "open", label: "Open top" }, { value: "closed", label: "Closed" }]} onChange={(value) => patchContainer({ top: value })} />
                <Segmented ariaLabel="Fluid wall condition" value={scene.container.fluidWallMode} options={[{ value: "no-slip", label: "No slip" }, { value: "free-slip", label: "Free slip" }]} onChange={(value) => patchContainer({ fluidWallMode: value })} />
              </div>
              {/* Whether the domain is *drawn* as a tank, which is separate from
                  the boundary it always is. A fresh scene starts as a room, so
                  this is how its author asks for the vessel. */}
              <Segmented ariaLabel="Container vessel" value={scene.container.vessel ?? "glass"} options={[{ value: "glass", label: "Glass tank" }, { value: "none", label: "No tank" }]} onChange={(value) => patchContainer({ vessel: value })} />
            </section>
            <section data-testid="voxel-domain-controls">
              <h3>Unified voxel domain</h3>
              <div className="field-grid">
                <NumberField label="Finest cell" unit="m" value={voxelDomain.finestCellSize_m} step={0.005} min={0.01} max={0.25} onChange={(value) => patchVoxelDomain({ finestCellSize_m: Math.max(0.01, value) })} />
              </div>
              <Segmented ariaLabel="Sparse voxel brick size" value={String(voxelDomain.brickSize_cells)} options={[{ value: "4", label: "4³-cell leaves", disabled: fluidEnabled, title: fluidEnabled ? "4³ leaves require a renderer-only scene; fluid owner pages currently use 8³ bricks." : undefined }, { value: "8", label: "8³-cell leaves" }]} onChange={(value) => patchVoxelDomain({ brickSize_cells: value === "4" ? 4 : 8 })} />
              <small className="control-hint">Container lattice: {voxelDimensions.join(" × ")} finest cells. The sparse world grows automatically to include authored environment objects and optional scene bounds.{fluidEnabled ? " Fluid scenes require 8³ leaves; 4³ is available for renderer-only scenes." : ""}</small>
            </section>
          </>}
          {section === "fluid" && <>
            <section>
              <h3>Water</h3>
              {/* The gate `planSceneRuntime` reads. Off attaches the live sparse
                  scene instead of a solver, so a set can be looked at while its
                  water is still in bring-up — and everything below stays
                  authored, describing the pond that returns when it is on. */}
              <Segmented ariaLabel="Fluid system" value={fluidEnabled ? "on" : "off"} options={[{ value: "off", label: "Off" }, { value: "on", label: "On" }]} onChange={(value) => simulation.setFluidSystem(value === "on")} />
              <small className="control-hint">{fluidEnabled
                ? "The fluid solver owns this scene. Turning water off renders the set alone, with no solver to bring up and no transport."
                : "Renderer only: the set draws from the live sparse scene and nothing waits on a fluid authority. The settings below are still authored, and take effect when water is turned back on."}</small>
            </section>
            <section>
              <h3>Fluid</h3>
              <Segmented ariaLabel="Fluid initial condition" value={scene.fluid.initialCondition} options={[{ value: "dam-break", label: "Dam break" }, { value: "tank-fill", label: "Tank fill" }]} onChange={(value) => patchFluid({ initialCondition: value })} />
              <RangeControl label="Density" unit="kg/m³" value={scene.fluid.density_kg_m3} min={700} max={1300} step={0.1} onChange={(value) => patchFluid({ density_kg_m3: value })} displayDigits={1} />
              <RangeControl label="Dynamic viscosity" unit="Pa·s" value={scene.fluid.dynamicViscosity_Pa_s} min={0} max={0.02} step={0.000001} onChange={(value) => patchFluid({ dynamicViscosity_Pa_s: value })} displayDigits={6} />
              <RangeControl label="Surface tension" unit="N/m" value={scene.fluid.surfaceTension_N_m} min={0} max={0.15} step={0.001} onChange={(value) => patchFluid({ surfaceTension_N_m: value })} displayDigits={3} />
              <RangeControl label="Gravity Y" unit="m/s²" value={scene.fluid.gravity_m_s2.y} min={-20} max={0} step={0.01} onChange={(value) => patchFluid({ gravity_m_s2: { ...scene.fluid.gravity_m_s2, y: value } })} displayDigits={3} />
            </section>
            <section>
              <div className="inflow-heading"><h3>Inflow jet</h3><button className="quiet-button" onClick={() => inflow ? patchFluid({ inflow: undefined }) : patchInflow({})}>{inflow ? "Remove" : "Add inflow"}</button></div>
              {inflow && <div className="field-grid three">
                <NumberField label="Center X" unit="m" value={inflow.center_m.x} step={0.01} onChange={(value) => patchInflow({ center_m: { ...inflow.center_m, x: value } })} />
                <NumberField label="Center Y" unit="m" value={inflow.center_m.y} step={0.01} onChange={(value) => patchInflow({ center_m: { ...inflow.center_m, y: value } })} />
                <NumberField label="Center Z" unit="m" value={inflow.center_m.z} step={0.01} onChange={(value) => patchInflow({ center_m: { ...inflow.center_m, z: value } })} />
                <NumberField label="Speed" unit="m/s" value={Math.round(jetSpeed * 100) / 100} step={0.05} min={0} onChange={(value) => patchJet(value, jetPitchDeg, jetYawDeg)} />
                <NumberField label="Pitch" unit="°" value={jetPitchDeg} step={5} min={-90} max={90} onChange={(value) => patchJet(jetSpeed, value, jetYawDeg)} />
                <NumberField label="Yaw" unit="°" value={jetYawDeg} step={5} min={-180} max={180} onChange={(value) => patchJet(jetSpeed, jetPitchDeg, value)} />
                <NumberField label="Radius" unit="m" value={inflow.radius_m} step={0.005} min={0.01} onChange={(value) => patchInflow({ radius_m: value })} />
                <NumberField label="Length" unit="m" value={inflow.length_m} step={0.01} min={0.01} onChange={(value) => patchInflow({ length_m: value })} />
                <NumberField label="Ramp" unit="s" value={inflow.ramp_s} step={0.05} min={0} onChange={(value) => patchInflow({ ramp_s: value })} />
                <NumberField label="Start" unit="s" value={inflow.start_s} step={0.5} min={0} onChange={(value) => patchInflow({ start_s: value })} />
                <NumberField label="End" unit="s" value={inflow.end_s} step={0.5} min={0} onChange={(value) => patchInflow({ end_s: value })} />
              </div>}
            </section>
          </>}
          {section === "numerics" && <section>
            <h3>Timing &amp; numerics</h3>
            <div className="field-grid">
              <NumberField label="Rigid / oracle rate" unit="Hz" value={fixedRate_hz} step={0.01} min={1} onChange={(value) => { const dt = 1 / Math.max(1, value); patchNumerics({ fixedDt_s: dt, maxDt_s: Math.max(scene.numerics.maxDt_s, dt) }); }} />
              <NumberField label="GPU step cap" unit="Hz" value={gpuStepRate_hz} step={0.01} min={1} max={fixedRate_hz} onChange={(value) => patchNumerics({ maxDt_s: Math.max(scene.numerics.fixedDt_s, 1 / Math.max(1, value)) })} />
              <NumberField label="Oracle cell" unit="m" value={scene.nominalResolution.length_m} step={0.0025} min={0.0125} max={0.08} onChange={(value) => patchScene({ nominalResolution: { length_m: value } })} />
              <NumberField label="PCG budget" unit="iterations" value={scene.numerics.pressureMaxIterations} step={20} min={8} max={1000} onChange={(value) => patchNumerics({ pressureMaxIterations: Math.round(value) })} />
            </div>
            <small className="control-hint">The desired simulation clock stays at ×1. Rigid bodies and the CPU oracle step at {fixedRate_hz.toFixed(1)} Hz ({(scene.numerics.fixedDt_s * 1000).toFixed(2)} ms). GPU advances densely fill a measured 16.67 ms queue window, with an outer-step cap of {gpuStepRate_hz.toFixed(1)} Hz ({(scene.numerics.maxDt_s * 1000).toFixed(2)} ms).</small>
          </section>}
          {section === "library" && <SceneLibraryPanel />}
        </div>
      </div>
      <footer>
        <span className="popover-note">Container &amp; fluid edits apply live to the GPU solve · the CPU oracle, rigid bodies, and timing take effect on Apply &amp; reset</span>
        <button className="quiet-button" onClick={() => setOpen(false)}>Close</button>
        <button className="primary-button" onClick={() => simulation.applyAndResetFluid()}>Apply &amp; reset</button>
      </footer>
    </section>
  );
}
