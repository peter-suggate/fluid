import { readFileSync } from "node:fs";
import { getSceneDefinition } from "./lib/core/scenes";
import { sceneDocument } from "./lib/core/scene-definition";
import { integratedInflowVolume } from "./lib/core/inflow-boundary";
import { sceneLatticeDimensions } from "./lib/core/scene-lattice-dimensions";
const C = process.env.HOME + "/code/me/fluid-scratch/coord/all/";
const rows = (f: string, old: boolean) => { const m = new Map<string, any>(); for (const l of readFileSync(C + f, "utf8").split("\n")) { const j = l.indexOf("{"); if (j < 0 || (l.slice(0, j).includes(" OLD ") !== old)) continue; try { const d = JSON.parse(l.slice(j)); m.set(d.id, d); } catch {} } return m; };
const b = rows("b.out", false), o = rows("old-pool.out", true);
for (const id of ["nbflip-ramp-channel","nbflip-ramp-staircase","nbflip-ramp-flat","nbflip-water-slide","nbflip-water-slide-long","nbflip-water-slide-theme-park","hero-garden-hose","garden-hose","nbflip-figure-7-pour"]) {
  const s = sceneDocument(getSceneDefinition(id)); const n = sceneLatticeDimensions(s); const h = s.container.width_m / n[0]!;
  const inflow = (s as any).fluid.inflow ?? (s as any).inflow;
  let want = NaN; try { want = integratedInflowVolume(inflow, 0, 1) / h ** 3; } catch (e) { want = NaN; }
  console.log(id.padEnd(32), "brick V1", b.get(id)?.V1?.toFixed?.(0), "old V1", o.get(id)?.V1?.toFixed?.(0), "analytic 1s cells", want.toFixed(0), "dt", s.numerics.fixedDt_s);
}
