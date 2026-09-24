import assert from "node:assert/strict";
import test from "node:test";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("airborne-off symmetric expansion retains its column symmetry for 30 frames", { timeout: 600_000 }, async () => {
  // The probe owns the repository-wide GPU lease. Keep its stage readbacks so
  // the first failing stage remains available when this trajectory regresses.
  const { stdout } = await promisify(execFile)(process.execPath,
    ["--import","tsx",fileURLToPath(new URL("../tools/probe-uniform-airborne-expansion-dawn.ts",import.meta.url))], {
      cwd:fileURLToPath(new URL("../",import.meta.url)),maxBuffer:8*1024*1024,
      env:{...process.env,FLUID_AIRBORNE_AB_MODE:"off",FLUID_AIRBORNE_AB_FRAMES:"30",
        FLUID_AIRBORNE_ASSERT_SYMMETRY:"1",FLUID_AIRBORNE_SYM_DUMP:"0",
        FLUID_AIRBORNE_SYM_VALUES:"{}",FLUID_AIRBORNE_SYM_LINEAR_EXTENSION:"0"},
    });
  const rows=stdout.trim().split("\n").map(line=>JSON.parse(line));
  assert.equal(rows.length,31);
  assert.equal(rows.at(-1).frame,30);
  console.log(JSON.stringify({frame:30,meanVolumeSymmetryError:rows.at(-1).d4Mean,heightSymmetryError:rows.at(-1).heightD4}));
});

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("airborne expansion avoids compressed-air pressure jets through rebound", { timeout: 600_000 }, async () => {
  const { stdout } = await promisify(execFile)(process.execPath,
    ["--import","tsx",fileURLToPath(new URL("../tools/probe-uniform-airborne-expansion-dawn.ts",import.meta.url))], {
      cwd:fileURLToPath(new URL("../",import.meta.url)),maxBuffer:8*1024*1024,
      env:{...process.env,FLUID_AIRBORNE_AB_MODE:"on",FLUID_AIRBORNE_AB_FRAMES:"90",
        FLUID_AIRBORNE_ABLATION:"",FLUID_AIRBORNE_GRAVITY_STOP_FRAME:"Infinity",
        FLUID_AIRBORNE_SYM_DENSE_PRESSURE:"0",FLUID_AIRBORNE_SYM_DUMP:"0",
        FLUID_AIRBORNE_STATE_DUMP:"0",FLUID_AIRBORNE_SYM_VALUES:"{}",FLUID_AIRBORNE_SYM_LINEAR_EXTENSION:"0"},
    });
  const rows=stdout.trim().split("\n").map(line=>JSON.parse(line));
  assert.equal(rows.length,91);
  assert.equal(rows.at(-1).frame,90);
  // Scene-specific regression budgets, not universal incompressibility or
  // energy claims. Before pressure support: peak V=62.41, projection work
  // +922.06. With continuous support and surface preservation: V=3.30 and
  // every projection step removes net kinetic energy. Cover the whole rebound, since
  // sampling frame 30 alone misses both failures.
  for(const row of rows) {
    assert.ok(Number.isFinite(row.peakCellVolume)&&row.peakCellVolume<4,
      `frame ${row.frame}: compressed volume ${row.peakCellVolume}`);
    if(row.frame>0)assert.ok(Number.isFinite(row.energyStages.projectionChange)&&row.energyStages.projectionChange<=1e-3,
      `frame ${row.frame}: projection kinetic work ${row.energyStages.projectionChange}`);
    if(row.frame<=45){
      assert.equal(row.heightD4,0,`frame ${row.frame}: airborne column-height symmetry`);
      assert.ok(row.d4Mean<1e-4,`frame ${row.frame}: airborne mean volume symmetry error ${row.d4Mean}`);
    }
  }
  console.log(JSON.stringify({frames:90,
    peakCellVolume:Math.max(...rows.map(row=>row.peakCellVolume)),
    maxProjectionWork:Math.max(...rows.slice(1).map(row=>row.energyStages.projectionChange)),
    finalKinetic:rows.at(-1).faceKinetic}));
});
