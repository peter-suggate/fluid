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
