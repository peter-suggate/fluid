import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";

const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
// This short regression covers the identified curvature/stencil faults through
// impact. The separate four-second verification command deliberately remains
// fail-closed on the later native-velocity asymmetry; see the investigation.
dawnTest("quarter pool preserves field and topology symmetry through impact", { timeout: 240_000 }, async () => {
  const output = await mkdtemp(join(tmpdir(), "cm12-quarter-symmetry-"));
  try {
    await promisify(execFile)(process.execPath, ["--import", "tsx", "tools/probe-pool-impact-symmetry-dawn.ts"], {
      env: { ...process.env, POOL_SYMMETRY_OUTPUT: output, POOL_SYMMETRY_STEPS: "18",
        POOL_SYMMETRY_DT: String(1/30), POOL_SYMMETRY_MAX_CELL: "0", POOL_SYMMETRY_OVERRIDES: "{}",
        POOL_SYMMETRY_FREEZE_TOPOLOGY: "0", POOL_SYMMETRY_LEGACY_FACE: "0", POOL_SYMMETRY_AUDIT_STEPS: "", POOL_SYMMETRY_VERIFY: "1" },
      timeout: 220_000, maxBuffer: 2*1024*1024,
    });
    const trace = JSON.parse(await readFile(join(output, "trace.json"), "utf8"));
    assert.equal(trace.length,19);
    for (const row of trace) {
      assert.ok(row.symmetry.topology.every((metric: { maximum: number }) => metric.maximum === 0),
        `step ${row.step}: accepted topology differs under a horizontal symmetry`);
    }
    assert.ok(trace[18].kinetic > 1, "the sphere must actually fall and impact");
    assert.ok(trace[18].histogram[4] > 0, "the test must exercise coarse/fine interfaces");
    assert.equal(JSON.parse(await readFile(join(output, "symmetry-verdict.json"), "utf8")).passed,true);
  } finally { await rm(output,{recursive:true,force:true}); }
});

// The full four-second command remains strict; this regression isolates the
// face-support faults before the later pressure-membership threshold split.
dawnTest("frozen quarter pool preserves symmetry through the first 52 steps", { timeout: 240_000 }, async () => {
  const output = await mkdtemp(join(tmpdir(), "cm12-frozen-quarter-symmetry-"));
  try {
    await promisify(execFile)(process.execPath, ["--import", "tsx", "tools/probe-pool-impact-symmetry-dawn.ts"], {
      env: { ...process.env, POOL_SYMMETRY_OUTPUT: output, POOL_SYMMETRY_STEPS: "52",
        POOL_SYMMETRY_DT: String(1/30), POOL_SYMMETRY_MAX_CELL: "0", POOL_SYMMETRY_OVERRIDES: "{}",
        POOL_SYMMETRY_FREEZE_TOPOLOGY: "1", POOL_SYMMETRY_LEGACY_FACE: "0",
        POOL_SYMMETRY_AUDIT_STEPS: "", POOL_SYMMETRY_VERIFY: "1" },
      timeout: 220_000, maxBuffer: 2*1024*1024,
    });
    const trace = JSON.parse(await readFile(join(output, "trace.json"), "utf8"));
    assert.equal(trace.length,53);
    for (const row of trace) {
      assert.equal(row.cells,trace[0].cells);
      assert.deepEqual(row.histogram,trace[0].histogram);
      assert.ok(row.symmetry.topology.every((metric: { maximum: number }) => metric.maximum === 0));
    }
    assert.ok(trace[52].kinetic>1);
    assert.equal(JSON.parse(await readFile(join(output,"configuration.json"),"utf8")).freezeTopology,true);
  } finally { await rm(output,{recursive:true,force:true}); }
});
