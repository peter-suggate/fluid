/**
 * Per-environment scenery check, so one scene can be art-directed without
 * running (or waiting on) the whole suite.
 *
 *   node --import tsx tools/check-scenery.ts garden
 *   node --import tsx tools/check-scenery.ts            # every environment
 *
 * Verifies the invariants a scenery module can plausibly break: determinism,
 * unique keys, dense owner order, finite bounds, light-record budget, and that
 * nothing has been parked inside the container where the water lives.
 */
import { environmentIds, type EnvironmentId } from "../lib/environments";
import { cloneScene, defaultScene } from "../lib/model";
import { buildSvoEnvironmentCoverage } from "../lib/svo-scene-coverage";
import { SVO_LIGHT_MAXIMUM_RECORDS } from "../lib/svo-light-abi";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";

const requested = process.argv.slice(2).filter((value) => !value.startsWith("-"));
const targets = (requested.length ? requested : [...environmentIds]) as EnvironmentId[];
for (const id of targets) {
  if (!environmentIds.includes(id)) throw new Error(`Unknown environment ${id}; expected one of ${environmentIds.join(", ")}`);
}

let failures = 0;
const fail = (id: string, message: string) => { failures++; console.error(`  ✗ ${id}: ${message}`); };

for (const id of targets) {
  const scene = cloneScene(defaultScene);
  const first = buildEnvironmentProxyCatalog(scene, id);
  const second = buildEnvironmentProxyCatalog(cloneScene(defaultScene), id);
  if (JSON.stringify(first) !== JSON.stringify(second)) fail(id, "catalog is not deterministic across rebuilds");

  const all = environmentProxyPrimitives(first);
  const keys = all.map((primitive) => primitive.key);
  const duplicates = keys.filter((key, index) => keys.indexOf(key) !== index);
  if (duplicates.length) fail(id, `duplicate keys: ${[...new Set(duplicates)].join(", ")}`);
  all.forEach((primitive, index) => {
    if (primitive.ownerIndex !== index) fail(id, `${primitive.key} owner index ${primitive.ownerIndex} breaks dense order at ${index}`);
    const { min, max } = primitive.aabb_m;
    const finite = [min.x, min.y, min.z, max.x, max.y, max.z, ...primitive.material.colorLinear].every(Number.isFinite);
    if (!finite) fail(id, `${primitive.key} has non-finite bounds or colour`);
    if (max.x < min.x || max.y < min.y || max.z < min.z) fail(id, `${primitive.key} has inverted bounds`);
    const { roughness, emission } = primitive.material;
    if (!(roughness >= 0 && roughness <= 1)) fail(id, `${primitive.key} roughness ${roughness} outside [0,1]`);
    if (!(emission >= 0)) fail(id, `${primitive.key} emission ${emission} is negative`);
  });

  // The container interior is the water's; scenery that intrudes will be
  // intersected by the free surface instead of framing it. Heightfield worlds
  // are exempt: there the terrain carves the basin, the container is the whole
  // garden, and the banks the scenery stands on are inside it by construction.
  if (first.shell.kind !== "terrain-heightfield") {
    const c = scene.container;
    const intruding = first.primitives.filter(({ aabb_m: box }) =>
      box.min.x < c.width_m / 2 && box.max.x > -c.width_m / 2
      && box.min.z < c.depth_m / 2 && box.max.z > -c.depth_m / 2
      && box.min.y < c.height_m && box.max.y > 0);
    if (intruding.length) fail(id, `${intruding.length} prop(s) inside the container volume: ${intruding.slice(0, 5).map(({ key }) => key).join(", ")}`);
  }

  const coverage = buildSvoEnvironmentCoverage(scene, id);
  const raster = coverage.entries.filter(({ visibleOwnership }) => visibleOwnership === "raster-only-procedural");
  if (raster.length) fail(id, `${raster.length} entry(ies) still raster-only: ${raster.map(({ key }) => key).join(", ")}`);
  const omitted = coverage.entries.filter(({ reason }) => reason === "light-record-capacity");
  if (omitted.length) fail(id, `${omitted.length} light(s) dropped past the ${SVO_LIGHT_MAXIMUM_RECORDS}-record cap: ${omitted.map(({ key }) => key).join(", ")}`);

  const lit = first.primitives.filter(({ tags }) => tags.includes("light")).length;
  console.log(`${failures ? " " : "✓"} ${id.padEnd(17)} props ${String(first.primitives.length).padStart(3)}  shell ${String(first.shell.primitives.length).padStart(2)}  lights ${String(lit).padStart(2)}  coverage ${coverage.summary.complete}✓ ${coverage.summary.degraded}~ ${coverage.summary.unsupported}✗`);
}

if (failures) { console.error(`\n${failures} scenery check failure(s)`); process.exit(1); }
console.log(`\nAll ${targets.length} environment(s) pass.`);
