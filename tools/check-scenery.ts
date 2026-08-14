/**
 * Per-scene scenery check, so one set can be art-directed without running (or
 * waiting on) the whole suite.
 *
 *   node --import tsx tools/check-scenery.ts garden
 *   node --import tsx tools/check-scenery.ts hero-garden
 *   node --import tsx tools/check-scenery.ts            # every subject
 *
 * Verifies the invariants a scenery module can plausibly break: determinism,
 * unique keys, dense owner order, finite bounds, light-record budget, and that
 * nothing has been parked inside the container where the water lives.
 */
import { environmentIds, type EnvironmentId } from "../lib/core/environments";
import { createHeroGardenHoseScene, HERO_GARDEN_CELL_M } from "../lib/core/hero-garden-scene";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/core/model";
import { buildSvoEnvironmentCoverage } from "../lib/svo/svo-scene-coverage";
import { SVO_LIGHT_MAXIMUM_RECORDS } from "../lib/svo/svo-light-abi";
import { terrainSampleShape } from "../lib/core/terrain";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/core/voxel-environments";

/**
 * What gets walked, and why it is not simply `environmentIds`.
 *
 * `buildEnvironmentProxyCatalog` reads `scene.scenery ?? sceneryGraphForEnvironment(...)`,
 * and a default scene carries no graph of its own — so walking the eight
 * environments walks eight *presets* and no authored document. The hero garden
 * is the one scene in the app that authors its own graph, and it is by some
 * distance the largest: about 2 200 props from four generator nodes, against a
 * couple of hundred in the richest preset. It was invisible to this check.
 *
 * The environment still has to be named for a document that carries a graph,
 * because the shell, the floor datum and the material closures come from the
 * preset either way; the hero pond is a garden.
 */
interface ScenerySubject {
  readonly id: string;
  readonly environment: EnvironmentId;
  /** Built twice per run, so the determinism check compares two real builds. */
  readonly build: () => SceneDescription;
}

/**
 * The hero garden at both ends of the detail ladder.
 *
 * A generator sizes its features in *detail voxels* — `bonsaiCanopyLadder`
 * raises its floret to a floor of leaves, and every other legibility floor in
 * `lib/voxel-scenery` is the same kind of number — so "is this set richer at a
 * finer leaf?" is a question with an answer, and it is one this check can give
 * in seconds where the Dawn lane needs the exclusive lock and eight minutes.
 *
 * Both rungs are built through `createHeroGardenHoseScene`'s own options rather
 * than by writing a size onto a finished document, because that is the only
 * thing that reaches a bake or a generator; see `SceneLattice`.
 *
 * `÷8` is the bottom of the ladder the octree can spend
 * (`SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM` = 3), so between these two rows
 * is the whole of what a finer lattice is able to buy this set today.
 */
const HERO_GARDEN_FINE_DIVISOR = 8;

const SUBJECTS: readonly ScenerySubject[] = [
  ...environmentIds.map((environment) => ({ id: environment, environment, build: () => cloneScene(defaultScene) })),
  { id: "hero-garden", environment: "garden" as EnvironmentId, build: () => createHeroGardenHoseScene() },
  {
    id: "hero-garden-fine",
    environment: "garden" as EnvironmentId,
    build: () => createHeroGardenHoseScene({
      cellSize_m: HERO_GARDEN_CELL_M,
      detailCellSize_m: HERO_GARDEN_CELL_M / HERO_GARDEN_FINE_DIVISOR,
    }),
  },
];

const requested = process.argv.slice(2).filter((value) => !value.startsWith("-"));
const targets = requested.length
  ? requested.map((name) => {
    const subject = SUBJECTS.find(({ id }) => id === name);
    if (!subject) throw new Error(`Unknown scenery subject ${name}; expected one of ${SUBJECTS.map(({ id }) => id).join(", ")}`);
    return subject;
  })
  : SUBJECTS;

let failures = 0;
const fail = (id: string, message: string) => { failures++; console.error(`  ✗ ${id}: ${message}`); };

for (const { id, environment, build } of targets) {
  const scene = build();
  const first = buildEnvironmentProxyCatalog(scene, environment);
  const second = buildEnvironmentProxyCatalog(build(), environment);
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

  const coverage = buildSvoEnvironmentCoverage(scene, environment);
  const unsupported = coverage.entries.filter(({ status }) => status === "unsupported");
  if (unsupported.length) fail(id, `${unsupported.length} entry(ies) the tracer does not own: ${unsupported.map(({ key }) => key).join(", ")}`);
  const omitted = coverage.entries.filter(({ reason }) => reason === "light-record-capacity");
  if (omitted.length) fail(id, `${omitted.length} light(s) dropped past the ${SVO_LIGHT_MAXIMUM_RECORDS}-record cap: ${omitted.map(({ key }) => key).join(", ")}`);

  const lit = first.primitives.filter(({ tags }) => tags.includes("light")).length;
  // The lattice the set was expanded at, and the ground it was seated on. Both
  // are inputs to construction, so a row that reports them is the only way to
  // tell a set that got richer at a finer leaf from one that ignored it.
  const detail_m = scene.voxelDomain.detailCellSize_m ?? scene.voxelDomain.finestCellSize_m;
  const bake_m = terrainSampleShape(scene.terrain)?.spacing_m;
  console.log(`${failures ? " " : "✓"} ${id.padEnd(17)} props ${String(first.primitives.length).padStart(4)}`
    + `  shell ${String(first.shell.primitives.length).padStart(2)}  lights ${String(lit).padStart(2)}`
    + `  coverage ${coverage.summary.complete}✓ ${coverage.summary.degraded}~ ${coverage.summary.unsupported}✗`
    + `  leaf ${(detail_m * 1000).toFixed(4).replace(/\.?0+$/, "").padStart(6)} mm`
    + (bake_m === undefined ? "" : `  ground ${(bake_m * 1000).toFixed(4).replace(/\.?0+$/, "")} mm`
      + ` = ${(bake_m / detail_m).toFixed(2)} leaves`));
}

if (failures) { console.error(`\n${failures} scenery check failure(s)`); process.exit(1); }
console.log(`\nAll ${targets.length} subject(s) pass.`);
