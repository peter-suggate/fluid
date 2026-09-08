import { readFileSync } from "node:fs";
import { createSparseAdaptiveMassAtlas, sparseBrickSpan } from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { packSparseCM12ResidentTopologyArchetypesForQA, sparseCM12HostTemplateVariantsEnabled } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const input = JSON.parse(readFileSync("/tmp/fluid-cm12-generation-atlas.json", "utf8"));
const atlas = createSparseAdaptiveMassAtlas(input.dimensions, input.bricks.map((brick: any) => ({
  ...brick, density: new Float64Array(brick.resolution ** 3).fill(brick.density),
  gamma: new Float64Array(brick.resolution ** 3).fill(brick.gamma),
})), input.generation, input.brickFineResolution, input.signedCoordinates, false);
const active = new Set<number>(input.active);
const grid = buildSparseAtlasCompositeGrid(atlas);
const apron = new Set<string>();
const activeMutable = atlas.bricks.filter(brick => active.has(brick.key) && sparseBrickSpan(brick) <= 2);
for (const brick of activeMutable) {
  const span = sparseBrickSpan(brick), [bx, by, bz] = brick.coordinate;
  for (let z = -1; z <= span; z++) for (let y = -1; y <= span; y++) for (let x = -1; x <= span; x++)
    apron.add(`${bx + x}/${by + y}/${bz + z}`);
}
let mutable = new Set(atlas.bricks.filter(brick => (active.has(brick.key) && sparseBrickSpan(brick) <= 2)
  || (sparseBrickSpan(brick) === 1 && apron.has(brick.coordinate.join("/")))).map(brick => brick.key));
const fits = () => sparseCM12HostTemplateVariantsEnabled(grid.cells.length, grid.gradientRows.length,
  mutable.size, atlas.brickFineResolution, {
    cells: grid.cells.reduce((count, cell) => count + Number(mutable.has(cell.brickKey)), 0),
    rows: grid.gradientRows.reduce((count, row) => count + Number(row.terms.every(term => mutable.has(grid.cells[term.cellId]!.brickKey))), 0),
  });
if (!fits()) mutable = new Set(activeMutable.map(brick => brick.key));
if (!fits()) throw new Error("Captured generation does not select the all-rung production catalogue");
console.log(JSON.stringify({ phase: "before-catalog", bricks: atlas.bricks.length,
  cells: grid.cells.length, rows: grid.gradientRows.length, mutable: mutable.size, ...process.memoryUsage() }));
const start = performance.now();
const output = packSparseCM12ResidentTopologyArchetypesForQA(atlas, grid, mutable,
  phase => console.log(JSON.stringify({ phase, milliseconds: performance.now() - start, ...process.memoryUsage() })));
console.log(JSON.stringify({ phase: "after-catalog", milliseconds: performance.now() - start,
  cellCount: output.cellCount, rowCount: output.rowCount, archetypes: output.gpuExpansion!.archetypeCount,
  rangeCount: output.gpuExpansion!.cellRangeCount, shadowBytes: output.words.byteLength,
  placementBytes: output.gpuExpansion!.words.byteLength, ...process.memoryUsage(),
  maxRSS_kB: process.resourceUsage().maxRSS }));
