import { solidFractionOccupies, type VoxelBlock } from "../../solid-patch/kind";
import type { SolidPatchOperation, SolidWorldCoordinate as Cell, SolidWorldVoxelPatch as Patch } from "../../solid-world";
import { OPERATION_OPPOSITE, mirrorControls, operationCaption, operationControl, operationTone, shellControl,
  toolPatchOperation } from "../controls";
import { assertStrokeCaps, faceBox, mirroredPatches, strokeHighlight, toolSurface, voxelToolUnavailable } from "../geometry";
import type { ToolAvailabilityContext, ToolContext, ToolGesture, ToolNumberControl, ToolUpdate, VoxelToolPlugin } from "../plugin";
import { blockPatches, rotateBlock90 } from "../voxel-grid";

const EMPTY_CLIPBOARD = "Copy a selection first.";
const turnsControl: ToolNumberControl = { id: "turns", label: "Quarter turns · about the up axis", short: "Turn", adjust: "primary", min: 0, max: 3, step: 1, initial: 0 };

/** A copied block as it is stamped: turned, cut down to its solid cells, and its patches for each operation. */
interface Stamp {
  readonly block: VoxelBlock;
  /** As if the block stood at the lattice origin; a stamp moves them, and never encodes the block again. */
  readonly patches: Map<SolidPatchOperation, readonly Patch[]>;
}
const stamps = new WeakMap<VoxelBlock, Map<number, Stamp>>();

/** The block without the empty margins a copied box carries, so it rests on a face by its own cells. */
function trimmed(block: VoxelBlock): VoxelBlock {
  const [sx, sy, sz] = block.size, lo = [sx, sy, sz], hi = [-1, -1, -1];
  for (let z = 0, cell = 0; z < sz; z += 1) for (let y = 0; y < sy; y += 1) for (let x = 0; x < sx; x += 1, cell += 1) {
    if (!solidFractionOccupies(block.solidFraction[cell]!)) continue;
    if (x < lo[0]!) lo[0] = x; if (x > hi[0]!) hi[0] = x;
    if (y < lo[1]!) lo[1] = y; if (y > hi[1]!) hi[1] = y;
    if (z < lo[2]!) lo[2] = z; if (z > hi[2]!) hi[2] = z;
  }
  if (hi[0]! < 0) throw new Error("The copied block holds no solid voxels; copy a selection that has some.");
  const size = lo.map((v, a) => hi[a]! - v + 1) as unknown as Cell;
  if (size.every((v, a) => v === block.size[a])) return block;
  const cells = size[0] * size[1] * size[2];
  const solidFraction = new Uint8Array(cells), signedDistanceQ8 = new Int16Array(cells), materialId = new Uint16Array(cells);
  for (let z = 0, cell = 0; z < size[2]; z += 1) for (let y = 0; y < size[1]; y += 1) {
    const row = lo[0]! + sx * (lo[1]! + y + sy * (lo[2]! + z));
    for (let x = 0; x < size[0]; x += 1, cell += 1) {
      solidFraction[cell] = block.solidFraction[row + x]!;
      signedDistanceQ8[cell] = block.signedDistanceQ8[row + x]!;
      materialId[cell] = block.materialId[row + x]!;
    }
  }
  return { size, solidFraction, signedDistanceQ8, materialId };
}

function stampOf(copied: VoxelBlock, turns: number): Stamp {
  let byTurns = stamps.get(copied);
  if (!byTurns) stamps.set(copied, byTurns = new Map());
  let stamp = byTurns.get(turns);
  if (!stamp) byTurns.set(turns, stamp = { block: trimmed(rotateBlock90(copied, 1, turns)), patches: new Map() });
  return stamp;
}

/**
 * Stamp the copied voxels where the pointer is.
 *
 * The block stands against the picked face, centred under the pointer: on the
 * face when attaching, sunk into it from the picked cell when erasing or
 * repainting, since those act on what is already there. Attaching writes the
 * block's cells with their own materials; erasing takes away the cells the
 * block has; repainting gives the solids under it the block's materials.
 */
function beginPattern(context: ToolContext): ToolGesture | undefined {
  const { scene, values, ray, clipboard } = context;
  if (!clipboard) throw new Error(EMPTY_CLIPBOARD);
  const first = toolSurface(context, ray);
  if (!first) return undefined;
  const operation = toolPatchOperation(context);
  const stamp = stampOf(clipboard.block, values[turnsControl.id]!);
  let atOrigin = stamp.patches.get(operation);
  if (!atOrigin) stamp.patches.set(operation, atOrigin = blockPatches(operation, [0, 0, 0], stamp.block));
  const size = stamp.block.size;
  let shown: { corner: Cell; update: ToolUpdate } | undefined;
  return { surface: first.picked, update(input): ToolUpdate | undefined {
    const surface = toolSurface(context, input, false);
    if (!surface) return undefined;
    const lo = surface.coordinate.map((v, a) => v - Math.floor((size[a]! - 1) / 2)), hi = lo.map((v, a) => v + size[a]! - 1);
    const corner = faceBox(surface, lo, hi, size[surface.faceAxis], operation === "fill").minimum;
    if (shown && corner.every((v, a) => v === shown!.corner[a])) return shown.update;
    const patches = mirroredPatches(context, atOrigin!.map((patch) => ({ ...patch,
      minimum: patch.minimum.map((v, a) => v + corner[a]!) as unknown as Cell,
      maximumExclusive: patch.maximumExclusive.map((v, a) => v + corner[a]!) as unknown as Cell })));
    assertStrokeCaps(patches, "The copied block is too large to stamp in one stroke; copy a smaller selection or use fewer mirror planes.");
    const update: ToolUpdate = { patches, tone: operationTone(operation), highlight: strokeHighlight(scene, patches),
      caption: `${operationCaption(operation)} PATTERN · ${size.join(" × ")}` };
    shown = { corner, update }; return update;
  } };
}

export const patternTool: VoxelToolPlugin = {
  id: "pattern", version: 1,
  ui: { label: "Pattern", hint: "Stamp the copied voxels against the face under the pointer.", group: "Brush", order: 5,
    opposite: OPERATION_OPPOSITE,
    icon: "M4 4h7v7H4ZM13 4h7v7h-7ZM4 13h7v7H4ZM13 13h7v7h-7Z",
    controls: [operationControl, turnsControl, ...mirrorControls, shellControl] },
  unavailable: (context: ToolAvailabilityContext) => context.clipboard ? voxelToolUnavailable(context) : EMPTY_CLIPBOARD,
  begin: beginPattern,
};
