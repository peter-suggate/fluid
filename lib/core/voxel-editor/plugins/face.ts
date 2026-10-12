import type { VoxelBlock } from "../../solid-patch/kind";
import type { SolidWorldCoordinate as Cell, SolidWorldVoxelPatch as Patch } from "../../solid-world";
import { OPERATION_OPPOSITE, byMaterialControl, depthControl, materialControl, mirrorControls, operationCaption,
  operationControl, operationTone, shellControl, toolMaterialId, toolPatchOperation } from "../controls";
import { MAX_STROKE_VOXELS, assertStrokeCaps, faceBox, facePoint, mirroredPatches, normalTravel, strokeHighlight,
  toolPassesThrough, toolSurface, voxelToolUnavailable } from "../geometry";
import { toolReader, type ToolContext, type ToolGesture, type ToolUpdate, type VoxelToolPlugin } from "../plugin";
import type { VoxelReader } from "../voxel-access";
import { DEFAULT_REGION_REACH, blockOfRegion, blockPatches, faceRegion, regionPatches } from "../voxel-grid";

/**
 * The whole face under the pointer, moved along its normal.
 *
 * The face is every exposed cell joined edge to edge to the picked one in its
 * own layer — optionally only those of its material. Dragged out of the solid
 * it is extruded by that many layers; dragged in, cut back by that many. A
 * click that does not move does one layer of whichever operation is lit, and
 * with paint lit the face's cells are repainted, as deep as the drag.
 *
 * An extrusion copies: each new cell takes the material of the face cell it
 * was pulled from, so a face of several materials grows as itself. The chosen
 * material is what a repaint writes.
 */
function beginFace(context: ToolContext): ToolGesture | undefined {
  const { scene, values, ray } = context;
  const surface = toolSurface(context, ray);
  // The ground plane is not a face of anything.
  if (!surface?.picked) return undefined;
  const lit = toolPatchOperation(context);
  const materialId = toolMaterialId(values);
  const reader = toolReader(context);
  // The face is found in the world the pick saw. A ray that passed through the
  // tank wall can land on a cell standing flush against it, and to the reader
  // that face is covered: asked that way, the search refused the very cell the
  // pick had just handed it.
  const seen: VoxelReader = { scene, fraction: (cell) => reader.fraction(cell), material: (cell) => reader.material(cell),
    solid: (cell) => reader.solid(cell) && !toolPassesThrough(context, cell),
    pick: (from, options) => reader.pick(from, options), read: (minimum, maximum) => reader.read(minimum, maximum) };
  const face = faceRegion(seen, surface, { byMaterial: values[byMaterialControl.id] === 1 });
  const axis = surface.faceAxis;
  const pressed = facePoint(scene, surface, ray);
  if (!pressed) return undefined;
  const layerSize = face.minimum.map((v, a) => face.maximumExclusive[a]! - v);
  const layerCells = layerSize[0]! * layerSize[1]! * layerSize[2]!;
  const last = face.maximumExclusive.map((v) => v - 1);
  let layer: VoxelBlock | undefined;
  /** The face's layer repeated `layers` times along the normal: for each cell of the prism, the layer cell it copies. */
  const sources = (size: readonly number[]): Uint32Array => {
    const source = new Uint32Array(size[0]! * size[1]! * size[2]!);
    const at = [0, 0, 0];
    for (let z = 0, cell = 0; z < size[2]!; z += 1) for (let y = 0; y < size[1]!; y += 1) for (let x = 0; x < size[0]!; x += 1, cell += 1) {
      at[0] = x; at[1] = y; at[2] = z; at[axis] = 0;
      source[cell] = at[0]! + layerSize[0]! * (at[1]! + layerSize[1]! * at[2]!);
    }
    return source;
  };
  /** Out of the solid extrudes and into it cuts, whichever is lit; a click that did not move does what is lit. */
  const operationAt = (depth: number) => lit === "paint" ? "paint" : depth > 0 ? "fill" : depth < 0 ? "clear" : lit;
  const propose = (depth: number): Patch[] => {
    const operation = operationAt(depth);
    const layers = Math.max(1, Math.abs(depth));
    if (layerCells * layers > MAX_STROKE_VOXELS) {
      throw new Error(`This face spans ${layerCells.toLocaleString("en-US")} voxels; move it at most `
        + `${Math.floor(MAX_STROKE_VOXELS / layerCells)} layers in one stroke.`);
    }
    const box = faceBox(surface, face.minimum, last, layers, operation === "fill");
    const size = box.minimum.map((v, a) => box.maximumExclusive[a]! - v) as unknown as Cell;
    const source = sources(size), cells = source.length;
    if (operation !== "fill") {
      const mask = new Uint8Array(cells);
      for (let cell = 0; cell < cells; cell += 1) mask[cell] = face.mask[source[cell]!]!;
      return regionPatches(operation, { ...box, mask }, materialId);
    }
    layer ??= blockOfRegion(reader, face);
    const solidFraction = new Uint8Array(cells), signedDistanceQ8 = new Int16Array(cells).fill(0x7fff);
    const materials = new Uint16Array(cells);
    for (let cell = 0; cell < cells; cell += 1) {
      const from = source[cell]!;
      if (face.mask[from] === 0) continue;
      // New cells are whole, half a cell deep, as a box fill writes them — whatever share of its own cell the face covered.
      solidFraction[cell] = 255; signedDistanceQ8[cell] = -128; materials[cell] = layer.materialId[from]!;
    }
    return blockPatches("fill", box.minimum, { size, solidFraction, signedDistanceQ8, materialId: materials });
  };
  let depth = 0, shown: { readonly depth: number; readonly update: ToolUpdate } | undefined;
  return { surface: true, update(input): ToolUpdate | undefined {
    depth = normalTravel(scene, surface, pressed, input, depthControl.max) ?? depth;
    // A drag crosses many samples per layer; the proposal is rebuilt only when the layer count changes.
    if (shown?.depth === depth) return shown.update;
    const patches = mirroredPatches(context, propose(depth));
    assertStrokeCaps(patches, "This face is too large to move that far in one stroke; drag a shorter distance.");
    const operation = operationAt(depth);
    const update: ToolUpdate = { patches, tone: operationTone(operation), highlight: strokeHighlight(scene, patches),
      caption: `${operationCaption(operation)} FACE · ${face.cellCount.toLocaleString("en-US")} voxels × ${Math.max(1, Math.abs(depth))}`
        + (face.clipped ? ` · stopped ${DEFAULT_REGION_REACH} voxels from the pointer; the face goes on` : "") };
    shown = { depth, update };
    return update;
  } };
}

export const faceTool: VoxelToolPlugin = {
  id: "face", version: 1,
  ui: { label: "Face", hint: "Press a face and drag along its normal: out extrudes the whole face, copying its materials; in cuts it back. A click does one layer.", group: "Brush", order: 1,
    // A drag has already said which way; only the click that did not move has an operation to swap.
    opposite: `on a click, ${OPERATION_OPPOSITE}`,
    icon: "M5 14h14v6H5ZM12 12V3M8.5 6.5 12 3l3.5 3.5",
    controls: [operationControl, materialControl, byMaterialControl, ...mirrorControls, shellControl] },
  unavailable: voxelToolUnavailable,
  begin: beginFace,
};
