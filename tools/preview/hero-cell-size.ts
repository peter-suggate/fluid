/**
 * The hero pond at whatever lattice `FLUID_PREVIEW_CELL_MM` names.
 *
 * One module rather than one per size, because the thing under test is the
 * lattice and nothing else: the document, the set, the environment and the
 * camera must be byte-identical between two rows of a sweep or the comparison
 * is not about resolution. That is also why it goes through `heroPreviewScene`
 * — see the note there — and why the only field it touches is `voxelDomain`.
 *
 * `FLUID_PREVIEW_BRICK_CELLS` is separate because brick size is the other axis
 * of the same question: a 4-cell brick is a finer leaf at the same voxel size,
 * and it costs eight times fewer voxels per leaf against eight times more
 * leaves. The two have to be movable independently to tell them apart.
 *
 * Evidence built with this module is in `docs/SVO_FINE_VOXEL_CAPACITY.md`.
 */
import { heroPreviewCamera, heroPreviewScene } from "./hero-still";

export const createScene = () => {
  const scene = heroPreviewScene();
  const millimetres = Number(process.env.FLUID_PREVIEW_CELL_MM ?? scene.voxelDomain.finestCellSize_m * 1000);
  if (!Number.isFinite(millimetres) || millimetres <= 0) throw new RangeError("FLUID_PREVIEW_CELL_MM must be a positive number of millimetres");
  const brickCells = Number(process.env.FLUID_PREVIEW_BRICK_CELLS ?? scene.voxelDomain.brickSize_cells);
  if (brickCells !== 4 && brickCells !== 8) throw new RangeError("FLUID_PREVIEW_BRICK_CELLS must be 4 or 8");
  return { ...scene, voxelDomain: { ...scene.voxelDomain, finestCellSize_m: millimetres / 1000, brickSize_cells: brickCells } };
};

export const camera = heroPreviewCamera();
