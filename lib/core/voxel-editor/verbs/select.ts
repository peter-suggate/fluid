import { sceneLatticeRegion, type VoxelRegionVerbPlugin } from "../verb";

/**
 * The two verbs that change what is selected and nothing else. Neither is an
 * edit, so neither stands behind the solid-edit gate: a selection can always
 * be made and always be put down.
 */

/**
 * Select every cell of the lattice, so the next verb acts on all its solids.
 *
 * A plain box — no cells are read — which is why it is cheap on any scene. The
 * vessel's shell stands outside the lattice and is not part of it.
 */
export const selectAllVerb: VoxelRegionVerbPlugin = {
  id: "select-all",
  version: 1,
  ui: {
    label: "All", hint: "Select every voxel inside the scene's lattice", group: "Select", order: 30, ring: false,
    icon: "M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4M9 9h6v6H9z",
    keys: ["Mod+a"],
    standalone: true,
  },
  unavailable: () => undefined,
  run: ({ scene }) => {
    const region = sceneLatticeRegion(scene);
    return { patches: [], region, caption: `Selected the whole lattice: ${region.maximumExclusive.join(" × ")} voxels` };
  },
};

/** Put the selection down. Nothing is announced: the outline going is the whole of the news. */
export const deselectVerb: VoxelRegionVerbPlugin = {
  id: "deselect",
  version: 1,
  ui: {
    label: "None", hint: "Put the selection down", group: "Select", order: 31, ring: false,
    icon: "M6 6l12 12M18 6L6 18",
    keys: ["Escape"],
  },
  unavailable: () => undefined,
  run: () => ({ patches: [], region: null, caption: "" }),
};
