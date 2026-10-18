/**
 * The records a solver publishes for its own grid-overlay views.
 *
 * Each view reads what the step already wrote, so it adds a draw and no
 * simulation work. Legacy single views bind their source directly. Composed
 * layers copy the window header and tile records into one presentation buffer,
 * retaining the same storage-binding budget. Missing sources denote the dense
 * schedule (whole-domain window and fine sampling everywhere).
 *
 * Shapes only, like `levelset-consumer-abi`: the solver that fills a record and
 * the shader that reads it both name its words from here, and neither imports
 * the other.
 */

/** A solver's own buffer, bound as-is for the view on screen. */
export interface GPUFluidViewRecords {
  readonly records: GPUBufferBinding;
}

/**
 * Uniform Geometric's two-level tile classes, for the fine-tiles view: one
 * four-word record per 4³ tile of the finest lattice, x fastest over n/4 tiles
 * per axis. Word {@link TILE_CLASS_RECORD_CLASS_WORD} is the class; the other
 * three are the solver's 4h face table, which the view never reads. Published
 * only while the step encodes the sampler; with it off every cell reads the
 * finest lattice anyway.
 */
export type GPUFluidTileClassSource = GPUFluidViewRecords;

export const TILE_CLASS_RECORD_WORDS = 4;
export const TILE_CLASS_RECORD_CLASS_WORD = 3;
/** Class bits: the sampler reads the finest lattice. */
export const TILE_CLASS_FINE = 1;
/** Class bits: FINE dilated by the shell reach; the extension's working set. */
export const TILE_CLASS_SHELL = 2;

/**
 * The uniform solvers' solve-window header, for the solve-window view. Cell
 * boxes are [minimum, maximum) on the finest lattice. Published only while the
 * window is on; withdrawn, the view reads the whole domain, which is what the
 * dense schedule dispatches.
 */
export type GPUFluidSolveWindowSource = GPUFluidViewRecords;

/** This step's own padded box: liquid, near-surface band and sources. */
export const SOLVE_WINDOW_SEED_WORD = 0;
/** The window: this step's box united with the last. Every kernel reads its origin. */
export const SOLVE_WINDOW_BOX_WORD = 7;
/**
 * The group counts the host launched from the window origin, sized from a box
 * a few steps old. All three are zero only on the indirect A/B lane, where the
 * GPU's own exact counts launch the window itself.
 */
export const SOLVE_WINDOW_HOST_GROUPS_WORD = 228;
export const SOLVE_WINDOW_RECORD_WORDS = SOLVE_WINDOW_HOST_GROUPS_WORD + 3;

/** Pool-block page records (32-word header, four words per block), followed
 * in the overlay's combined buffer by the page directory and its 32-word
 * moving-window header. Published page coordinates (record words 0..2) and
 * the resident bounds (header words 20..22, 24..26) are signed *world* pages,
 * equal to the allocator's own records and unchanged by a window shift. A
 * reader takes the lattice page as world - F, with F the directory header's
 * words 0..2 from the same copy; the directory slot of a world page is
 * world & (edge - 1). Optional frame is the live solver params prefix (grid,
 * cell size, live lattice origin), so metres = origin + h * 32 * (world - F).
 * Optional tiers are the work tier (0 cold, 1 warm, 2 hot) of each 16h
 * region, a word a region: region 8·block + the octant of the block it lies
 * in (x | y<<1 | z<<2); `size` bounds the region count. Their resolution
 * settings (NB-FLIP warmResolution, coldRegions) name what each tier is
 * computed at: the heat layer's labels. */
export interface GPUFluidVolumePageSource extends GPUFluidViewRecords {
  readonly directory: GPUBufferBinding;
  readonly frame?: GPUBufferBinding;
  readonly tiers?: GPUBufferBinding & { readonly size: number; readonly warmResolution?: "4h" | "h"; readonly coldRegions?: "frozen16h" | "fine" };
}
