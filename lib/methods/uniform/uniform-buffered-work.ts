/** Launch width a GPU relayout keeps in reserve, in the budget's units. Its
 * census admits h tiles up to the capacity in one frame and the count reaches
 * the host UNIFORM_MIXED_RECEIPT_RING frames later, so a budget decayed to one
 * group runs them on one group for both frames (128-class pool at rest, a
 * Fine region drawn: 3662 + 3655 ms of passes against 22.5 steady). A host
 * layout announces its own counts and reserves nothing. Wider than this buys
 * little on the Apple GPUs measured (those two frames 42.3 + 24.0 ms here,
 * 39.9 + 22.5 at the ceilings) and the empty launches cost more at rest
 * (+0.25 ms a frame here, +0.61 at the ceilings). */
export const UNIFORM_WORK_RELAYOUT_RESERVE=1024;
/** A host request edit a GPU relayout builds, counted before any receipt
 * reports it. tiles: tiles whose width it can change (those entering and
 * leaving the requests). reach: those and every tile within one tile of
 * them, each once; a tile's class (width, seam or regular) is the widths of
 * its 3x3x3, so no other tile's class moves. Edits not yet in a receipt add. */
export interface UniformWorkEdit{readonly tiles:number;readonly reach:number}
/** Direct grid-stride launch budget from completed-frame evidence. Keep 25%
 * headroom, grow immediately, and release at most half the budget per frame,
 * never below `reserve` (work that can arrive unannounced).
 * This sizes parallelism, never storage or the live list: kernels must stride
 * the actual GPU count, so even a sudden growth beyond the estimate is complete.
 * One group remains for empty lists and their receipt/diagnostic side effects. */
export function uniformBufferedWork(previous:number,observed:number,ceiling:number,reserve=0):number{
 if(!Number.isInteger(previous)||previous<1||!Number.isInteger(observed)||observed<0||!Number.isInteger(ceiling)||ceiling<1||!Number.isInteger(reserve)||reserve<0)
  throw new Error("Buffered work needs positive integer budgets, a nonnegative count and a nonnegative reserve");
 return Math.min(ceiling,Math.max(1,reserve,Math.ceil(previous/2),Math.ceil(observed*1.25)));
}
