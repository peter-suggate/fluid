/**
 * Observation contract between the Seep solver (lane S, owner of this file) and
 * the comparison harness (lane H, tools/seep-*.ts). See
 * docs/plans/seep-rewrite-2026-11-14.md §3.
 *
 * The harness, not the solver, measures the solver: it binds the buffers named
 * here, includes `wgsl`, and computes the shared observable schema with the same
 * six-tetrahedron fill (lib/core/level-set-fill.wgsl.ts) and zone definitions it
 * uses for Narrow-band FLIP. The solver only promises that the accessors read its
 * current, completed state after `awaitFrameCompletion()`.
 *
 * Lattice: the seep box, 512 x 256 x 64 cells at h = 5 mm, cell (0,0,0) at the
 * box's low corner. Vertex v is the cell corner at v * h (0..dims inclusive).
 */
export interface SeepObservationSource {
  readonly dimensions: readonly [number, number, number];
  readonly cellSize_m: number;
  /**
   * Read-only storage buffers in bind order. The harness binds entry i at
   * @group(OBSERVATION_GROUP) @binding(i) as `var<storage, read>` with the
   * declared WGSL element type.
   */
  readonly buffers: readonly { readonly name: string; readonly wgslType: string; readonly buffer: GPUBuffer }[];
  /**
   * WGSL providing, given the bindings above declared by `declarations(group)`:
   *  - fn seepVertexPhi(v: vec3i) -> f32   signed distance in cells, negative in liquid;
   *                                         defined for 0 <= v <= dims (clamped outside)
   *  - fn seepCellOpen(c: vec3i) -> bool   false for solid cells and outside the box
   *  - fn seepFaceU(c: vec3i, axis: u32) -> f32  MAC velocity in m/s on cell c's
   *                                         upper face along axis (0 x, 1 y, 2 z)
   *  - fn seepParticleCount() -> u32       live FLIP particles
   */
  declarations(group: number): string;
  readonly wgsl: string;
}
