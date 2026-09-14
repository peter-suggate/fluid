# Sharpening for level-set-plus-volume: claim, verdict, plan

Date: 2026-09-14. Applies to `TransportExperiment::LevelSetVolume` in `fluid-core`.

## The claim and the verdict

Claim: because the surface is a signed distance, its normal field is smooth, so diffuse volume can be pulled back to the surface along that normal without the lumpy interfaces of Chentanez and Müller (CM12).

Verdict: half right. The **direction** is smooth, and that is a real advantage over CM12, whose return rays follow the gradient of the diffuse density itself and land with grid-scale noise. The **amount** is not smooth: the mass arriving at a surface cell is the integral of diffuse volume along its normal ray tube, which is curvature-biased and quantised per donor cell. If the surface is then repositioned from V per cell, that noise becomes the surface and the claim buys nothing.

Two further facts decide the design:

- **Conservation forces φ to move.** Transport conserves ΣV exactly and does not conserve the φ-enclosed area. On the 180-frame run the φ area fell 41.7 % while V drifted 4e-9. A pass that only moves V along the normal can never close that deficit; it piles V into an over-full band while the drawn surface keeps shrinking.
- **CM12's lumpiness has four causes and only one is the gradient direction.** The surface is the contour of the field being modified; deposition is one ray, one point, per donor with a cubic nonlinear amount; the operator is one-sided so bumps are added and never removed; and rays converge on concave patches. Having φ removes the first cause and the direction noise. The amount noise and the one-sidedness remain unless the pass is designed around them.

Closest prior art: the accurate conservative level set of Desjardins, Moureau and Pitsch 2008, which compresses a conservative scalar along the normal of a separately maintained distance field. No published scheme has our exact split; treat the direction as ours to validate.

## The formulation

Two-sided, φ first, then V. Band B is cells within two widths of φ = 0; H_i(φ) is the linear-cut liquid fraction from vertex φ; a_i is interface length in cell i; C_i capacity.

1. Mismatch per cell: m_i = V_i − C_i·H_i(φ). Per unit interface length: σ_i = m_i / a_i on B, else 0. Per unit length, not per cell, or the coarse side of a seam contributes 4× and paints a seam bias onto the surface.
2. Smooth σ tangentially with an area-symmetric kernel, radius two coarsest band cells, three to five passes. Symmetric so the smoothing conserves Σ a_i σ_i.
3. Per φ-connected region, one multiplier λ_R so the regional enclosed area matches the regional ΣV.
4. Normal speed s_i = clamp(σ̄_i + λ_R, ±0.5 Δx_i/Δt); φ ← φ − s Δt |∇φ| on B; redistance B. This moves the surface smoothly: a smoothed source plus a per-region constant, applied as a normal flow.
5. Recompute H and the residual r_i = V_i − C_i·H_i. Donors are r_i > 0 with |φ_i| ≤ d_max (1.5 to 2 widths) and whose diffuse island holds at least half a cell of volume. Trace each donor along −sign(φ)·n from the redistanced φ to φ = 0, capped at d_max. Deposit into a world-space ball at the landing point, weights by cell overlap times remaining capacity (C_j − V_j)⁺, normalised. Under-filled cells draw the same way in reverse from cells over capacity.
6. Remaining excess goes to the pressure drain, expressed on the fraction V/C − 1, not on volume, so it is seam-symmetric.

Guaranteed: ΣV to roundoff, 0 ≤ V ≤ C after the drain, a smooth φ update, regional volume match to first order per pass, thin features preserved by the distance and island gates. Not guaranteed: exact per-cell V = C·H (deliberately, that slack is where sub-cell features live), topology control, momentum consistency (s is a geometric speed the projection never sees), and at large s the correction reads as surface tension exactly as CM12's D does.

## Where the current implementation would sabotage this

Found by reading the code and two native 30-frame runs; each must be fixed before or with step S1.

- The vertex lattice φ that the surface is drawn from is the raw semi-Lagrangian sample and is never redistanced; only the cell-centre copy is (`levelset_volume.rs:410-435`). A sharpening pass would read its normal from a field that is no longer a distance.
- No per-cell mismatch exists. `clipped_area` computes the φ-implied fill per fine cell and immediately sums it away (`levelset_surface.rs:19-36, :69`); only a global area error survives, and it is not in the native receipt.
- The seam metric is hardcoded off for this mode (`world.rs:718-720`) because it gates on a volume fraction in (0,1). It has to be rewritten against φ. The number this direction exists to make small is unmeasured.
- Velocity extension is seeded from V/C > 0.5 and zeroes every other cell (`numerics.rs:173-182`), so diffuse cells that φ calls liquid start from zero velocity. Seed from φ.
- The φ pressure geometry is silently zero where the 2×2 fit degenerates or at walls (`levelset_volume.rs:105-116`), and a sparse-air row with invalid geometry is skipped entirely (`numerics.rs:3090`), so that liquid gets no atmospheric condition.
- The drain is on volume, not fraction (`numerics.rs:3484-3516`), and on Figure 7 over-capacity rises monotonically through frame 30 (0 → 395 cells, 116 units) despite it. The gather is still a rigid translated box (`levelset_volume.rs:208-238`); at Courant 5 to 7 that is the diffusion source sharpening would be cleaning up after. The corner-traced quad from the earlier options list is the biggest lever on the source and is still open.
- Cut cells drop the φ normal on topology transfer (`transfer.rs:393`), so refinement near walls smears the interface before any sharpening runs.
- `maximum_over_capacity_ratio` skips near-solid cut cells (`levelset_volume.rs:401-404`); Figure 7 frame 15 reports 171 cells with ratio 0.

## Plan

- **S0, measure (half a day).** Keep the per-cell φ-implied fill; publish m_i split into inside-band, outside-band, and per-region area deficit; rewrite the seam metric against φ; fix the ratio counter.
- **S1, prerequisites (one day).** Redistance the vertex φ on the band every frame with the existing BVH; seed extension from φ; drain on fraction; keep the φ normal through cut-cell transfers.
- **S2, the φ side (two days).** Steps 1 to 4. Go/no-go: per-region area deficit near zero on both lanes, seam metric not worse than before, no visible ripple on the half-pool surface. If the surface visibly rounds, the smoothing radius or the speed clamp is too large.
- **S3, the V side (two days).** Step 5 with the three gates. Go/no-go: band |m| falls, over-capacity decays over frames, Figure 7's sheet length is not shorter than without S3.
- **S4, the source (one day, can run in parallel).** Corner-traced quad gather. Expect the row residual and the raw mismatch to fall so that S2 and S3 have less to do.
- **Test.** Dam break 32 and Figure 7, 30 and 180 frames, native harness; Peter checks the app. Metrics per frame: area deficit per region, band |m|, over-capacity count and decay, seam metric, sheet length on Figure 7.

Not in scope: topology repair, momentum-consistent correction, 3D.

## S4 isolated result

On the working tree based at `4b2f9058`, the Tall Cells hillside scene was run
for 40 frames in the SIMD Wasm harness at `dt = 1/30`, paper stepping, 256
pressure iterations, and `1e-6` pressure tolerance. The control and candidate
were identical except for selecting the old rigid centre-translated receiver
box or the RK2 corner-traced receiver footprint.

The corner footprint reduced the peak normalized row residual from 85.00 to
21.82. At frame 28 it reduced row residual from 13.18 to 1.86, total excess
from 10.97 to 6.97, phi/V L1 mismatch from 156.06 to 141.37, and maximum
normalized phi/V mismatch from 58.12 to 5.82. The result is mixed later: at
frame 40, excess increased from 4.92 to 10.05 and maximum normalized mismatch
from 6.12 to 15.14, although L1 mismatch remained lower (211.69 to 201.56).
Both variants completed without a fault; peak velocity was 116.32 versus
120.23 and peak Courant was 4.59 versus 4.55. This supports S4 as a reduction
of the early source error and worst row residual. It does not show that the
remaining phi/V mismatch is bounded or that sharpening is complete.
