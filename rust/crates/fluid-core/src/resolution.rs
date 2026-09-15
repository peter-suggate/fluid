//! Exact two-dimensional port of the Slice sparse-CM12 resolution policy.
//!
//! Arithmetic intentionally uses `f32` only at the TypeScript policy's
//! `Math.fround` boundaries. Geometry, ratios, and score construction remain
//! `f64`, matching JavaScript scalar arithmetic.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::topology::{BrickSeed, CompiledTopology, BRICK_FINE_RESOLUTION};
use crate::types::{Fields, RowKind};
use crate::{levelset_surface, presentation::RdfSurface};

const ACTIVITY_FIXED: f64 = 65_536.0;
const B: i32 = BRICK_FINE_RESOLUTION;
const VOLUME_ROUNDOFF_RATIO: f64 = 9.536_743_164_062_5e-7;

pub const SLICE_RESOLUTION_ACTIVITY_SOURCES_SHA256: &str =
    "6bc865d859c46f3d41343a0a34ebec4187e5bca45b8958ece4ba1afeab4265b1";
pub const SLICE_RESOLUTION_POLICY_SHA256: &str =
    "778e296e35e67a183422dbda7b2c95fdeb352b26b37da9bbf4eec18201fc3129";

pub mod activity_reason {
    pub const SURFACE: u32 = 1 << 0;
    pub const DEFORMATION: u32 = 1 << 1;
    pub const TEMPORAL: u32 = 1 << 2;
    pub const FINE_DETAIL: u32 = 1 << 3;
    pub const PREDICTED_FACE: u32 = 1 << 4;
    pub const FIRST_STEP: u32 = 1 << 5;
    pub const OCCUPIED: u32 = 1 << 6;
    pub const VELOCITY_FLOOR: u32 = 1 << 7;
    pub const THIN_FLUID: u32 = 1 << 8;
    pub const CUT_BOUNDARY: u32 = 1 << 9;
    pub const STATIC_BOUNDARY_B8: u32 = 1 << 10;
    pub const STATIC_BOUNDARY_B4: u32 = 1 << 11;
    pub const STATIC_BOUNDARY_B2: u32 = 1 << 12;
    pub const STATIC_BOUNDARY_B1: u32 = 1 << 13;
    pub const DENSITY_SURFACE: u32 = 1 << 14;
}

pub mod resolution_fault {
    pub const NONE: u32 = 0;
    pub const INVALID_RESOLUTION: u32 = 1 << 0;
    pub const TWO_TO_ONE: u32 = 1 << 1;
    pub const LEAF_CAPACITY: u32 = 1 << 2;
    pub const CELL_CAPACITY: u32 = 1 << 3;
    pub const MISSING_BACKING: u32 = 1 << 4;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(default)]
pub struct ActivityPolicy {
    pub activity_signals: bool,
    pub coarse_first: bool,
    pub energy_threshold: f64,
    pub curvature_tolerance: f64,
    pub anticipation_seconds: f64,
    pub anticipation_radius_bricks: i32,
    pub surface_quiet_epochs: u8,
    pub surface_displacement_tolerance_cells: f64,
    pub surface_normal_tolerance_degrees: f64,
    pub surface_coarsening_enabled: bool,
    pub forced_surface_resolution_for_qa: Option<u8>,
    pub freeze_topology: bool,
    pub legacy_face_transport_for_qa: bool,
    pub finest_travel_cells: f64,
    pub four_travel_cells: f64,
    pub two_travel_cells: f64,
    pub thin_feature_cells: f64,
    pub thin_feature_density: f64,
    pub residency_density: f64,
    pub residency_mass_fine_cells: f64,
    pub surface_density_minimum: f64,
    pub surface_density_maximum: f64,
    pub detail_tolerance: f64,
    pub front_lookahead_steps: u32,
    pub topology_cadence_steps: u32,
    pub prepare_bricks_per_frame: usize,
    pub promote_epochs: u8,
    pub demote_epochs: u8,
    pub promote_score: f64,
    pub demote_score: f64,
    pub emergency_score: f64,
}

impl Default for ActivityPolicy {
    fn default() -> Self {
        Self {
            activity_signals: true,
            coarse_first: true,
            energy_threshold: 8.0,
            curvature_tolerance: 0.25,
            anticipation_seconds: 0.5,
            anticipation_radius_bricks: 3,
            surface_quiet_epochs: 2,
            surface_displacement_tolerance_cells: 1.0,
            surface_normal_tolerance_degrees: 30.0,
            surface_coarsening_enabled: true,
            forced_surface_resolution_for_qa: None,
            freeze_topology: false,
            legacy_face_transport_for_qa: false,
            finest_travel_cells: 1.0,
            four_travel_cells: 0.5,
            two_travel_cells: 0.25,
            thin_feature_cells: 2.0,
            thin_feature_density: 0.0,
            residency_density: 0.005,
            residency_mass_fine_cells: 1.0,
            surface_density_minimum: 0.05,
            surface_density_maximum: 0.95,
            detail_tolerance: 0.08,
            front_lookahead_steps: 4,
            topology_cadence_steps: 1,
            prepare_bricks_per_frame: 64,
            promote_epochs: 2,
            demote_epochs: 1,
            promote_score: 160.0 / 255.0,
            demote_score: 96.0 / 255.0,
            emergency_score: 224.0 / 255.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolutionRegion {
    pub minimum_fine: [f64; 2],
    pub maximum_fine: [f64; 2],
    pub minimum_cell_width: u8,
    pub maximum_cell_width: Option<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfaceProofState {
    pub generation_by_target_resolution: BTreeMap<u8, u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrickActivityHistory {
    pub score_byte: u8,
    pub reasons: u32,
    pub hot_epochs: u8,
    pub quiet_epochs: u8,
    pub proof_epochs: u8,
    pub mean_density: f32,
    pub density_moments: [f32; 2],
    pub mean_velocity: [f32; 2],
    pub velocity_travel: f32,
    pub support_mask: u16,
    pub swept_support_mask: u16,
    pub last_transition_step: u32,
    pub surface_proof: Option<SurfaceProofState>,
}

impl Default for BrickActivityHistory {
    fn default() -> Self {
        Self {
            score_byte: 0,
            reasons: activity_reason::FIRST_STEP,
            hot_epochs: 0,
            quiet_epochs: 0,
            proof_epochs: 0,
            mean_density: 0.0,
            density_moments: [0.0; 2],
            mean_velocity: [0.0; 2],
            velocity_travel: 0.0,
            support_mask: 0,
            swept_support_mask: 0,
            last_transition_step: 0,
            surface_proof: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolutionPolicyState {
    pub accepted_steps: u32,
    pub accepted_generation: u32,
    pub scheduling_cursor: usize,
    pub scheduling_credits: usize,
    pub history: BTreeMap<u32, BrickActivityHistory>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(default)]
pub struct ResolutionPolicyOptions {
    pub policy: ActivityPolicy,
    /// Size coarse-first geometric cells from resolved velocity variation.
    /// Absolute translation still drives swept support and page activation.
    #[serde(skip)]
    pub translation_invariant_motion_sizing: bool,
    /// Let inactive metadata shed stale fine rungs before 2:1 closure.
    #[serde(skip)]
    pub coarsen_inactive_pages: bool,
    /// Allocate demanded pages at the coarsest donor-compatible rung.
    #[serde(skip)]
    pub coarsest_demanded_pages: bool,
    pub refinement_regions: Vec<ResolutionRegion>,
    pub static_boundary_floor_by_brick: BTreeMap<u32, u8>,
    pub moving_rigid_bodies: bool,
    pub injection_demanded_brick_keys: BTreeSet<u32>,
    pub frozen_brick_keys: BTreeSet<u32>,
    pub maximum_leaves: Option<usize>,
    pub maximum_cells: Option<usize>,
    pub allocate_missing_pages: bool,
    pub free_leaf_ids: Vec<u32>,
}

impl Default for ResolutionPolicyOptions {
    fn default() -> Self {
        Self {
            policy: ActivityPolicy::default(),
            translation_invariant_motion_sizing: false,
            coarsen_inactive_pages: false,
            coarsest_demanded_pages: false,
            refinement_regions: vec![],
            static_boundary_floor_by_brick: BTreeMap::new(),
            moving_rigid_bodies: false,
            injection_demanded_brick_keys: BTreeSet::new(),
            frozen_brick_keys: BTreeSet::new(),
            maximum_leaves: None,
            maximum_cells: None,
            allocate_missing_pages: true,
            free_leaf_ids: vec![],
        }
    }
}

impl ResolutionPolicyOptions {
    pub fn production() -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrickResolutionReceipt {
    pub brick_key: u32,
    pub accepted_resolution: u8,
    pub requested_resolution: u8,
    pub scheduled_resolution: u8,
    pub accepted_active: bool,
    pub candidate_active: bool,
    pub score_byte: u8,
    pub reasons: u32,
    pub plan_reasons: u32,
    pub support_mask: u16,
    pub swept_support_mask: u16,
    pub fault_bits: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolutionPolicyReceipt {
    pub topology_epoch: bool,
    pub accepted_generation: u32,
    pub candidate_generation: u32,
    pub measured_brick_count: usize,
    pub surface_brick_count: usize,
    pub occupied_brick_count: usize,
    pub activated_brick_count: usize,
    pub allocated_brick_count: usize,
    pub claimed_leaf_ids: Vec<u32>,
    pub retired_brick_count: usize,
    pub promoted_brick_count: usize,
    pub demoted_brick_count: usize,
    pub deferred_demotion_count: usize,
    pub maximum_score_byte: u8,
    pub fault_bits: u32,
    pub bricks: Vec<BrickResolutionReceipt>,
}

#[derive(Clone, Debug)]
pub struct ResolutionPolicyDecision {
    pub candidate_bricks: Vec<BrickSeed>,
    pub state: ResolutionPolicyState,
    pub receipt: ResolutionPolicyReceipt,
}

#[derive(Clone, Debug)]
pub struct ProjectedTransportSupportDecision {
    pub candidate_bricks: Vec<BrickSeed>,
    pub demanded_brick_keys: BTreeSet<u32>,
    pub claimed_leaf_ids: Vec<u32>,
    pub fault_bits: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolutionError {
    UnsupportedDimension,
    FieldShape,
    InvalidStep,
    InvalidResolution(u8),
    InvalidOrDuplicateFreeLeaf(u32),
    ActiveFreeLeaf(u32),
}

pub fn initialize_resolution_policy(topology: &CompiledTopology<2>) -> ResolutionPolicyState {
    ResolutionPolicyState {
        accepted_generation: topology.graph.topology_generation,
        history: topology
            .bricks
            .iter()
            .map(|b| (b.seed.key, BrickActivityHistory::default()))
            .collect(),
        ..ResolutionPolicyState::default()
    }
}

#[derive(Clone, Debug)]
struct Measurement {
    history: BrickActivityHistory,
    surface: bool,
    thin: bool,
    occupied: bool,
    detail: f32,
    deeply_enclosed: bool,
    curvature_floor: u8,
}

fn f(value: f64) -> f32 {
    value as f32
}
fn js_round(value: f64) -> f64 {
    (value + 0.5).floor()
}
fn valid_resolution(value: u8) -> Result<u8, ResolutionError> {
    match value {
        1 | 2 | 4 | 8 => Ok(value),
        _ => Err(ResolutionError::InvalidResolution(value)),
    }
}
fn span(brick: &BrickSeed) -> i32 {
    brick.span_bricks as i32
}
fn width(brick: &BrickSeed, rung: u8) -> f64 {
    (B * span(brick)) as f64 / rung as f64
}
fn bounds(brick: &BrickSeed) -> ([i32; 2], [i32; 2]) {
    let lo = [B * brick.coordinate[0], B * brick.coordinate[1]];
    let size = B * span(brick);
    (lo, [lo[0] + size, lo[1] + size])
}
fn overlaps(a0: i32, a1: i32, b0: i32, b1: i32) -> bool {
    a1.min(b1) > a0.max(b0)
}
fn face_neighbors(bricks: &[BrickSeed], index: usize) -> Vec<usize> {
    let (al, ah) = bounds(&bricks[index]);
    bricks
        .iter()
        .enumerate()
        .filter_map(|(j, b)| {
            if j == index {
                return None;
            }
            let (bl, bh) = bounds(b);
            (((ah[0] == bl[0] || bh[0] == al[0]) && overlaps(al[1], ah[1], bl[1], bh[1]))
                || ((ah[1] == bl[1] || bh[1] == al[1]) && overlaps(al[0], ah[0], bl[0], bh[0])))
            .then_some(j)
        })
        .collect()
}
fn owner_at(bricks: &[BrickSeed], x: f64, y: f64) -> Option<usize> {
    bricks.iter().position(|b| {
        let (lo, hi) = bounds(b);
        x >= lo[0] as f64 && x < hi[0] as f64 && y >= lo[1] as f64 && y < hi[1] as f64
    })
}
fn thresholds(policy: &ActivityPolicy, dt: f64, cell_size: f64) -> [f32; 4] {
    if policy.coarse_first {
        let finest = f((2.0 * policy.energy_threshold).sqrt() * dt / cell_size);
        [
            f(finest as f64 / 8.0),
            f(finest as f64 / 4.0),
            f(finest as f64 / 2.0),
            finest,
        ]
    } else {
        [
            f(policy.two_travel_cells / 2.0),
            f(policy.two_travel_cells),
            f(policy.four_travel_cells),
            f(policy.finest_travel_cells),
        ]
    }
}
fn velocity_floor(travel: f32, ts: [f32; 4], enabled: bool) -> u8 {
    if !enabled {
        return 1;
    }
    for i in (0..4).rev() {
        if travel >= ts[i] {
            return [1, 2, 4, 8][i];
        }
    }
    1
}
fn clamp_byte(value: f64) -> u8 {
    js_round(255.0 * value.clamp(0.0, 1.0)).clamp(0.0, 255.0) as u8
}
fn region_bounds(brick: &BrickSeed, regions: &[ResolutionRegion]) -> (u8, u8) {
    let (lo, hi) = bounds(brick);
    let nominal = (B * span(brick)) as f64;
    let (mut floor, mut ceiling) = (1_u8, 0_u8);
    for r in regions {
        let intersects = hi[0] as f64 > r.minimum_fine[0]
            && r.maximum_fine[0] > lo[0] as f64
            && hi[1] as f64 > r.minimum_fine[1]
            && r.maximum_fine[1] > lo[1] as f64;
        if intersects {
            floor = floor.max(r.minimum_cell_width);
        }
        let contained = lo[0] as f64 >= r.minimum_fine[0]
            && lo[1] as f64 >= r.minimum_fine[1]
            && hi[0] as f64 <= r.maximum_fine[0]
            && hi[1] as f64 <= r.maximum_fine[1];
        if contained {
            if let Some(c) = r.maximum_cell_width {
                ceiling = if ceiling == 0 { c } else { ceiling.min(c) };
            }
        }
    }
    let maximum = (nominal / floor as f64).clamp(1.0, 8.0) as u8;
    let minimum = if ceiling == 0 {
        1
    } else {
        (nominal / ceiling as f64).clamp(1.0, 8.0) as u8
    };
    (maximum, minimum)
}
fn apply_regions(requested: u8, brick: &BrickSeed, regions: &[ResolutionRegion]) -> u8 {
    let (max, min) = region_bounds(brick, regions);
    requested.min(max).max(min)
}

fn record_donor_compatible_rung(
    required: &mut BTreeMap<u32, u8>,
    receiver: &BrickSeed,
    donor: &BrickSeed,
    donor_required_rung: Option<u8>,
) {
    let donor_width = width(donor, donor.resolution);
    let maximum_receiver_width = donor_required_rung.map_or(2.0 * donor_width, |rung| {
        (2.0 * donor_width).min(width(donor, rung))
    });
    let mut rung = 1;
    while rung < 8 && width(receiver, rung) > maximum_receiver_width {
        rung *= 2;
    }
    required
        .entry(receiver.key)
        .and_modify(|current| *current = (*current).max(rung))
        .or_insert(rung);
}

fn donor_transport_rung(measurement: &Measurement, policy: &ActivityPolicy, ts: [f32; 4]) -> u8 {
    let motion = velocity_floor(
        measurement.history.velocity_travel,
        ts,
        policy.activity_signals,
    );
    measurement
        .curvature_floor
        .max(motion)
        .max(if measurement.thin { 8 } else { 1 })
}

fn accumulate_direct_surface_normals(
    brick: &BrickSeed,
    surface: &RdfSurface,
    normal_min: &mut [f64; 2],
    normal_max: &mut [f64; 2],
) -> bool {
    let nx = surface.dimensions[0] as usize;
    let ny = surface.dimensions[1] as usize;
    let stride = nx + 1;
    let (lo, hi) = bounds(brick);
    let x0 = lo[0].max(0) as usize;
    let y0 = lo[1].max(0) as usize;
    let x1 = hi[0].max(0).min(nx as i32) as usize;
    let y1 = hi[1].max(0).min(ny as i32) as usize;
    let mut crossed = false;
    for y in y0..y1 {
        for x in x0..x1 {
            let phi = [
                surface.vertex_phi_fine[x + stride * y] as f64,
                surface.vertex_phi_fine[x + 1 + stride * y] as f64,
                surface.vertex_phi_fine[x + 1 + stride * (y + 1)] as f64,
                surface.vertex_phi_fine[x + stride * (y + 1)] as f64,
            ];
            for triangle in levelset_surface::triangles(x as f64, y as f64, phi) {
                if !triangle.iter().any(|vertex| vertex[2] < 0.0)
                    || !triangle.iter().any(|vertex| vertex[2] >= 0.0)
                {
                    continue;
                }
                crossed = true;
                let [a, b, c] = triangle;
                let ab = [b[0] - a[0], b[1] - a[1]];
                let ac = [c[0] - a[0], c[1] - a[1]];
                let det = ab[0] * ac[1] - ab[1] * ac[0];
                if det.abs() <= f64::EPSILON {
                    continue;
                }
                let bp = b[2] - a[2];
                let cp = c[2] - a[2];
                let gradient = [
                    (bp * ac[1] - cp * ab[1]) / det,
                    (ab[0] * cp - ac[0] * bp) / det,
                ];
                let length = gradient[0].hypot(gradient[1]);
                if !length.is_finite() || length <= 1.0e-12 {
                    continue;
                }
                for axis in 0..2 {
                    let normal = gradient[axis] / length;
                    normal_min[axis] = normal_min[axis].min(normal);
                    normal_max[axis] = normal_max[axis].max(normal);
                }
            }
        }
    }
    crossed
}

/// Publish the next-rung certificate after accepted direct-surface publication.
/// Certify the next-rung bilinear reconstruction against the accepted phi.
/// Existing V/phi mismatch is not incremental restriction error. Thin features,
/// deformation, region bounds and solid geometry remain independent guards.
pub fn publish_direct_surface_proofs(
    topology: &CompiledTopology<2>,
    fields: &Fields,
    surface: &RdfSurface,
    options: &ResolutionPolicyOptions,
    state: &mut ResolutionPolicyState,
    dt: f64,
    cell_size: f64,
) -> Result<(), ResolutionError> {
    validate_inputs(topology, fields, dt, cell_size)?;
    validate_direct_surface(topology, Some(surface))?;
    let policy = &options.policy;
    for brick in &topology.bricks {
        let history = state.history.entry(brick.seed.key).or_default();
        history.surface_proof = None;
        // GPU surface-proof publication is gated by the accepted SURFACE
        // reason. Empty/bulk quiet epochs belong to ordinary coarsening and
        // must survive this publication even when no surface proof exists.
        if history.reasons & activity_reason::SURFACE == 0 { continue; }
        let target = (brick.seed.resolution / 2).max(1);
        if !brick.seed.active || brick.seed.resolution <= 1 || brick.seed.span_bricks != 1
            || !policy.activity_signals || !policy.surface_coarsening_enabled
            || options.injection_demanded_brick_keys.contains(&brick.seed.key)
            || policy.forced_surface_resolution_for_qa.is_some()
            || history.reasons & activity_reason::THIN_FLUID != 0
            || velocity_floor(history.velocity_travel, thresholds(policy, dt, cell_size),
                policy.activity_signals) > target
            || options.static_boundary_floor_by_brick.get(&brick.seed.key)
                .is_some_and(|&floor| floor > target)
            || apply_regions(target, &brick.seed, &options.refinement_regions) != target
        {
            history.proof_epochs = 0;
            continue;
        }
        let step = width(&brick.seed, target);
        let (lo, _) = bounds(&brick.seed);
        // Certify positional error of the candidate bilinear reconstruction.
        // Existing V/phi disagreement and noisy normals are not damage caused
        // by this restriction; conservative remapping retains extensive V.
        let sample = |p: [f64; 2]| crate::levelset_redistance::sample_scalar(
            surface, p.map(|v| v as f32)).unwrap() as f64;
        let tolerance = policy.surface_displacement_tolerance_cells;
        let mut valid = true;
        for y in 0..8 { for x in 0..8 {
            let p = [lo[0] as f64 + x as f64 + 0.5,
                lo[1] as f64 + y as f64 + 0.5];
            let lower = p.map(|v| (v / step).floor() * step);
            let t = [(p[0]-lower[0])/step, (p[1]-lower[1])/step];
            let fine = sample(p);
            let mut coarse = 0.0;
            let mut minimum = f64::INFINITY;
            let mut maximum = f64::NEG_INFINITY;
            for cy in 0..2 { for cx in 0..2 {
                let weight = if cx == 0 {1.0-t[0]} else {t[0]}
                    * if cy == 0 {1.0-t[1]} else {t[1]};
                let corner = sample([lower[0]+cx as f64*step,
                    lower[1]+cy as f64*step]);
                minimum = minimum.min(corner); maximum = maximum.max(corner);
                coarse += weight * corner;
            }}
            valid &= fine.is_finite() && coarse.is_finite();
            // A same-sign candidate box cannot erase an enclosed feature.
            valid &= !(fine < 0.0 && minimum >= 0.0)
                && !(fine > 0.0 && maximum < 0.0);
            if (fine < 0.0) != (coarse < 0.0) {
                valid &= fine.abs().min(coarse.abs()) <= tolerance;
            }
        }}
        if valid {
            history.surface_proof = Some(SurfaceProofState {
                generation_by_target_resolution: BTreeMap::from([
                    (target, topology.graph.topology_generation)]),
            });
        } else {
            history.proof_epochs = 0;
        }
    }
    Ok(())
}

fn constrained_demanded_rung(
    brick: &BrickSeed,
    donor_rung: u8,
    options: &ResolutionPolicyOptions,
    retain_current: bool,
) -> u8 {
    if options.policy.freeze_topology || options.frozen_brick_keys.contains(&brick.key) {
        return brick.resolution;
    }
    let static_floor = options
        .static_boundary_floor_by_brick
        .get(&brick.key)
        .copied()
        .unwrap_or(1);
    let requested = donor_rung
        .max(static_floor)
        .max(if retain_current { brick.resolution } else { 1 });
    apply_regions(requested, brick, &options.refinement_regions)
}

fn measure(
    topology: &CompiledTopology<2>,
    fields: &Fields,
    bi: usize,
    old: Option<&BrickActivityHistory>,
    policy: &ActivityPolicy,
    topology_epoch: bool,
    dt: f64,
    ts: [f32; 4],
    translation_invariant_motion_sizing: bool,
    direct_surface: Option<&RdfSurface>,
) -> Measurement {
    let brick = &topology.bricks[bi];
    if !brick.seed.active {
        let mut h = old.cloned().unwrap_or_default();
        h.score_byte = 0;
        h.reasons &= 0x3c00;
        h.mean_density = 0.0;
        h.density_moments = [0.0; 2];
        h.mean_velocity = [0.0; 2];
        h.velocity_travel = 0.0;
        h.support_mask = 0;
        h.swept_support_mask = 0;
        return Measurement {
            history: h,
            surface: false,
            thin: false,
            occupied: false,
            detail: 0.0,
            deeply_enclosed: false,
            curvature_floor: 1,
        };
    }
    let cells =
        &topology.graph.cells[brick.cell_range.start as usize..brick.cell_range.end as usize];
    let mut density_sum = 0.0_f64;
    let mut moment = [0.0_f64; 2];
    let mut deformation = 0.0_f32;
    let mut predicted = 0.0_f32;
    let mut detail = 0.0_f32;
    let mut travel = 0.0_f32;
    let mut relative_travel = 0.0_f32;
    let mut approach_low = [f64::NEG_INFINITY; 2];
    let mut approach_high = [f64::INFINITY; 2];
    let (page_lo, page_hi) = bounds(&brick.seed);
    let page_centre = [(page_lo[0]+page_hi[0]) as f64*0.5,
        (page_lo[1]+page_hi[1]) as f64*0.5];
    let mut motion_samples = Vec::new();
    let mut axes = 0_u8;
    let mut occupied_cell = false;
    let mut substantial = false;
    let mut thin = false;
    let mut cut = false;
    let mut density_surface = false;
    let mut support_mask = 0_u16;
    let mut swept_mask = 0_u16;
    let mut momentum = [0.0_f32; 2];
    let mut momentum_mass = 0.0_f32;
    let (mut normal_min, mut normal_max) = ([1.0_f64; 2], [-1.0_f64; 2]);
    let feature_density = policy.residency_density.max(policy.thin_feature_density);
    for (local_index, cell) in cells.iter().enumerate() {
        let id = cell.id as usize;
        let rho = fields.density[id] as f64;
        let cap = (fields.capacity[id] as f64).max(1e-6);
        let fill = rho / cap;
        density_sum += js_round(rho * ACTIVITY_FIXED);
        let local = [
            local_index % brick.seed.resolution as usize,
            local_index / brick.seed.resolution as usize,
        ];
        for axis in 0..2 {
            moment[axis] += js_round(
                rho * (2 * local[axis] as i32 + 1 - brick.seed.resolution as i32) as f64
                    / brick.seed.resolution as f64
                    * ACTIVITY_FIXED,
            );
        }
        cut |= fields.capacity[id] < 0.999;
        occupied_cell |= rho > policy.residency_density;
        substantial |= fill > policy.surface_density_minimum;
        let own_phi = direct_surface.and_then(|surface|
            crate::levelset_redistance::sample_scalar(surface, [cell.center[0], cell.center[1]]));
        let wet = own_phi.map_or(fill >= 0.5, |phi| phi < 0.0);
        let vx = fields.cell_velocity[2 * id];
        let vy = fields.cell_velocity[2 * id + 1];
        if policy.coarse_first && wet {
            momentum[0] = f(momentum[0] as f64 + f(vx as f64 * rho) as f64);
            momentum[1] = f(momentum[1] as f64 + f(vy as f64 * rho) as f64);
            momentum_mass = f(momentum_mass as f64 + rho);
        }
        let mut swept_min = [0.0_f64; 2];
        let mut swept_max = [0.0_f64; 2];
        let transport_roundoff = VOLUME_ROUNDOFF_RATIO * cap * cell.measure as f64;
        for &row_id in &topology.graph.incidences[id] {
            let row = &topology.graph.rows[row_id as usize];
            let axis = row.axis as usize;
            let open = row.open_fraction as f64;
            let wall = row.solid_velocity as f64;
            let stored = fields
                .face_velocity
                .get(row.id as usize)
                .copied()
                .unwrap_or(if axis == 0 { vx } else { vy }) as f64;
            let fluid = if open > 1e-6 {
                (stored - (1.0 - open) * wall) / open
            } else {
                wall
            };
            let endpoint = fluid + dt * fields.acceleration_fine[axis] as f64;
            for candidate in [fluid, endpoint] {
                if dt * row.measure as f64 * candidate.abs() <= transport_roundoff {
                    continue;
                }
                swept_min[axis] = swept_min[axis].min(candidate);
                swept_max[axis] = swept_max[axis].max(candidate);
            }
        }
        let mut interface_cell = false;
        let mut exposed = 0_u8;
        for &row_id in &topology.graph.incidences[id] {
            let row = &topology.graph.rows[row_id as usize];
            let axis = row.axis as usize;
            let own = row
                .terms
                .iter()
                .find(|t| t.cell_id == cell.id)
                .expect("incidence term");
            let others: Vec<_> = row
                .terms
                .iter()
                .filter(|t| t.cell_id != cell.id && own.coefficient * t.coefficient < 0.0)
                .collect();
            let mut side_has_fluid = false;
            for term in &others {
                let nid = term.cell_id as usize;
                let neighbor_fill =
                    fields.density[nid] as f64 / (fields.capacity[nid] as f64).max(1e-6);
                side_has_fluid |= neighbor_fill > feature_density;
                let neighbor_wet = direct_surface.map_or(neighbor_fill >= 0.5, |surface| {
                    let neighbor = &topology.graph.cells[nid];
                    crate::levelset_redistance::sample_scalar(surface,
                        [neighbor.center[0], neighbor.center[1]]).is_some_and(|phi| phi < 0.0)
                });
                if policy.coarse_first && direct_surface.is_some()
                    && neighbor_fill > feature_density {
                    let position = topology.graph.cells[nid].center[axis] as f64;
                    let velocity = fields.cell_velocity[2*nid+axis] as f64;
                    if position < page_centre[axis] {
                        approach_low[axis] = approach_low[axis].max(velocity);
                    } else {
                        approach_high[axis] = approach_high[axis].min(velocity);
                    }
                }
                let crosses = neighbor_wet != wet;
                let same_brick = topology.graph.cells[nid].brick_key == Some(brick.seed.key);
                if crosses && (!wet || same_brick || policy.coarse_first) {
                    interface_cell = true;
                    density_surface = true;
                    axes |= 1 << axis;
                }
                if crosses {
                    let liquid = if wet { id } else { nid };
                    let v = fields.cell_velocity[2 * liquid + axis] as f64;
                    predicted =
                        predicted.max(f(dt * v.abs() / (0.25 * row.distance as f64).max(1e-12)));
                }
                if wet && neighbor_wet {
                    let nvx = fields.cell_velocity[2 * nid];
                    let nvy = fields.cell_velocity[2 * nid + 1];
                    let dv = (vx as f64 - nvx as f64)
                        .abs()
                        .max((vy as f64 - nvy as f64).abs());
                    deformation =
                        deformation.max(f(dt * dv / (0.15 * row.distance as f64).max(1e-12)));
                    if translation_invariant_motion_sizing {
                        relative_travel = relative_travel.max(f(
                            dt * (vx as f64 - nvx as f64).hypot(vy as f64 - nvy as f64),
                        ));
                    }
                }
            }
            if row.kind == RowKind::SparseAir && others.is_empty() && wet {
                interface_cell = true;
                axes |= 1 << axis;
                let v = fields.cell_velocity[2 * id + axis] as f64;
                predicted =
                    predicted.max(f(dt * v.abs() / (0.25 * row.distance as f64).max(1e-12)));
            }
            if rho > feature_density && !side_has_fluid {
                let side = usize::from(row.center[axis] > cell.center[axis]);
                exposed |= 1 << (2 * axis + side);
            }
        }
        let thickness = rho.clamp(0.0, 1.0) * (cell.widths[0] as f64).min(cell.widths[1] as f64);
        let cell_thin = own_phi.map_or(
            fill > feature_density && thickness < policy.thin_feature_cells,
            |phi| wet && -(phi as f64) <= 0.5 * policy.thin_feature_cells)
            && ((exposed & 3) == 3 || (exposed & 12) == 12);
        thin |= cell_thin;
        if direct_surface.is_none()
            && policy.coarse_first
            && (interface_cell || cell_thin)
            && fields.interface_normal.len() >= 2 * id + 2
        {
            let n = [
                fields.interface_normal[2 * id] as f64,
                fields.interface_normal[2 * id + 1] as f64,
            ];
            if n[0] * n[0] + n[1] * n[1] > 0.5 {
                for a in 0..2 {
                    normal_min[a] = normal_min[a].min(n[a]);
                    normal_max[a] = normal_max[a].max(n[a]);
                }
            }
        }
        // Uniform translation of submerged liquid does not create material
        // detail.  Use velocity as a resolution floor only where a represented
        // interface (or a thin feature) is moving; swept-support planning below
        // still follows every nonzero donor into its receiver pages.  Including
        // every wet cell here makes a falling, otherwise rigid liquid body
        // refine its entire interior solely because its absolute speed grows.
        if (interface_cell && wet)
            || cell_thin
            || (policy.coarse_first && wet
                && (!translation_invariant_motion_sizing || direct_surface.is_some()))
        {
            travel = travel.max(f(dt * (vx as f64).hypot(vy as f64)));
            if translation_invariant_motion_sizing || direct_surface.is_some() {
                motion_samples.push([vx, vy]);
            }
        }
        if interface_cell || cell_thin || rho != 0.0 {
            let (blo, bhi) = bounds(&brick.seed);
            let cx = cell.center[0] as f64;
            let cy = cell.center[1] as f64;
            let touches_low_x = cx - 0.5 * cell.widths[0] as f64 <= blo[0] as f64;
            let touches_high_x = cx + 0.5 * cell.widths[0] as f64 >= bhi[0] as f64;
            let dxs: &[i32] = if translation_invariant_motion_sizing {
                match (touches_low_x, touches_high_x) {
                    (true, true) => &[-1, 0, 1],
                    (true, false) => &[-1, 0],
                    (false, true) => &[0, 1],
                    (false, false) => &[0],
                }
            } else if touches_low_x {
                &[-1, 0]
            } else if touches_high_x {
                &[0, 1]
            } else {
                &[0]
            };
            let touches_low_y = cy - 0.5 * cell.widths[1] as f64 <= blo[1] as f64;
            let touches_high_y = cy + 0.5 * cell.widths[1] as f64 >= bhi[1] as f64;
            let dys: &[i32] = if translation_invariant_motion_sizing {
                match (touches_low_y, touches_high_y) {
                    (true, true) => &[-1, 0, 1],
                    (true, false) => &[-1, 0],
                    (false, true) => &[0, 1],
                    (false, false) => &[0],
                }
            } else if touches_low_y {
                &[-1, 0]
            } else if touches_high_y {
                &[0, 1]
            } else {
                &[0]
            };
            for &dy in dys {
                for &dx in dxs {
                    if dx != 0 || dy != 0 {
                        let bit = (dx + 1 + 3 * (dy + 1)) as u32;
                        if interface_cell || cell_thin {
                            support_mask |= 1 << bit;
                        }
                    }
                }
            }
            if rho != 0.0 {
                swept_mask |= 1 << 4;
                let min_dx = if cx - 0.5 * cell.widths[0] as f64 + dt * swept_min[0] < blo[0] as f64
                {
                    -1
                } else {
                    0
                };
                let max_dx = if cx + 0.5 * cell.widths[0] as f64 + dt * swept_max[0] > bhi[0] as f64
                {
                    1
                } else {
                    0
                };
                let min_dy = if cy - 0.5 * cell.widths[1] as f64 + dt * swept_min[1] < blo[1] as f64
                {
                    -1
                } else {
                    0
                };
                let max_dy = if cy + 0.5 * cell.widths[1] as f64 + dt * swept_max[1] > bhi[1] as f64
                {
                    1
                } else {
                    0
                };
                for dy in min_dy..=max_dy {
                    for dx in min_dx..=max_dx {
                        if dx != 0 || dy != 0 {
                            let bit = (dx + 1 + 3 * (dy + 1)) as u32;
                            support_mask |= 1 << bit;
                            swept_mask |= 1 << bit;
                        }
                    }
                }
            }
        }
    }
    let direct_surface_crossing = direct_surface.is_some_and(|surface| {
        accumulate_direct_surface_normals(
            &brick.seed,
            surface,
            &mut normal_min,
            &mut normal_max,
        )
    });
    if brick.seed.resolution > 1 {
        let base = brick.cell_range.start as usize;
        let r = brick.seed.resolution as usize;
        for y in (0..r).step_by(2) {
            for x in (0..r).step_by(2) {
                let values = [0, 1, r, r + 1].map(|o| {
                    js_round(fields.density[base + x + r * y + o] as f64 * ACTIVITY_FIXED)
                });
                let sum: f64 = values.iter().sum();
                for value in values {
                    detail = detail.max(f((4.0 * value - sum).abs() / (4.0 * ACTIVITY_FIXED)));
                }
            }
        }
    }
    let count = cells.len().max(1) as f64;
    let mean_density = f(density_sum / (count * ACTIVITY_FIXED));
    let moments = [
        f(moment[0] / (count * ACTIVITY_FIXED)),
        f(moment[1] / (count * ACTIVITY_FIXED)),
    ];
    let mass_fine =
        f(density_sum / ACTIVITY_FIXED * cells.first().map_or(0.0, |c| c.measure as f64));
    let mean_velocity = if momentum_mass > 1e-8 {
        [
            f(momentum[0] as f64 / momentum_mass as f64),
            f(momentum[1] as f64 / momentum_mass as f64),
        ]
    } else {
        [0.0; 2]
    };
    // A signed-distance ball containing every cell certifies flooded bulk,
    // including bricks against a closed wall. Uniform submerged translation
    // is support demand, not a liquid-air resolution feature.
    let direct_deep_liquid = direct_surface.is_some_and(|surface| {
        !cells.is_empty() && cells.iter().all(|cell| {
            crate::levelset_redistance::sample_scalar(surface,
                [cell.center[0], cell.center[1]]).is_some_and(|phi| {
                phi < 0.0 && (surface.segments_fine.is_empty()
                    || -phi >= 0.5 * cell.widths[0].hypot(cell.widths[1]))
            })
        })
    });
    if direct_deep_liquid && !direct_surface_crossing { travel = 0.0; }
    if policy.coarse_first && direct_surface.is_some() {
        let mut minimum = [f64::INFINITY; 2];
        let mut maximum = [f64::NEG_INFINITY; 2];
        for velocity in motion_samples {
            for axis in 0..2 {
                minimum[axis] = minimum[axis].min(velocity[axis] as f64);
                maximum[axis] = maximum[axis].max(velocity[axis] as f64);
            }
        }
        travel = f(dt * (maximum[0]-minimum[0]).max(0.0)
            .hypot((maximum[1]-minimum[1]).max(0.0)));
        if direct_deep_liquid && !direct_surface_crossing { travel = 0.0; }
    } else if policy.coarse_first && translation_invariant_motion_sizing {
        travel = relative_travel;
        for velocity in motion_samples {
            travel = travel.max(f(dt * (velocity[0] as f64-mean_velocity[0] as f64)
                .hypot(velocity[1] as f64-mean_velocity[1] as f64)));
        }
    }
    if policy.coarse_first && direct_surface.is_some() {
        travel = travel.max(f(dt * (approach_low[0]-approach_high[0]).max(0.0)
            .hypot((approach_low[1]-approach_high[1]).max(0.0))));
    }
    let represented = substantial || thin;
    let occupied =
        occupied_cell && represented && mass_fine as f64 >= policy.residency_mass_fine_cells;
    let surface = direct_surface.map_or(occupied && axes != 0, |_| direct_surface_crossing);
    let shape = if axes.count_ones() >= 2 { 1.0_f32 } else { 0.0 };
    let temporal = if !policy.coarse_first {
        old.map_or(0.0, |o| {
            ((mean_density as f64 - o.mean_density as f64).abs() / 0.05)
                .max((moments[0] as f64 - o.density_moments[0] as f64).abs() / 0.02)
                .max((moments[1] as f64 - o.density_moments[1] as f64).abs() / 0.02)
        })
    } else {
        0.0
    };
    let scored_detail = if surface && !thin && shape == 0.0 {
        0.0
    } else {
        detail as f64
    };
    let dynamic = if surface || thin {
        (deformation as f64).max(temporal)
    } else {
        0.0
    };
    let feature = dynamic
        .max(predicted as f64)
        .max(shape as f64)
        .max(if thin { 1.0 } else { 0.0 })
        .max((scored_detail / policy.detail_tolerance - 1.0).max(0.0));
    let scored_velocity = if surface || thin {
        travel as f64 / (ts[3] as f64).max(1e-6)
    } else {
        0.0
    };
    let normal_diameter = if normal_max[0] >= normal_min[0] && normal_max[1] >= normal_min[1] {
        (normal_max[0] - normal_min[0])
            .max(0.0)
            .hypot((normal_max[1] - normal_min[1]).max(0.0))
    } else {
        0.0
    };
    let mut curvature_floor = 1_u8;
    while direct_surface.is_none() && curvature_floor < 8
        && normal_diameter / curvature_floor as f64 > policy.curvature_tolerance
    {
        curvature_floor *= 2;
    }
    let coarse_score = (normal_diameter
        / (brick.seed.resolution as f64 * policy.curvature_tolerance.max(0.02)))
    .max(travel as f64 / (ts[3] as f64).max(1e-6));
    let score_byte = clamp_byte(if policy.coarse_first {
        coarse_score
    } else {
        scored_velocity.max(feature)
    });
    let mut reasons = 0_u32;
    if surface {
        reasons |= activity_reason::SURFACE;
    }
    if deformation as f64 >= policy.emergency_score {
        reasons |= activity_reason::DEFORMATION;
    }
    if temporal > 0.0 {
        reasons |= activity_reason::TEMPORAL;
    }
    if detail as f64 > policy.detail_tolerance {
        reasons |= activity_reason::FINE_DETAIL;
    }
    if predicted as f64 >= policy.emergency_score {
        reasons |= activity_reason::PREDICTED_FACE;
    }
    if old.is_none() {
        reasons |= activity_reason::FIRST_STEP;
    }
    if occupied {
        reasons |= activity_reason::OCCUPIED;
    }
    if velocity_floor(travel, ts, policy.activity_signals) > 1 {
        reasons |= activity_reason::VELOCITY_FLOOR;
    }
    if thin {
        reasons |= activity_reason::THIN_FLUID;
    }
    if cut {
        reasons |= activity_reason::CUT_BOUNDARY;
    }
    if density_surface {
        reasons |= activity_reason::DENSITY_SURFACE;
    }
    if policy.coarse_first {
        reasons |= (curvature_floor as u32) << 16;
    }
    let hot = policy.activity_signals && feature >= policy.promote_score;
    let quiet = !thin
        && score_byte as f64 / 255.0 <= policy.demote_score
        && scored_detail <= policy.detail_tolerance;
    let hot_epochs = if topology_epoch {
        if hot {
            old.map_or(0, |o| o.hot_epochs).saturating_add(1)
        } else {
            0
        }
    } else {
        old.map_or(0, |o| o.hot_epochs)
    };
    let quiet_epochs = if topology_epoch {
        if quiet {
            old.map_or(0, |o| o.quiet_epochs).saturating_add(1)
        } else {
            0
        }
    } else {
        old.map_or(0, |o| o.quiet_epochs)
    };
    let (own_lo, own_hi) = bounds(&brick.seed);
    let neighbors = face_neighbors(
        &topology
            .bricks
            .iter()
            .map(|r| r.seed.clone())
            .collect::<Vec<_>>(),
        bi,
    );
    let mut enclosed_sides = [false; 4];
    for &ni in &neighbors {
        let (lo, hi) = bounds(&topology.bricks[ni].seed);
        if hi[0] == own_lo[0] {
            enclosed_sides[0] = true;
        }
        if lo[0] == own_hi[0] {
            enclosed_sides[1] = true;
        }
        if hi[1] == own_lo[1] {
            enclosed_sides[2] = true;
        }
        if lo[1] == own_hi[1] {
            enclosed_sides[3] = true;
        }
    }
    let deeply_enclosed = occupied
        && !direct_surface_crossing
        && enclosed_sides.iter().all(|&v| v)
        && neighbors.iter().all(|&ni| {
            let nb = &topology.bricks[ni];
            nb.seed.active
                && nb.cell_range.start < nb.cell_range.end
                && (nb.cell_range.clone()).all(|id| {
                    fields.density[id as usize] as f64
                        / (fields.capacity[id as usize] as f64).max(1e-6)
                        >= 0.5
                })
        });
    Measurement {
        history: BrickActivityHistory {
            score_byte,
            reasons,
            hot_epochs,
            quiet_epochs,
            proof_epochs: old.map_or(0, |o| o.proof_epochs),
            mean_density,
            density_moments: moments,
            mean_velocity,
            velocity_travel: travel,
            support_mask,
            swept_support_mask: swept_mask,
            last_transition_step: old.map_or(0, |o| o.last_transition_step),
            surface_proof: old.and_then(|o| o.surface_proof.clone()),
        },
        surface,
        thin,
        occupied,
        detail,
        deeply_enclosed,
        curvature_floor,
    }
}

fn directional_demand(
    bricks: &[BrickSeed],
    measurements: &BTreeMap<u32, Measurement>,
    include_interface_support: bool,
) -> BTreeSet<u32> {
    let mut demanded = BTreeSet::new();
    for source in bricks.iter().filter(|b| b.active) {
        let mask = measurements.get(&source.key).map_or(0, |m| {
            m.history.swept_support_mask
                | if include_interface_support {
                    m.history.support_mask
                } else {
                    0
                }
        });
        for bit in 0..9_i32 {
            if bit == 4 || mask & (1 << bit) == 0 {
                continue;
            }
            let dx = bit % 3 - 1;
            let dy = bit / 3 - 1;
            let s = span(source);
            let qx = source.coordinate[0]
                + if dx < 0 {
                    -1
                } else if dx > 0 {
                    s
                } else {
                    0
                };
            let qy = source.coordinate[1]
                + if dy < 0 {
                    -1
                } else if dy > 0 {
                    s
                } else {
                    0
                };
            if let Some(i) = owner_at(bricks, (qx * B) as f64 + 0.5, (qy * B) as f64 + 0.5) {
                if bricks[i].key != source.key {
                    demanded.insert(bricks[i].key);
                }
            }
        }
    }
    demanded
}

fn approach_travel(delta: [f64; 2], sweep: [f64; 2], extent: f64) -> f64 {
    let (mut enter, mut leave, mut approach) = (0.0_f64, 1.0_f64, 0.0_f64);
    for axis in 0..2 {
        let distance = delta[axis].abs();
        let motion = sweep[axis];
        if distance >= extent {
            let closing = delta[axis].signum() * motion;
            if closing <= 0.0 {
                return 0.0;
            }
            let axis_enter = (distance - extent) / closing;
            if axis_enter > enter || approach == 0.0 {
                approach = closing;
            } else if axis_enter == enter {
                approach = approach.min(closing);
            }
            enter = enter.max(axis_enter);
            leave = leave.min((distance + extent) / closing);
        } else if motion.abs() > 1e-8 {
            leave = leave.min((extent + motion.signum() * delta[axis]) / motion.abs());
        }
    }
    if enter < leave {
        approach
    } else {
        0.0
    }
}

fn incoming_floor(
    topology: &CompiledTopology<2>,
    bi: usize,
    measurements: &BTreeMap<u32, Measurement>,
    policy: &ActivityPolicy,
) -> u8 {
    if policy.anticipation_seconds <= 0.0 {
        return 1;
    }
    let brick = &topology.bricks[bi].seed;
    let (lo, hi) = bounds(brick);
    let center = [0.5 * (lo[0] + hi[0]) as f64, 0.5 * (lo[1] + hi[1]) as f64];
    let receiver = &measurements[&brick.key];
    let mut required = 1;
    for source in &topology.bricks {
        let source = &source.seed;
        if !source.active || source.key == brick.key {
            continue;
        }
        if (source.coordinate[0] - brick.coordinate[0]).abs() > policy.anticipation_radius_bricks
            || (source.coordinate[1] - brick.coordinate[1]).abs()
                > policy.anticipation_radius_bricks
        {
            continue;
        }
        let m = &measurements[&source.key];
        if !m.occupied || !(m.surface || m.thin) {
            continue;
        }
        let (sl, sh) = bounds(source);
        let sc = [0.5 * (sl[0] + sh[0]) as f64, 0.5 * (sl[1] + sh[1]) as f64];
        let sweep = [
            policy.anticipation_seconds
                * (m.history.mean_velocity[0] - receiver.history.mean_velocity[0]) as f64,
            policy.anticipation_seconds
                * (m.history.mean_velocity[1] - receiver.history.mean_velocity[1]) as f64,
        ];
        if sweep[0].hypot(sweep[1]) <= 1.0 {
            continue;
        }
        let delta = [center[0] - sc[0], center[1] - sc[1]];
        let extent = 0.5 * B as f64 * (span(brick) + span(source)) as f64;
        let approach = approach_travel(delta, sweep, extent);
        if approach <= 1.0 {
            continue;
        }
        let gap = (delta[0].abs() - extent)
            .max(0.0)
            .hypot((delta[1].abs() - extent).max(0.0));
        let demand = B as f64 * (approach / (B as f64).max(gap + B as f64)).min(1.0);
        let mut rung = 1;
        while rung < 8 && (rung as f64) < demand {
            rung *= 2;
        }
        required = required.max(rung);
    }
    required
}

fn close_region_caps(
    bricks: &[BrickSeed],
    targets: &mut BTreeMap<u32, u8>,
    regions: &[ResolutionRegion],
) {
    if regions.is_empty() {
        return;
    }
    let mut caps: BTreeMap<_, _> = bricks
        .iter()
        .map(|b| (b.key, region_bounds(b, regions).0))
        .collect();
    loop {
        let mut changed = false;
        for i in 0..bricks.len() {
            for j in face_neighbors(bricks, i) {
                let propagated = (2 * caps[&bricks[j].key] as i32 * span(&bricks[i])
                    / span(&bricks[j]))
                .clamp(1, 8) as u8;
                if propagated < caps[&bricks[i].key] {
                    caps.insert(bricks[i].key, propagated);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for b in bricks {
        targets.insert(b.key, targets[&b.key].min(caps[&b.key]));
    }
}

fn close_two_to_one(
    bricks: &[BrickSeed],
    targets: &mut BTreeMap<u32, u8>,
    regions: &[ResolutionRegion],
) {
    loop {
        let mut changed = false;
        for i in 0..bricks.len() {
            for j in face_neighbors(bricks, i) {
                if bricks[j].key < bricks[i].key {
                    continue;
                }
                let aw = width(&bricks[i], targets[&bricks[i].key]);
                let bw = width(&bricks[j], targets[&bricks[j].key]);
                if aw.max(bw) <= 2.0 * aw.min(bw) {
                    continue;
                }
                let ci = if aw > bw { i } else { j };
                let current = targets[&bricks[ci].key];
                let raised = apply_regions((current * 2).min(8), &bricks[ci], regions);
                if raised != current {
                    targets.insert(bricks[ci].key, raised);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

fn validate_inputs(
    topology: &CompiledTopology<2>,
    fields: &Fields,
    dt: f64,
    cell_size: f64,
) -> Result<(), ResolutionError> {
    let n = topology.graph.cells.len();
    if fields.density.len() != n
        || fields.capacity.len() != n
        || fields.cell_velocity.len() != 2 * n
    {
        return Err(ResolutionError::FieldShape);
    }
    if !(dt > 0.0 && cell_size > 0.0) {
        return Err(ResolutionError::InvalidStep);
    }
    Ok(())
}

fn validate_direct_surface(
    topology: &CompiledTopology<2>,
    surface: Option<&RdfSurface>,
) -> Result<(), ResolutionError> {
    let Some(surface) = surface else {
        return Ok(());
    };
    let dimensions = [
        topology.graph.dimensions[0] as u32,
        topology.graph.dimensions[1] as u32,
    ];
    let expected = (dimensions[0] as usize + 1).saturating_mul(dimensions[1] as usize + 1);
    if surface.dimensions != dimensions
        || surface.vertex_phi_fine.len() != expected
        || surface.vertex_phi_fine.iter().any(|value| !value.is_finite())
    {
        return Err(ResolutionError::FieldShape);
    }
    Ok(())
}

fn validate_free(bricks: &[BrickSeed], free: &[u32]) -> Result<BTreeSet<u32>, ResolutionError> {
    let mut set = BTreeSet::new();
    for &id in free {
        if !set.insert(id) {
            return Err(ResolutionError::InvalidOrDuplicateFreeLeaf(id));
        }
        if bricks.iter().any(|b| b.id == id && b.active) {
            return Err(ResolutionError::ActiveFreeLeaf(id));
        }
    }
    Ok(set)
}

pub fn plan_projected_transport_support(
    topology: &CompiledTopology<2>,
    fields: &Fields,
    dt: f64,
    cell_size: f64,
    policy: &ActivityPolicy,
    maximum_leaves: Option<usize>,
    maximum_cells: Option<usize>,
    free_leaf_ids: &[u32],
    include_interface_support: bool,
) -> Result<ProjectedTransportSupportDecision, ResolutionError> {
    plan_projected_transport_support_impl(
        topology,
        fields,
        dt,
        cell_size,
        policy,
        maximum_leaves,
        maximum_cells,
        free_leaf_ids,
        include_interface_support,
        None,
        None,
    )
}

pub fn plan_projected_transport_support_with_options(
    topology: &CompiledTopology<2>,
    fields: &Fields,
    dt: f64,
    cell_size: f64,
    options: &ResolutionPolicyOptions,
    include_interface_support: bool,
) -> Result<ProjectedTransportSupportDecision, ResolutionError> {
    plan_projected_transport_support_impl(
        topology,
        fields,
        dt,
        cell_size,
        &options.policy,
        options.maximum_leaves,
        options.maximum_cells,
        &options.free_leaf_ids,
        include_interface_support,
        Some(options),
        None,
    )
}

pub fn plan_projected_transport_support_with_surface(
    topology: &CompiledTopology<2>,
    fields: &Fields,
    dt: f64,
    cell_size: f64,
    options: &ResolutionPolicyOptions,
    include_interface_support: bool,
    surface: &RdfSurface,
) -> Result<ProjectedTransportSupportDecision, ResolutionError> {
    plan_projected_transport_support_impl(
        topology,
        fields,
        dt,
        cell_size,
        &options.policy,
        options.maximum_leaves,
        options.maximum_cells,
        &options.free_leaf_ids,
        include_interface_support,
        Some(options),
        Some(surface),
    )
}

fn plan_projected_transport_support_impl(
    topology: &CompiledTopology<2>,
    fields: &Fields,
    dt: f64,
    cell_size: f64,
    policy: &ActivityPolicy,
    maximum_leaves: Option<usize>,
    maximum_cells: Option<usize>,
    free_leaf_ids: &[u32],
    include_interface_support: bool,
    allocation_options: Option<&ResolutionPolicyOptions>,
    direct_surface: Option<&RdfSurface>,
) -> Result<ProjectedTransportSupportDecision, ResolutionError> {
    validate_inputs(topology, fields, dt, cell_size)?;
    validate_direct_surface(topology, direct_surface)?;
    let ts = thresholds(policy, dt, cell_size);
    let mut measurements = BTreeMap::new();
    for i in 0..topology.bricks.len() {
        measurements.insert(
            topology.bricks[i].seed.key,
            measure(
                topology,
                fields,
                i,
                None,
                policy,
                false,
                dt,
                ts,
                include_interface_support,
                direct_surface,
            ),
        );
    }
    let accepted: Vec<_> = topology.bricks.iter().map(|r| r.seed.clone()).collect();
    let free_set = validate_free(&accepted, free_leaf_ids)?;
    let mut working: Vec<_> = accepted
        .iter()
        .filter(|b| !free_set.contains(&b.id))
        .cloned()
        .collect();
    let mut demanded = BTreeSet::new();
    let mut required = BTreeMap::new();
    let mut claimed = Vec::new();
    // Free IDs are removed from `working` before reuse. Keep the monotonic
    // fallback above them too, or exhausting the free stack can claim its
    // highest ID a second time in the same candidate.
    let mut next_id = accepted
        .iter()
        .map(|b| b.id)
        .chain(free_leaf_ids.iter().copied())
        .max()
        .map_or(0, |v| v + 1);
    let mut free_stack = free_leaf_ids.to_vec();
    let mut next_key = working.iter().map(|b| b.key).max().map_or(0, |v| v + 1);
    let mut sources = accepted.clone();
    sources.sort_by_key(|b| b.key);
    for source in sources.iter().filter(|b| b.active) {
        let history = &measurements[&source.key].history;
        let mask = history.swept_support_mask
            | if include_interface_support {
                history.support_mask
            } else {
                0
            };
        for bit in 0..9_i32 {
            if bit == 4 || mask & (1 << bit) == 0 {
                continue;
            }
            let dx = bit % 3 - 1;
            let dy = bit / 3 - 1;
            let s = span(source);
            let qx = source.coordinate[0]
                + if dx < 0 {
                    -1
                } else if dx > 0 {
                    s
                } else {
                    0
                };
            let qy = source.coordinate[1]
                + if dy < 0 {
                    -1
                } else if dy > 0 {
                    s
                } else {
                    0
                };
            let (fx, fy) = ((qx * B) as f64 + 0.5, (qy * B) as f64 + 0.5);
            if fx < 0.0
                || fy < 0.0
                || fx >= topology.graph.dimensions[0] as f64
                || fy >= topology.graph.dimensions[1] as f64
            {
                continue;
            }
            let ri = if let Some(i) = owner_at(&working, fx, fy) {
                i
            } else {
                let id = free_stack.pop().unwrap_or_else(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
                claimed.push(id);
                working.push(BrickSeed {
                    id,
                    key: next_key,
                    coordinate: [qx, qy, 0],
                    span_bricks: 1,
                    resolution: 1,
                    active: true,
                    density: vec![],
                    gamma: vec![],
                    refinement_region_scale: None,
                });
                next_key += 1;
                working.len() - 1
            };
            if working[ri].key == source.key {
                continue;
            }
            demanded.insert(working[ri].key);
            let source_rung = allocation_options
                .filter(|options| options.coarsest_demanded_pages)
                .map(|_| donor_transport_rung(&measurements[&source.key], policy, ts))
                .filter(|&rung| rung > 1);
            record_donor_compatible_rung(&mut required, &working[ri], source, source_rung);
        }
    }
    let mut targets: BTreeMap<_, _> = working.iter().map(|b| {
        let target = if let Some(options) = allocation_options
            .filter(|options| options.coarsest_demanded_pages && options.coarsen_inactive_pages)
        {
            if !b.active
                && !options.policy.freeze_topology
                && !options.frozen_brick_keys.contains(&b.key)
            {
                apply_regions(1, b, &options.refinement_regions)
            } else {
                b.resolution
            }
        } else {
            b.resolution
        };
        (b.key, target)
    }).collect();
    let mut active: BTreeMap<_, _> = working.iter().map(|b| (b.key, b.active)).collect();
    for b in &working {
        if demanded.contains(&b.key) {
            let donor_rung = *required.get(&b.key).unwrap_or(&1);
            let target = if let Some(options) = allocation_options
                .filter(|options| options.coarsest_demanded_pages)
            {
                constrained_demanded_rung(b, donor_rung, options, b.active)
            } else {
                b.resolution.max(donor_rung)
            };
            targets.insert(b.key, target);
            active.insert(b.key, true);
        }
    }
    if let Some(options) = allocation_options.filter(|options| options.coarsest_demanded_pages) {
        close_region_caps(&working, &mut targets, &options.refinement_regions);
        close_two_to_one(&working, &mut targets, &options.refinement_regions);
    } else {
        close_two_to_one(&working, &mut targets, &[]);
    }
    let mut candidate = working.clone();
    for b in &mut candidate {
        b.resolution = targets[&b.key];
        b.active = active[&b.key];
    }
    let leaves = candidate.iter().filter(|b| b.active).count();
    let cells: usize = candidate
        .iter()
        .filter(|b| b.active)
        .map(|b| (b.resolution as usize).pow(2))
        .sum();
    let mut faults = 0;
    if maximum_leaves.is_some_and(|v| leaves > v) {
        faults |= resolution_fault::LEAF_CAPACITY;
    }
    if maximum_cells.is_some_and(|v| cells > v) {
        faults |= resolution_fault::CELL_CAPACITY;
    }
    Ok(ProjectedTransportSupportDecision {
        candidate_bricks: if faults == 0 { candidate } else { accepted },
        demanded_brick_keys: demanded,
        claimed_leaf_ids: claimed,
        fault_bits: faults,
    })
}

pub fn plan_resolution(
    topology: &CompiledTopology<2>,
    fields: &Fields,
    previous: &ResolutionPolicyState,
    dt: f64,
    cell_size: f64,
    options: &ResolutionPolicyOptions,
) -> Result<ResolutionPolicyDecision, ResolutionError> {
    plan_resolution_impl(topology, fields, previous, dt, cell_size, options, None)
}

pub fn plan_resolution_with_surface(
    topology: &CompiledTopology<2>,
    fields: &Fields,
    previous: &ResolutionPolicyState,
    dt: f64,
    cell_size: f64,
    options: &ResolutionPolicyOptions,
    surface: &RdfSurface,
) -> Result<ResolutionPolicyDecision, ResolutionError> {
    plan_resolution_impl(
        topology,
        fields,
        previous,
        dt,
        cell_size,
        options,
        Some(surface),
    )
}

fn plan_resolution_impl(
    topology: &CompiledTopology<2>,
    fields: &Fields,
    previous: &ResolutionPolicyState,
    dt: f64,
    cell_size: f64,
    options: &ResolutionPolicyOptions,
    direct_surface: Option<&RdfSurface>,
) -> Result<ResolutionPolicyDecision, ResolutionError> {
    validate_inputs(topology, fields, dt, cell_size)?;
    validate_direct_surface(topology, direct_surface)?;
    let policy = &options.policy;
    let accepted_steps = previous.accepted_steps + 1;
    let topology_epoch =
        policy.topology_cadence_steps != 0 && accepted_steps % policy.topology_cadence_steps == 0;
    let ts = thresholds(policy, dt, cell_size);
    let accepted: Vec<_> = topology.bricks.iter().map(|r| r.seed.clone()).collect();
    for b in &accepted {
        valid_resolution(b.resolution)?;
    }
    let mut measurements = BTreeMap::new();
    for i in 0..topology.bricks.len() {
        measurements.insert(
            accepted[i].key,
            measure(
                topology,
                fields,
                i,
                previous.history.get(&accepted[i].key),
                policy,
                topology_epoch,
                dt,
                ts,
                options.translation_invariant_motion_sizing,
                direct_surface,
            ),
        );
    }
    let mut material_demand = directional_demand(
        &accepted,
        &measurements,
        options.translation_invariant_motion_sizing,
    );
    material_demand.extend(options.injection_demanded_brick_keys.iter().copied());
    let free_set = validate_free(&accepted, &options.free_leaf_ids)?;
    let mut working: Vec<_> = accepted
        .iter()
        .filter(|b| !free_set.contains(&b.id))
        .cloned()
        .collect();
    let mut allocated = BTreeSet::new();
    let mut claimed = Vec::new();
    let mut allocated_resolution = BTreeMap::new();
    let mut demanded_resolution = BTreeMap::new();
    if options.allocate_missing_pages {
        // The fallback range must not overlap IDs claimed from the free stack.
        let mut next_id = accepted
            .iter()
            .map(|b| b.id)
            .chain(options.free_leaf_ids.iter().copied())
            .max()
            .map_or(0, |v| v + 1);
        let mut next_key = working.iter().map(|b| b.key).max().map_or(0, |v| v + 1);
        let mut free_stack = options.free_leaf_ids.clone();
        let mut sources = accepted.clone();
        sources.sort_by_key(|b| b.key);
        for source in sources.iter().filter(|b| b.active) {
            let history = &measurements[&source.key].history;
            let mask = history.swept_support_mask
                | if options.translation_invariant_motion_sizing {
                    history.support_mask
                } else {
                    0
                };
            for bit in 0..9_i32 {
                if bit == 4 || mask & (1 << bit) == 0 {
                    continue;
                }
                let dx = bit % 3 - 1;
                let dy = bit / 3 - 1;
                let s = span(source);
                let qx = source.coordinate[0]
                    + if dx < 0 {
                        -1
                    } else if dx > 0 {
                        s
                    } else {
                        0
                    };
                let qy = source.coordinate[1]
                    + if dy < 0 {
                        -1
                    } else if dy > 0 {
                        s
                    } else {
                        0
                    };
                let (fx, fy) = ((qx * B) as f64 + 0.5, (qy * B) as f64 + 0.5);
                if fx < 0.0
                    || fy < 0.0
                    || fx >= topology.graph.dimensions[0] as f64
                    || fy >= topology.graph.dimensions[1] as f64
                {
                    continue;
                }
                let ri = if let Some(i) = owner_at(&working, fx, fy) {
                    i
                } else {
                    let id = free_stack.pop().unwrap_or_else(|| {
                        let id = next_id;
                        next_id += 1;
                        id
                    });
                    claimed.push(id);
                    let key = next_key;
                    next_key += 1;
                    working.push(BrickSeed {
                        id,
                        key,
                        coordinate: [qx, qy, 0],
                        span_bricks: 1,
                        resolution: if options.coarsest_demanded_pages { 1 } else { 8 },
                        active: true,
                        density: vec![],
                        gamma: vec![],
                        refinement_region_scale: None,
                    });
                    allocated.insert(key);
                    allocated_resolution.insert(
                        key,
                        if options.coarsest_demanded_pages { 1 } else { 8 },
                    );
                    measurements.insert(
                        key,
                        Measurement {
                            history: BrickActivityHistory {
                                reasons: 0,
                                last_transition_step: accepted_steps,
                                ..BrickActivityHistory::default()
                            },
                            surface: false,
                            thin: false,
                            occupied: false,
                            detail: 0.0,
                            deeply_enclosed: false,
                            curvature_floor: 1,
                        },
                    );
                    working.len() - 1
                };
                material_demand.insert(working[ri].key);
                let source_rung = options
                    .coarsest_demanded_pages
                    .then(|| donor_transport_rung(&measurements[&source.key], policy, ts))
                    .filter(|&rung| rung > 1);
                record_donor_compatible_rung(
                    &mut demanded_resolution,
                    &working[ri],
                    source,
                    source_rung,
                );
            }
        }
    }
    let mut targets = BTreeMap::new();
    let mut candidate_active = BTreeMap::new();
    let mut plan_reasons = BTreeMap::new();
    for (bi, brick) in accepted.iter().enumerate() {
        let current = brick.resolution;
        let m = &measurements[&brick.key];
        let mut requested = current;
        let mut reason = 32_u32;
        let frozen = policy.freeze_topology || options.frozen_brick_keys.contains(&brick.key);
        if frozen {
            targets.insert(brick.key, current);
            candidate_active.insert(brick.key, brick.active);
            plan_reasons.insert(brick.key, 32);
            continue;
        }
        if !brick.active {
            // Inactive pages have no represented cells, so retaining their old
            // rung can only impose a stale 2:1 floor on nearby active pages.
            // Region constraints and the existing closure still grade their
            // metadata, and material demand overrides reactivation to B8.
            targets.insert(
                brick.key,
                if options.coarsen_inactive_pages {
                    apply_regions(1, brick, &options.refinement_regions)
                } else {
                    current
                },
            );
            candidate_active.insert(brick.key, false);
            plan_reasons.insert(brick.key, 128);
            continue;
        }
        let measured_floor = velocity_floor(m.history.velocity_travel, ts, policy.activity_signals);
        let static_floor = *options
            .static_boundary_floor_by_brick
            .get(&brick.key)
            .unwrap_or(&1);
        let moving_floor = if options.moving_rigid_bodies
            && m.history.reasons & activity_reason::CUT_BOUNDARY != 0
        {
            4
        } else {
            1
        };
        let enclosed = policy.activity_signals && m.deeply_enclosed && measured_floor == 1;
        let surface = m.surface && !enclosed;
        let slow_surface = surface && !m.thin && measured_floor == 1;
        let next = (current / 2).max(1);
        let touches_liquid = face_neighbors(&accepted, bi)
            .into_iter()
            .any(|j| accepted[j].active && measurements[&accepted[j].key].occupied);
        let injection = options.injection_demanded_brick_keys.contains(&brick.key);
        let page_demand = injection
            || (touches_liquid
                && (!m.occupied
                    || (m.history.mean_density as f64) < policy.surface_density_minimum));
        let required = if enclosed {
            static_floor.max(moving_floor)
        } else {
            measured_floor
                .max(if surface { next } else { 1 })
                .max(if m.thin || page_demand { 8 } else { 1 })
                .max(static_floor)
                .max(moving_floor)
        };
        let emergency = js_round(255.0 * policy.emergency_score) as u8;
        if required > current
            || (!surface && !enclosed && !slow_surface && m.history.score_byte >= emergency)
        {
            let urgent = m.thin || page_demand;
            requested = if urgent {
                required
            } else {
                required.max(2 * current).min(8)
            };
            reason = if page_demand || m.history.reasons & activity_reason::PREDICTED_FACE != 0 {
                2
            } else if m.thin {
                256
            } else if measured_floor > current {
                64
            } else {
                4
            };
        } else if topology_epoch {
            if surface && current > 1 {
                let proof = m
                    .history
                    .surface_proof
                    .as_ref()
                    .and_then(|p| p.generation_by_target_resolution.get(&next))
                    .copied();
                let fresh = policy.surface_coarsening_enabled
                    && proof == Some(topology.graph.topology_generation);
                let threshold = 0.5
                    * (ts[(current.trailing_zeros()) as usize] as f64
                        + ts[(next.trailing_zeros()) as usize] as f64);
                let epochs = if fresh
                    && required <= next
                    && (m.history.velocity_travel as f64) < threshold
                {
                    m.history.proof_epochs.saturating_add(1)
                } else {
                    0
                };
                measurements
                    .get_mut(&brick.key)
                    .unwrap()
                    .history
                    .proof_epochs = epochs;
                if epochs >= policy.demote_epochs.max(8) {
                    requested = next;
                    reason = 16;
                }
            } else if !enclosed && !slow_surface && m.history.hot_epochs >= policy.promote_epochs {
                requested = (2 * current).min(8);
                reason = 8;
            } else if current > required
                && (enclosed || slow_surface || m.history.quiet_epochs >= policy.demote_epochs)
                && m.history.reasons & activity_reason::FINE_DETAIL == 0
            {
                requested = if enclosed {
                    required
                } else {
                    required.max(current / 2)
                };
                reason = if enclosed { 2048 } else { 16 };
            }
        }
        if policy.coarse_first && policy.forced_surface_resolution_for_qa.is_none() {
            let m = &measurements[&brick.key];
            let geometry_floor = m.curvature_floor.max(static_floor).max(moving_floor);
            let incoming = if geometry_floor.max(measured_floor) < 8
                && !m.thin
                && !injection
                && (surface || page_demand)
            {
                incoming_floor(topology, bi, &measurements, policy)
            } else {
                1
            };
            let incoming_retention = if direct_surface.is_some() { 1 } else { current.min(incoming) };
            let frontier = [3, 5, 1, 7].into_iter().any(|bit| {
                if (m.history.support_mask | m.history.swept_support_mask) & (1 << bit) == 0 {
                    return false;
                }
                let dx = bit % 3 - 1;
                let dy = bit / 3 - 1;
                owner_at(
                    &accepted,
                    ((brick.coordinate[0] + dx) * B) as f64 + 0.5,
                    ((brick.coordinate[1] + dy) * B) as f64 + 0.5,
                )
                .is_none_or(|j| !accepted[j].active)
            });
            let moving_frontier = direct_surface.is_none() && (frontier || page_demand) && measured_floor > 1;
            let safety = if m.thin || injection || moving_frontier {
                8
            } else {
                1
            };
            let material_floor = if direct_surface.is_some() && m.history.mean_density > 0.0 { 2 } else { 1 };
            let coarse_required = geometry_floor.max(material_floor)
                .max(measured_floor)
                .max(incoming_retention)
                .max(safety);
            requested = current;
            reason = 32;
            if coarse_required > current {
                requested = coarse_required;
                reason = 4;
                measurements
                    .get_mut(&brick.key)
                    .unwrap()
                    .history
                    .proof_epochs = 0;
            } else if coarse_required < current && enclosed {
                requested = coarse_required;
                reason = 2048;
            } else if coarse_required < current {
                let next = (current / 2).max(1);
                let m = &measurements[&brick.key];
                let fresh = !surface
                    || (policy.surface_coarsening_enabled && m.history
                        .surface_proof
                        .as_ref()
                        .and_then(|p| p.generation_by_target_resolution.get(&next))
                        .copied()
                        == Some(topology.graph.topology_generation));
                // The legacy branch above may have touched the measurement's
                // epoch counter. GPU coarse-first reloads the accepted history
                // here; count at most one proof epoch per accepted frame.
                let mut epochs = if fresh {
                    previous.history.get(&brick.key).map_or(0, |h| h.proof_epochs)
                } else { 0 };
                if fresh && topology_epoch {
                    epochs = epochs.saturating_add(1);
                }
                measurements
                    .get_mut(&brick.key)
                    .unwrap()
                    .history
                    .proof_epochs = epochs;
                if epochs >= policy.surface_quiet_epochs {
                    requested = coarse_required.max(current / 2);
                    reason = 16;
                }
            } else {
                measurements
                    .get_mut(&brick.key)
                    .unwrap()
                    .history
                    .proof_epochs = 0;
            }
        }
        requested = apply_regions(requested, brick, &options.refinement_regions);
        targets.insert(brick.key, requested);
        candidate_active.insert(brick.key, true);
        plan_reasons.insert(brick.key, reason);
    }
    for brick in working.iter().filter(|b| allocated.contains(&b.key)) {
        let target = if options.coarsest_demanded_pages {
            constrained_demanded_rung(
                brick,
                *demanded_resolution.get(&brick.key).unwrap_or(&1),
                options,
                false,
            )
        } else {
            allocated_resolution[&brick.key]
        };
        targets.insert(brick.key, target);
        candidate_active.insert(brick.key, true);
        plan_reasons.insert(brick.key, 0x8000_0001);
    }
    for brick in accepted.iter().filter(|b| !b.active) {
        if material_demand.contains(&brick.key) {
            candidate_active.insert(brick.key, true);
            let target = if options.injection_demanded_brick_keys.contains(&brick.key) {
                apply_regions(8, brick, &options.refinement_regions)
            } else if options.coarsest_demanded_pages {
                constrained_demanded_rung(
                    brick,
                    *demanded_resolution.get(&brick.key).unwrap_or(&1),
                    options,
                    false,
                )
            } else {
                apply_regions(8, brick, &options.refinement_regions)
            };
            targets.insert(
                brick.key,
                target,
            );
            plan_reasons.insert(brick.key, 0x8000_0001);
        }
    }
    for (bi, brick) in accepted.iter().enumerate().filter(|(_, b)| b.active) {
        let m = &measurements[&brick.key];
        let range = topology.bricks[bi].cell_range.clone();
        let exact_empty = range
            .into_iter()
            .all(|id| fields.density[id as usize] == 0.0);
        let moving =
            options.moving_rigid_bodies && m.history.reasons & activity_reason::CUT_BOUNDARY != 0;
        if exact_empty
            && !m.occupied
            && !material_demand.contains(&brick.key)
            && !moving
            && !options.injection_demanded_brick_keys.contains(&brick.key)
        {
            candidate_active.insert(brick.key, false);
            plan_reasons.insert(brick.key, 0x8000_0000);
        }
    }
    close_region_caps(&working, &mut targets, &options.refinement_regions);
    close_two_to_one(&working, &mut targets, &options.refinement_regions);
    let mut faults = 0;
    for i in 0..working.len() {
        for j in face_neighbors(&working, i) {
            if !candidate_active[&working[i].key] || !candidate_active[&working[j].key] {
                continue;
            }
            let (a, b) = (
                width(&working[i], targets[&working[i].key]),
                width(&working[j], targets[&working[j].key]),
            );
            if a.max(b) > 2.0 * a.min(b) {
                faults |= resolution_fault::TWO_TO_ONE;
            }
        }
    }
    let changed: Vec<_> = (0..working.len())
        .filter(|&i| {
            allocated.contains(&working[i].key)
                || targets[&working[i].key] != working[i].resolution
                || candidate_active[&working[i].key] != working[i].active
        })
        .collect();
    let urgent: BTreeSet<_> = changed
        .iter()
        .copied()
        .filter(|&i| {
            targets[&working[i].key] > working[i].resolution
                || candidate_active[&working[i].key] != working[i].active
                || region_bounds(&working[i], &options.refinement_regions).0 < working[i].resolution
        })
        .collect();
    let ordinary: BTreeSet<_> = changed
        .iter()
        .copied()
        .filter(|i| !urgent.contains(i) && targets[&working[*i].key] < working[*i].resolution)
        .collect();
    let mut ordered: Vec<_> = (0..working.len()).collect();
    ordered.sort_by_key(|&i| working[i].key);
    let rotate = if ordered.is_empty() {
        0
    } else {
        previous.scheduling_cursor % ordered.len()
    };
    ordered.rotate_left(rotate);
    let budget = policy.prepare_bricks_per_frame;
    let credit_budget = (budget + 1).min(previous.scheduling_credits + budget);
    let mut admitted = BTreeSet::new();
    let mut considered = BTreeSet::new();
    for &i in &ordered {
        if !ordinary.contains(&i) || considered.contains(&working[i].key) {
            continue;
        }
        let (lo, hi) = bounds(&working[i]);
        let reflected_lo = topology.graph.dimensions[0] as i32 - hi[0];
        let mirror = (0..working.len()).find(|&j| {
            let (ml, mh) = bounds(&working[j]);
            ml[0] == reflected_lo
                && ml[1] == lo[1]
                && mh[0] - ml[0] == hi[0] - lo[0]
                && mh[1] == hi[1]
        });
        let paired = mirror.filter(|&j| {
            working[j].key != working[i].key
                && ordinary.contains(&j)
                && working[j].resolution == working[i].resolution
                && targets[&working[j].key] == targets[&working[i].key]
        });
        let orbit = if let Some(j) = paired {
            vec![i, j]
        } else {
            vec![i]
        };
        for &j in &orbit {
            considered.insert(working[j].key);
        }
        if admitted.len() + orbit.len() > credit_budget {
            continue;
        }
        for j in orbit {
            admitted.insert(j);
        }
    }
    for &i in &ordinary {
        if !admitted.contains(&i) {
            targets.insert(working[i].key, working[i].resolution);
        }
    }
    let mut candidate = working.clone();
    for b in &mut candidate {
        b.resolution = targets[&b.key];
        b.active = candidate_active[&b.key];
    }
    let leaves = candidate.iter().filter(|b| b.active).count();
    let cells: usize = candidate
        .iter()
        .filter(|b| b.active)
        .map(|b| (b.resolution as usize).pow(2))
        .sum();
    if options.maximum_leaves.is_some_and(|v| leaves > v) {
        faults |= resolution_fault::LEAF_CAPACITY;
    }
    if options.maximum_cells.is_some_and(|v| cells > v) {
        faults |= resolution_fault::CELL_CAPACITY;
    }
    if faults != 0 {
        candidate = accepted.clone();
    }
    let published = faults == 0
        && (candidate.len() != accepted.len()
            || candidate.iter().any(|b| {
                accepted
                    .iter()
                    .find(|a| a.key == b.key)
                    .is_none_or(|a| a.resolution != b.resolution || a.active != b.active)
            }));
    let mut next_history = BTreeMap::new();
    let mut records = Vec::new();
    let (mut promoted, mut demoted, mut activated, mut retired, mut max_score) = (0, 0, 0, 0, 0);
    for brick in &working {
        let mut h = measurements[&brick.key].history.clone();
        let Some(scheduled) = candidate.iter().find(|b| b.key == brick.key) else {
            continue;
        };
        let was_accepted = !allocated.contains(&brick.key) && brick.active;
        let did_change = allocated.contains(&brick.key)
            || scheduled.resolution != brick.resolution
            || scheduled.active != brick.active;
        if did_change {
            h.hot_epochs = 0;
            h.quiet_epochs = 0;
            h.proof_epochs = 0;
            h.last_transition_step = accepted_steps;
        }
        max_score = max_score.max(h.score_byte);
        if scheduled.resolution > brick.resolution {
            promoted += 1;
        }
        if scheduled.resolution < brick.resolution {
            demoted += 1;
        }
        if !was_accepted && scheduled.active {
            activated += 1;
        }
        if was_accepted && !scheduled.active {
            retired += 1;
        }
        records.push(BrickResolutionReceipt {
            brick_key: brick.key,
            accepted_resolution: brick.resolution,
            requested_resolution: targets[&brick.key],
            scheduled_resolution: scheduled.resolution,
            accepted_active: was_accepted,
            candidate_active: scheduled.active,
            score_byte: measurements[&brick.key].history.score_byte,
            reasons: measurements[&brick.key].history.reasons,
            plan_reasons: plan_reasons[&brick.key],
            support_mask: measurements[&brick.key].history.support_mask,
            swept_support_mask: measurements[&brick.key].history.swept_support_mask,
            fault_bits: faults,
        });
        next_history.insert(brick.key, h);
    }
    let ordinary_len = ordinary.len();
    let ordered_len = ordered.len();
    Ok(ResolutionPolicyDecision {
        candidate_bricks: candidate,
        state: ResolutionPolicyState {
            accepted_steps,
            accepted_generation: topology.graph.topology_generation,
            scheduling_cursor: if ordered_len > 0 {
                (previous.scheduling_cursor + budget.min(ordinary_len)) % ordered_len
            } else {
                0
            },
            scheduling_credits: credit_budget - admitted.len(),
            history: next_history,
        },
        receipt: ResolutionPolicyReceipt {
            topology_epoch,
            accepted_generation: topology.graph.topology_generation,
            candidate_generation: topology.graph.topology_generation + u32::from(published),
            measured_brick_count: accepted.len(),
            surface_brick_count: measurements.values().filter(|m| m.surface).count(),
            occupied_brick_count: measurements.values().filter(|m| m.occupied).count(),
            activated_brick_count: activated,
            allocated_brick_count: allocated.len(),
            claimed_leaf_ids: claimed,
            retired_brick_count: retired,
            promoted_brick_count: promoted,
            demoted_brick_count: demoted,
            deferred_demotion_count: ordinary_len - admitted.len(),
            maximum_score_byte: max_score,
            fault_bits: faults,
            bricks: records,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::BoundaryMode;
    use crate::topology::{compile_topology, TopologySeed};

    fn brick(id: u32, coordinate: [i32; 2], resolution: u8, active: bool) -> BrickSeed {
        BrickSeed {
            id,
            key: id,
            coordinate: [coordinate[0], coordinate[1], 0],
            span_bricks: 1,
            resolution,
            active,
            density: vec![],
            gamma: vec![],
            refinement_region_scale: None,
        }
    }
    fn setup(bricks: Vec<BrickSeed>, dimensions: [u32; 2]) -> (CompiledTopology<2>, Fields) {
        let topology = compile_topology(TopologySeed {
            dimensions: [dimensions[0], dimensions[1], 1],
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Closed; 6],
            bricks,
        })
        .unwrap();
        let n = topology.graph.cells.len();
        let fields = Fields {
            density: vec![0.0; n],
            capacity: vec![1.0; n],
            cell_velocity: vec![0.0; 2 * n],
            face_velocity: vec![],
            acceleration_fine: [0.0; 3],
            interface_normal: vec![0.0; 2 * n],
            ..Fields::default()
        };
        (topology, fields)
    }
    fn options() -> ResolutionPolicyOptions {
        ResolutionPolicyOptions::default()
    }
    #[test]
    fn dry_receiver_refines_for_closing_streams_not_translation_or_separation() {
        let (topology, mut fields) = setup(vec![brick(0,[0,0],8,true),
            brick(1,[1,0],4,true),brick(2,[2,0],8,true)], [24,8]);
        let vertices = (0..=8).flat_map(|_| (0..=24).map(|x|
            (x as f32-8.0).min(16.0-x as f32))).collect();
        let surface = levelset_surface::publish([24,8],vertices,0.0).unwrap();
        let options = coarsest_support_options();
        let run = |fields: &mut Fields, left: f32, right: f32| {
            for cell in &topology.graph.cells {
                let id=cell.id as usize;
                let x=cell.center[0];
                fields.density[id]=if x<8.0||x>16.0 {1.0} else {0.0};
                fields.cell_velocity[2*id]=if x<8.0 {left} else if x>16.0 {right} else {0.0};
            }
            let result=plan_resolution_with_surface(&topology, fields,
                &initialize_resolution_policy(&topology), 1.0/30.0, 0.05,
                &options,&surface).unwrap();
            (result.state.history[&1].velocity_travel,
                result.receipt.bricks.iter().find(|b| b.brick_key==1).unwrap().requested_resolution)
        };
        let collision=run(&mut fields,80.0,-80.0);
        let translated_collision=run(&mut fields,200.0,40.0);
        let translation=run(&mut fields,20.0,20.0);
        let separating=run(&mut fields,-20.0,20.0);
        assert!(collision.0>1.0&&collision.1==8, "collision={collision:?}");
        assert_eq!(collision,translated_collision);
        assert_eq!(translation.0,0.0);
        assert_eq!(separating.0,0.0);
    }

    fn coarsest_support_options() -> ResolutionPolicyOptions {
        let mut options = options();
        options.translation_invariant_motion_sizing = true;
        options.coarsen_inactive_pages = true;
        options.coarsest_demanded_pages = true;
        options
    }

    fn circle_surface(dimensions: [u32; 2], centre: [f32; 2], radius: f32) -> RdfSurface {
        let vertices = (0..=dimensions[1])
            .flat_map(|y| {
                (0..=dimensions[0]).map(move |x| {
                    (x as f32 - centre[0]).hypot(y as f32 - centre[1]) - radius
                })
            })
            .collect();
        levelset_surface::publish(dimensions, vertices, 0.0).unwrap()
    }

    fn plane_surface(dimensions: [u32; 2], x_intercept: f32) -> RdfSurface {
        let vertices = (0..=dimensions[1])
            .flat_map(|_| (0..=dimensions[0]).map(move |x| x as f32 - x_intercept))
            .collect();
        levelset_surface::publish(dimensions, vertices, 0.0).unwrap()
    }

    #[test]
    fn published_direct_phi_proof_allows_demotion_without_moving_contour() {
        let (topology, mut fields) = setup(vec![brick(0, [0, 0], 8, true)], [8, 8]);
        let surface = plane_surface([8, 8], 4.25);
        fields.density = levelset_surface::implied_fill_fine_cells(&surface).unwrap();
        let original = surface.vertex_phi_fine.clone();
        let options = coarsest_support_options();
        let mut state = initialize_resolution_policy(&topology);
        state.history.get_mut(&0).unwrap().reasons |= activity_reason::SURFACE;
        for frame in 1..=2 {
            publish_direct_surface_proofs(&topology, &fields, &surface, &options,
                &mut state, 1.0 / 30.0, 0.05).unwrap();
            assert_eq!(state.history[&0].surface_proof.as_ref().unwrap()
                .generation_by_target_resolution[&4], 1);
            let decision = plan_resolution_with_surface(&topology, &fields, &state,
                1.0 / 30.0, 0.05, &options, &surface).unwrap();
            state = decision.state;
            assert_eq!(decision.receipt.bricks[0].scheduled_resolution,
                if frame == 1 { 8 } else { 4 });
            // A committed topology change retires the consumed epoch count.
            assert_eq!(state.history[&0].proof_epochs, if frame == 1 { 1 } else { 0 });
        }
        assert_eq!(surface.vertex_phi_fine, original);
        // A certificate belongs to one generation, never an arbitrary future one.
        let mut changed = topology.clone();
        changed.graph.topology_generation += 1;
        let decision = plan_resolution_with_surface(&changed, &fields, &state,
            1.0 / 30.0, 0.05, &options, &surface).unwrap();
        assert_eq!(decision.receipt.bricks[0].scheduled_resolution, 8);
        assert_eq!(decision.state.history[&0].proof_epochs, 0);
    }

    #[test]
    fn direct_phi_proof_bounds_added_error_and_preserves_thin_features() {
        let (topology, mut fields) = setup(vec![brick(0, [0, 0], 4, true)], [8, 8]);
        let surface = plane_surface([8, 8], 4.0);
        for cell in &topology.graph.cells {
            fields.density[cell.id as usize] = if cell.center[0] < 4.0 { 1.0 } else { 0.0 };
        }
        let mut options = coarsest_support_options();
        let mut state = initialize_resolution_policy(&topology);
        state.history.get_mut(&0).unwrap().reasons |= activity_reason::SURFACE;
        let publish = |fields: &Fields, surface: &RdfSurface, options: &ResolutionPolicyOptions,
                       state: &mut ResolutionPolicyState| {
            publish_direct_surface_proofs(&topology, fields, surface, options,
                state, 1.0 / 30.0, 0.05).unwrap();
        };
        publish(&fields, &surface, &options, &mut state);
        assert!(state.history[&0].surface_proof.is_some());
        // Existing volume disagreement must not permanently freeze topology.
        fields.density[1] = 3.0;
        state.history.get_mut(&0).unwrap().proof_epochs = 1;
        publish(&fields, &surface, &options, &mut state);
        assert!(state.history[&0].surface_proof.is_some());
        fields.density[1] = 1.0;
        // This droplet lies between candidate vertices and would disappear.
        let curved = circle_surface([8, 8], [2.0, 2.0], 1.0);
        publish(&fields, &curved, &options, &mut state);
        assert!(state.history[&0].surface_proof.is_none());
        publish(&fields, &surface, &options, &mut state);
        assert!(state.history[&0].surface_proof.is_some());
        state.history.get_mut(&0).unwrap().reasons |= activity_reason::THIN_FLUID;
        publish(&fields, &surface, &options, &mut state);
        assert!(state.history[&0].surface_proof.is_none());
        state.history.get_mut(&0).unwrap().reasons &= !activity_reason::THIN_FLUID;
        options.policy.surface_coarsening_enabled = false;
        publish(&fields, &surface, &options, &mut state);
        assert!(state.history[&0].surface_proof.is_none());
    }

    #[test]
    fn surface_publication_preserves_non_surface_coarsening_epochs() {
        let (topology, fields) = setup(vec![brick(0, [0, 0], 2, true)], [8, 8]);
        let surface = plane_surface([8, 8], -4.0);
        let mut state = initialize_resolution_policy(&topology);
        state.history.get_mut(&0).unwrap().proof_epochs = 1;
        let options = coarsest_support_options();
        publish_direct_surface_proofs(&topology, &fields, &surface, &options,
            &mut state, 1.0 / 30.0, 0.05).unwrap();
        assert_eq!(state.history[&0].proof_epochs, 1);
        assert!(state.history[&0].surface_proof.is_none());
        let plan = plan_resolution_with_surface(&topology, &fields, &state,
            1.0 / 30.0, 0.05, &options, &surface).unwrap();
        assert!(plan.receipt.bricks[0].scheduled_resolution <= 1
            || !plan.receipt.bricks[0].candidate_active);
    }

    #[test]
    fn direct_phi_uniform_translation_only_requires_graded_receiver_support() {
        let (topology, mut fields) = setup(vec![brick(0, [0, 0], 8, true)], [16, 8]);
        let surface = plane_surface([16, 8], 7.0);
        fields.density.fill(1.0);
        for velocity in fields.cell_velocity.chunks_exact_mut(2) {
            velocity[0] = 20.0;
        }
        let options = coarsest_support_options();
        let projected = plan_projected_transport_support_with_surface(
            &topology, &fields, 1.0, 1.0, &options, true, &surface).unwrap();
        let post = plan_resolution_with_surface(&topology, &fields,
            &initialize_resolution_policy(&topology), 1.0, 1.0, &options, &surface).unwrap();
        for bricks in [&projected.candidate_bricks, &post.candidate_bricks] {
            assert_eq!(bricks.iter().find(|b| b.coordinate == [1, 0, 0]).unwrap().resolution, 4);
        }
    }

    fn golden() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../testdata/slice-resolution-policy-golden.json"
        ))
        .unwrap()
    }

    #[test]
    fn ts_golden_swept_frontier_activation() {
        let (topology, mut fields) = setup(
            vec![brick(0, [0, 0], 8, true), brick(1, [1, 0], 8, false)],
            [16, 8],
        );
        for cell in &topology.graph.cells {
            if cell.brick_key == Some(0) && cell.minimum[0] == 7.0 {
                fields.density[cell.id as usize] = 1.0;
                fields.cell_velocity[2 * cell.id as usize] = 1.0;
            }
        }
        let decision = plan_resolution(
            &topology,
            &fields,
            &initialize_resolution_policy(&topology),
            1.0 / 60.0,
            0.05,
            &options(),
        )
        .unwrap();
        let record = decision
            .receipt
            .bricks
            .iter()
            .find(|r| r.brick_key == 1)
            .unwrap();
        assert_eq!(
            (
                record.accepted_active,
                record.candidate_active,
                record.scheduled_resolution
            ),
            (false, true, 8)
        );
        assert_eq!(decision.receipt.fault_bits, 0);
        assert_eq!(
            (
                decision.receipt.candidate_generation,
                decision.receipt.surface_brick_count,
                decision.receipt.occupied_brick_count,
                decision.receipt.activated_brick_count,
                decision.receipt.maximum_score_byte,
            ),
            (2, 1, 1, 1, 3)
        );
        let donor = decision
            .receipt
            .bricks
            .iter()
            .find(|r| r.brick_key == 0)
            .unwrap();
        assert_eq!(
            (
                donor.score_byte,
                donor.reasons,
                donor.plan_reasons,
                donor.support_mask,
                donor.swept_support_mask,
            ),
            (3, 82_249, 32, 422, 48)
        );
        assert_eq!(
            serde_json::json!([
                donor.brick_key,
                donor.accepted_resolution,
                donor.requested_resolution,
                donor.scheduled_resolution,
                donor.accepted_active,
                donor.candidate_active,
                donor.score_byte,
                donor.reasons,
                donor.plan_reasons,
                donor.support_mask,
                donor.swept_support_mask,
                donor.fault_bits
            ]),
            golden()["sweptFrontier"]["donor"]
        );
    }

    #[test]
    fn ts_golden_exact_zero_retirement() {
        for value in [0.0_f32, 1.0e-30_f32] {
            let (topology, mut fields) = setup(vec![brick(0, [0, 0], 8, true)], [8, 8]);
            fields.density[0] = value;
            let d = plan_resolution(
                &topology,
                &fields,
                &initialize_resolution_policy(&topology),
                1.0 / 60.0,
                0.05,
                &options(),
            )
            .unwrap();
            assert_eq!(d.receipt.bricks[0].candidate_active, value != 0.0);
        }
    }

    #[test]
    fn inactive_fine_metadata_does_not_pin_adjacent_residual_page() {
        let (topology, mut fields) = setup(
            vec![brick(0, [0, 0], 2, true), brick(1, [1, 0], 4, false)],
            [16, 8],
        );
        // A conservative tail keeps the active page resident, but is below
        // occupancy and thin-feature thresholds and should not keep it fine.
        fields.density[0] = 3.4e-8;
        let mut state = initialize_resolution_policy(&topology);
        state.history.get_mut(&0).unwrap().proof_epochs = 1;

        let mut policy = options();
        policy.coarsen_inactive_pages = true;
        let decision = plan_resolution(
            &topology,
            &fields,
            &state,
            1.0 / 30.0,
            0.05,
            &policy,
        )
        .unwrap();
        let active = decision.receipt.bricks.iter()
            .find(|record| record.brick_key == 0).unwrap();
        let inactive = decision.receipt.bricks.iter()
            .find(|record| record.brick_key == 1).unwrap();
        assert_eq!((active.requested_resolution, active.scheduled_resolution), (1, 1));
        assert!(active.candidate_active);
        assert_eq!((inactive.requested_resolution, inactive.scheduled_resolution), (1, 1));
        assert!(!inactive.candidate_active);

        let candidate = compile_topology::<2>(TopologySeed {
            dimensions: [16, 8, 1],
            generation: 2,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Closed; 6],
            bricks: decision.candidate_bricks,
        }).unwrap();
        let inactive = candidate.bricks.iter()
            .find(|page| page.seed.key == 1).unwrap();
        assert_eq!(inactive.cell_range.start, inactive.cell_range.end);
    }

    #[test]
    fn donor_compatible_rung_uses_physical_width_and_combines_donors() {
        let mut receiver = brick(10, [2, 0], 1, false);
        receiver.span_bricks = 2;
        let coarse = brick(1, [0, 0], 2, true);
        let fine = brick(2, [1, 0], 8, true);
        let mut required = BTreeMap::new();
        record_donor_compatible_rung(&mut required, &receiver, &coarse, None);
        assert_eq!(required[&receiver.key], 2);
        record_donor_compatible_rung(&mut required, &receiver, &fine, None);
        assert_eq!(required[&receiver.key], 8);

        let coarsest = brick(3, [0, 0], 1, true);
        let mut baseline = BTreeMap::new();
        record_donor_compatible_rung(&mut baseline, &receiver, &coarsest, None);
        assert_eq!(baseline[&receiver.key], 1);
    }

    #[test]
    fn direct_phi_curvature_measure_is_independent_of_adaptive_rung() {
        let surface = circle_surface([8, 8], [4.0, 4.0], 3.0);
        let policy = ActivityPolicy::default();
        let collect = |resolution| {
            let (topology, fields) = setup(vec![brick(0, [0, 0], resolution, true)], [8, 8]);
            let ts = thresholds(&policy, 1.0 / 30.0, 1.0);
            measure(
                &topology,
                &fields,
                0,
                None,
                &policy,
                false,
                1.0 / 30.0,
                ts,
                true,
                Some(&surface),
            )
        };
        let fine = collect(8);
        let coarse = collect(1);
        assert!(fine.surface && coarse.surface);
        assert_eq!(fine.curvature_floor, coarse.curvature_floor);
        assert_eq!(fine.curvature_floor, 1);
    }

    #[test]
    fn direct_phi_crossing_is_not_hidden_by_density_enclosure() {
        let bricks = (0..3)
            .flat_map(|y| (0..3).map(move |x| brick((x + 3 * y) as u32, [x, y], 1, true)))
            .collect();
        let (topology, mut fields) = setup(bricks, [24, 24]);
        fields.density.fill(1.0);
        let surface = circle_surface([24, 24], [12.0, 12.0], 2.0);
        let policy = ActivityPolicy::default();
        let measured = measure(
            &topology,
            &fields,
            4,
            None,
            &policy,
            false,
            1.0 / 30.0,
            thresholds(&policy, 1.0 / 30.0, 1.0),
            true,
            Some(&surface),
        );
        assert!(measured.surface);
        assert!(!measured.deeply_enclosed);
        assert_eq!(measured.curvature_floor, 1);
    }

    #[test]
    fn direct_phi_requirement_stops_receiver_rung_ratchet_in_both_planners() {
        // Isolate curvature below the absolute-speed transport floor. A fast
        // translated interface requires fine support in the GPU policy too.
        let (topology, mut fields) = setup(vec![brick(0, [0, 0], 8, true)], [16, 8]);
        for cell in &topology.graph.cells {
            let id = cell.id as usize;
            fields.density[id] = 1.0;
            fields.cell_velocity[2 * id] = 0.1;
        }
        let options = coarsest_support_options();
        let curved = circle_surface([16, 8], [7.0, 4.0], 3.0);
        let planar = plane_surface([16, 8], 7.0);

        let projected_curved = plan_projected_transport_support_with_surface(
            &topology, &fields, 1.0, 1.0, &options, true, &curved,
        ).unwrap();
        let projected_planar = plan_projected_transport_support_with_surface(
            &topology, &fields, 1.0, 1.0, &options, true, &planar,
        ).unwrap();
        let resolution_at = |decision: &ProjectedTransportSupportDecision| {
            decision.candidate_bricks.iter()
                .find(|brick| brick.coordinate == [1, 0, 0]).unwrap().resolution
        };
        assert_eq!(resolution_at(&projected_curved), resolution_at(&projected_planar));
        assert_eq!(resolution_at(&projected_planar), 4);

        let previous = initialize_resolution_policy(&topology);
        let post_curved = plan_resolution_with_surface(
            &topology, &fields, &previous, 1.0, 1.0, &options, &curved,
        ).unwrap();
        let post_planar = plan_resolution_with_surface(
            &topology, &fields, &previous, 1.0, 1.0, &options, &planar,
        ).unwrap();
        let post_resolution = |decision: &ResolutionPolicyDecision| {
            decision.candidate_bricks.iter()
                .find(|brick| brick.coordinate == [1, 0, 0]).unwrap().resolution
        };
        assert_eq!(post_resolution(&post_curved), post_resolution(&post_planar));
        assert_eq!(post_resolution(&post_planar), 4);
    }

    #[test]
    fn coarsest_support_sizes_new_page_in_both_planners() {
        let (topology, mut fields) = setup(vec![brick(0, [0, 0], 8, true)], [16, 8]);
        for cell in &topology.graph.cells {
            let id = cell.id as usize;
            fields.density[id] = 1.0;
            fields.cell_velocity[2 * id] = 20.0;
        }
        let options = coarsest_support_options();
        let projected = plan_projected_transport_support_with_options(
            &topology, &fields, 1.0, 1.0, &options, true,
        ).unwrap();
        let projected_receiver = projected.candidate_bricks.iter()
            .find(|brick| brick.coordinate == [1, 0, 0]).unwrap();
        assert!(projected_receiver.active);
        assert_eq!(projected_receiver.resolution, 4);

        let post = plan_resolution(
            &topology,
            &fields,
            &initialize_resolution_policy(&topology),
            1.0,
            1.0,
            &options,
        ).unwrap();
        let post_receiver = post.candidate_bricks.iter()
            .find(|brick| brick.coordinate == [1, 0, 0]).unwrap();
        assert!(post_receiver.active);
        assert_eq!(post_receiver.resolution, 4);
    }

    #[test]
    fn coarsest_support_reactivation_combines_donors_and_honors_constraints() {
        let (topology, mut fields) = setup(
            vec![
                brick(0, [0, 0], 4, true),
                brick(1, [1, 0], 8, false),
                brick(2, [2, 0], 8, true),
            ],
            [24, 8],
        );
        for cell in &topology.graph.cells {
            let id = cell.id as usize;
            fields.density[id] = 1.0;
            fields.cell_velocity[2 * id] = if cell.brick_key == Some(0) { 20.0 } else { -20.0 };
        }
        let options = coarsest_support_options();
        let projected = plan_projected_transport_support_with_options(
            &topology, &fields, 1.0, 1.0, &options, true,
        ).unwrap();
        let receiver = projected.candidate_bricks.iter()
            .find(|brick| brick.key == 1).unwrap();
        assert!(receiver.active);
        assert_eq!(receiver.resolution, 4);
        let post = plan_resolution(
            &topology,
            &fields,
            &initialize_resolution_policy(&topology),
            1.0,
            1.0,
            &options,
        ).unwrap();
        let receiver = post.candidate_bricks.iter()
            .find(|brick| brick.key == 1).unwrap();
        assert!(receiver.active);
        assert_eq!(receiver.resolution, 4);

        let mut constrained = options.clone();
        constrained.static_boundary_floor_by_brick.insert(1, 4);
        constrained.refinement_regions.push(ResolutionRegion {
            minimum_fine: [8.0, 0.0],
            maximum_fine: [16.0, 8.0],
            minimum_cell_width: 1,
            maximum_cell_width: Some(2),
        });
        let projected = plan_projected_transport_support_with_options(
            &topology, &fields, 1.0, 1.0, &constrained, true,
        ).unwrap();
        assert_eq!(projected.candidate_bricks.iter()
            .find(|brick| brick.key == 1).unwrap().resolution, 4);

        constrained.frozen_brick_keys.insert(1);
        let projected = plan_projected_transport_support_with_options(
            &topology, &fields, 1.0, 1.0, &constrained, true,
        ).unwrap();
        assert_eq!(projected.candidate_bricks.iter()
            .find(|brick| brick.key == 1).unwrap().resolution, 8);
    }

    #[test]
    fn projected_support_does_not_coarsen_an_active_receiver() {
        let (topology, mut fields) = setup(
            vec![brick(0, [0, 0], 4, true), brick(1, [1, 0], 8, true)],
            [16, 8],
        );
        for cell in &topology.graph.cells {
            if cell.brick_key == Some(0) {
                let id = cell.id as usize;
                fields.density[id] = 1.0;
                fields.cell_velocity[2 * id] = 20.0;
            }
        }
        let projected = plan_projected_transport_support_with_options(
            &topology,
            &fields,
            1.0,
            1.0,
            &coarsest_support_options(),
            true,
        ).unwrap();
        assert_eq!(projected.candidate_bricks.iter()
            .find(|brick| brick.key == 1).unwrap().resolution, 8);
    }

    #[test]
    fn free_leaf_claims_do_not_overlap_monotonic_fallback_ids() {
        let (topology, mut fields) = setup(vec![brick(0, [1, 1], 8, true)], [24, 24]);
        fields.density.fill(0.99);
        let mut options = coarsest_support_options();
        options.free_leaf_ids = vec![1];

        let projected = plan_projected_transport_support_with_options(
            &topology,
            &fields,
            1.0 / 30.0,
            0.05,
            &options,
            true,
        ).unwrap();
        assert!(projected.claimed_leaf_ids.len() > options.free_leaf_ids.len());
        assert_eq!(
            projected.claimed_leaf_ids.iter().copied().collect::<BTreeSet<_>>().len(),
            projected.claimed_leaf_ids.len(),
        );
        compile_topology::<2>(TopologySeed {
            dimensions: [24, 24, 1],
            generation: 2,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Closed; 6],
            bricks: projected.candidate_bricks,
        }).unwrap();

        let post = plan_resolution(
            &topology,
            &fields,
            &initialize_resolution_policy(&topology),
            1.0 / 30.0,
            0.05,
            &options,
        ).unwrap();
        assert!(post.receipt.claimed_leaf_ids.len() > options.free_leaf_ids.len());
        assert_eq!(
            post.receipt.claimed_leaf_ids.iter().copied().collect::<BTreeSet<_>>().len(),
            post.receipt.claimed_leaf_ids.len(),
        );
        compile_topology::<2>(TopologySeed {
            dimensions: [24, 24, 1],
            generation: 2,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Closed; 6],
            bricks: post.candidate_bricks,
        }).unwrap();
    }

    #[test]
    fn ts_golden_projected_support_changes_receiver_only() {
        let (topology, mut fields) = setup(
            vec![
                brick(0, [0, 0], 8, true),
                brick(1, [1, 0], 4, false),
                brick(3, [3, 0], 1, true),
            ],
            [32, 8],
        );
        for cell in &topology.graph.cells {
            if cell.brick_key == Some(0) && cell.minimum[0] == 7.0 {
                fields.density[cell.id as usize] = 1.0;
                fields.cell_velocity[2 * cell.id as usize] = 1.0;
            }
        }
        let d = plan_projected_transport_support(
            &topology,
            &fields,
            1.0 / 60.0,
            0.05,
            &ActivityPolicy::default(),
            Some(4),
            Some(256),
            &[],
            false,
        )
        .unwrap();
        assert_eq!(
            d.candidate_bricks
                .iter()
                .map(|b| (b.key, b.resolution, b.active))
                .collect::<Vec<_>>(),
            vec![(0, 8, true), (1, 4, true), (3, 1, true)]
        );
        assert_eq!(d.demanded_brick_keys, BTreeSet::from([1]));
    }

    #[test]
    fn projected_support_keeps_a_coarse_interface_halo_against_inward_mean_flow() {
        let (topology, mut fields) = setup(vec![brick(0, [0, 0], 1, true)], [8, 16]);
        fields.density[0] = 0.99;
        fields.cell_velocity[1] = -2.0;
        let mut geometric = options();
        geometric.translation_invariant_motion_sizing = true;
        let planned = plan_resolution(
            &topology,
            &fields,
            &initialize_resolution_policy(&topology),
            1.0 / 30.0,
            0.05,
            &geometric,
        )
        .unwrap();
        let source = planned
            .receipt
            .bricks
            .iter()
            .find(|brick| brick.brick_key == 0)
            .unwrap();
        assert_eq!(source.support_mask & 0x1ef, 0x1ef);
        let d = plan_projected_transport_support(
            &topology,
            &fields,
            1.0 / 30.0,
            0.05,
            &ActivityPolicy::default(),
            Some(4),
            Some(256),
            &[],
            true,
        )
        .unwrap();
        assert!(d.candidate_bricks.iter().any(|brick| {
            brick.active && brick.coordinate[..2] == [0, 1] && brick.resolution == 1
        }), "{:?}", d.candidate_bricks);
    }

    #[test]
    fn ts_golden_region_cap_closes_outward_halo() {
        let (topology, mut fields) = setup(
            vec![brick(0, [0, 0], 8, true), brick(1, [1, 0], 8, true)],
            [16, 8],
        );
        fields.density.fill(1.0);
        let mut o = options();
        o.refinement_regions.push(ResolutionRegion {
            minimum_fine: [0.0, 0.0],
            maximum_fine: [8.0, 8.0],
            minimum_cell_width: 8,
            maximum_cell_width: None,
        });
        let d = plan_resolution(
            &topology,
            &fields,
            &initialize_resolution_policy(&topology),
            1.0 / 60.0,
            0.05,
            &o,
        )
        .unwrap();
        assert_eq!(
            d.receipt
                .bricks
                .iter()
                .map(|r| r.scheduled_resolution)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn ts_golden_reflection_pair_accumulates_credit() {
        let (topology, mut fields) = setup(
            vec![brick(0, [0, 0], 8, true), brick(1, [1, 0], 8, true)],
            [16, 8],
        );
        fields.density.fill(1.0);
        let mut state = initialize_resolution_policy(&topology);
        for h in state.history.values_mut() {
            h.quiet_epochs = 8;
        }
        let mut o = options();
        o.policy.coarse_first = false;
        o.policy.prepare_bricks_per_frame = 1;
        o.policy.demote_epochs = 1;
        let first = plan_resolution(&topology, &fields, &state, 1.0 / 60.0, 0.05, &o).unwrap();
        assert_eq!(
            (
                first.receipt.demoted_brick_count,
                first.state.scheduling_credits
            ),
            (0, 1)
        );
        let second =
            plan_resolution(&topology, &fields, &first.state, 1.0 / 60.0, 0.05, &o).unwrap();
        assert_eq!(
            (
                second.receipt.demoted_brick_count,
                second.state.scheduling_credits
            ),
            (2, 0)
        );
    }

    #[test]
    fn ts_golden_zero_budget_defers_every_ordinary_demotion() {
        let (topology, mut fields) = setup(
            vec![brick(0, [0, 0], 8, true), brick(1, [1, 0], 8, true)],
            [16, 8],
        );
        fields.density.fill(1.0);
        let mut state = initialize_resolution_policy(&topology);
        for h in state.history.values_mut() {
            h.quiet_epochs = 8;
        }
        let mut o = options();
        o.policy.coarse_first = false;
        o.policy.prepare_bricks_per_frame = 0;
        o.policy.demote_epochs = 1;
        let d = plan_resolution(&topology, &fields, &state, 1.0 / 60.0, 0.05, &o).unwrap();
        assert_eq!(d.receipt.demoted_brick_count, 0);
        assert_eq!(d.receipt.deferred_demotion_count, 2);
        assert_eq!(d.state.scheduling_credits, 0);
    }

    #[test]
    fn ts_golden_capacity_fault_rolls_back_allocated_page() {
        let (topology, mut fields) = setup(vec![brick(0, [0, 0], 8, true)], [16, 8]);
        for cell in &topology.graph.cells {
            if cell.minimum[0] == 7.0 {
                fields.density[cell.id as usize] = 1.0;
                fields.cell_velocity[2 * cell.id as usize] = 1.0;
            }
        }
        let mut o = options();
        o.maximum_leaves = Some(1);
        let d = plan_resolution(
            &topology,
            &fields,
            &initialize_resolution_policy(&topology),
            1.0 / 60.0,
            0.05,
            &o,
        )
        .unwrap();
        assert_ne!(d.receipt.fault_bits, 0);
        assert_eq!(d.candidate_bricks.len(), 1);
        assert_eq!(d.receipt.candidate_generation, 1);
    }

    #[test]
    fn macro_span_participates_in_physical_width_grading() {
        let mut macro_brick = brick(0, [0, 0], 8, true);
        macro_brick.span_bricks = 2;
        let (topology, mut fields) = setup(vec![macro_brick, brick(1, [2, 0], 8, true)], [24, 16]);
        fields.density.fill(1.0);
        let mut o = options();
        o.refinement_regions.push(ResolutionRegion {
            minimum_fine: [0.0, 0.0],
            maximum_fine: [16.0, 16.0],
            minimum_cell_width: 16,
            maximum_cell_width: None,
        });
        let d = plan_resolution(
            &topology,
            &fields,
            &initialize_resolution_policy(&topology),
            1.0 / 60.0,
            0.05,
            &o,
        )
        .unwrap();
        assert_eq!(
            d.receipt
                .bricks
                .iter()
                .filter(|r| r.brick_key <= 1)
                .map(|r| r.scheduled_resolution)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
        assert_eq!(d.receipt.allocated_brick_count, 0);
        assert_eq!(d.receipt.fault_bits, 0);
    }

    #[test]
    fn fresh_surface_proof_advances_coarse_first_rung() {
        let (topology, mut fields) = setup(vec![brick(0, [0, 0], 8, true)], [8, 8]);
        for cell in &topology.graph.cells {
            if cell.minimum[0] < 4.0 {
                fields.density[cell.id as usize] = 1.0;
            }
        }
        let mut state = initialize_resolution_policy(&topology);
        let h = state.history.get_mut(&0).unwrap();
        h.proof_epochs = 1;
        h.surface_proof = Some(SurfaceProofState {
            generation_by_target_resolution: BTreeMap::from([(4, 1)]),
        });
        let d = plan_resolution(&topology, &fields, &state, 1.0 / 60.0, 0.05, &options()).unwrap();
        assert_eq!(d.receipt.bricks[0].scheduled_resolution, 4);
        assert_eq!(d.receipt.bricks[0].plan_reasons, 16);
    }

    #[test]
    fn deep_liquid_topology_is_invariant_to_uniform_translation() {
        let mut bricks = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                bricks.push(brick((x + 3 * y) as u32, [x, y], 8, true));
            }
        }
        let (topology, mut fields) = setup(bricks, [24, 24]);
        // Conservative transfers can leave nominally full cells a few ulps
        // below one.  They remain bulk because occupancy is classified against
        // capacity and neighbouring liquid, independently of stored normals.
        fields.density.fill(1.0 - 4.0 * f32::EPSILON);
        fields.interface_normal.fill(1.0);
        let stationary = fields.clone();
        for velocity in fields.cell_velocity.chunks_exact_mut(2) {
            // Cell velocities use finest-cell coordinates: 200 cells/s is
            // 10 m/s for this 5 cm grid, representative of fast free fall.
            velocity.copy_from_slice(&[0.0, -200.0]);
        }
        let mut state = initialize_resolution_policy(&topology);
        state.history.get_mut(&4).unwrap().proof_epochs = 1;
        let plan = |sample: &Fields| {
            let mut geometric = options();
            geometric.translation_invariant_motion_sizing = true;
            plan_resolution(&topology, sample, &state, 1.0 / 30.0, 0.05, &geometric).unwrap()
        };
        let still = plan(&stationary);
        let falling = plan(&fields);
        let centre = |decision: &ResolutionPolicyDecision| {
            decision
                .receipt
                .bricks
                .iter()
                .find(|record| record.brick_key == 4)
                .cloned()
                .unwrap()
        };
        let still = centre(&still);
        let falling = centre(&falling);
        assert_eq!(falling.requested_resolution, still.requested_resolution);
        assert_eq!(falling.scheduled_resolution, still.scheduled_resolution);
        assert_eq!(falling.score_byte, still.score_byte);
        assert_eq!(falling.reasons, still.reasons);
        assert_eq!(falling.plan_reasons, still.plan_reasons);
        assert_eq!(falling.reasons & activity_reason::SURFACE, 0);
        assert_eq!(falling.reasons & activity_reason::VELOCITY_FLOOR, 0);
        // The neighbours remain B8 surface bricks, so 2:1 closure raises the
        // bulk request to B4.  Uniform translation must not raise it further.
        assert_eq!(falling.scheduled_resolution, 4);
        assert_eq!(falling.plan_reasons, 2048);
    }

    #[test]
    fn geometric_surface_sizing_uses_velocity_variation_while_baseline_keeps_speed_floor() {
        let (topology, mut moving) = setup(vec![brick(0, [0, 0], 4, true)], [8, 8]);
        for cell in &topology.graph.cells {
            if cell.center[0] < 4.0 {
                let id = cell.id as usize;
                moving.density[id] = 1.0;
                moving.cell_velocity[2 * id + 1] = -200.0;
                moving.interface_normal[2 * id] = 1.0;
            }
        }
        let still = {
            let mut fields = moving.clone();
            fields.cell_velocity.fill(0.0);
            fields
        };
        let mut state = initialize_resolution_policy(&topology);
        let history = state.history.get_mut(&0).unwrap();
        history.proof_epochs = 1;
        history.surface_proof = Some(SurfaceProofState {
            generation_by_target_resolution: BTreeMap::from([(2, 1)]),
        });
        let mut geometric = options();
        geometric.translation_invariant_motion_sizing = true;
        let plan = |fields: &Fields, options: &ResolutionPolicyOptions| {
            plan_resolution(&topology, fields, &state, 1.0 / 30.0, 0.05, options)
                .unwrap()
                .receipt
                .bricks[0]
                .clone()
        };
        let stationary = plan(&still, &geometric);
        let translated = plan(&moving, &geometric);
        assert_eq!(translated.requested_resolution, stationary.requested_resolution);
        assert_eq!(translated.scheduled_resolution, stationary.scheduled_resolution);
        assert_eq!(translated.reasons & activity_reason::VELOCITY_FLOOR, 0);

        let baseline = plan(&moving, &options());
        assert_eq!(baseline.requested_resolution, 8);
        assert_ne!(baseline.reasons & activity_reason::VELOCITY_FLOOR, 0);

        let mut sheared = moving.clone();
        for cell in &topology.graph.cells {
            if cell.center[0] < 4.0 {
                let id = cell.id as usize;
                sheared.cell_velocity[2 * id + 1] = if cell.center[1] < 4.0 {
                    -200.0
                } else {
                    200.0
                };
            }
        }
        let impact = plan(&sheared, &geometric);
        assert_eq!(impact.requested_resolution, 8);
        assert_ne!(impact.reasons & activity_reason::VELOCITY_FLOOR, 0);
    }

    #[test]
    fn geometric_translation_still_activates_swept_receiver_pages() {
        let (topology, mut fields) = setup(
            vec![brick(0, [0, 1], 4, true), brick(1, [0, 0], 4, false)],
            [8, 16],
        );
        for cell in &topology.graph.cells {
            if cell.brick_key == Some(0) {
                let id = cell.id as usize;
                fields.density[id] = 1.0;
                fields.cell_velocity[2 * id + 1] = -200.0;
            }
        }
        let mut geometric = options();
        geometric.translation_invariant_motion_sizing = true;
        let decision = plan_resolution(
            &topology,
            &fields,
            &initialize_resolution_policy(&topology),
            1.0 / 30.0,
            0.05,
            &geometric,
        )
        .unwrap();
        let donor = decision
            .receipt
            .bricks
            .iter()
            .find(|record| record.brick_key == 0)
            .unwrap();
        let receiver = decision
            .receipt
            .bricks
            .iter()
            .find(|record| record.brick_key == 1)
            .unwrap();
        assert_eq!(donor.reasons & activity_reason::VELOCITY_FLOOR, 0);
        assert_ne!(donor.swept_support_mask, 0);
        assert!(receiver.candidate_active);
    }
}
