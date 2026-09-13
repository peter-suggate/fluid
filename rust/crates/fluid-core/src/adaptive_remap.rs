//! Experimental whole-step cellwise conservative remap for the 2-D advance lab.
//!
//! This module deliberately owns no world orchestration.  It consumes the
//! accepted graph and the post-projection velocity field, traces each shared
//! lattice point once, corrects every physical subface chain to its discrete
//! flux, and optionally gathers the old PLIC material into the resulting
//! pre-images.  Moving open regions and sources are M7 work and are rejected
//! by the committing path rather than being approximated here.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::band_projection::cleanup_pressure_receiver_rates_2d;
use crate::numerics::{
    clip_segment, streamfunction_extension_rates_2d, FaceConsistentVelocity2d, FaceVelocitySample2d,
};
use crate::{Fields, Graph, NumericalFault, ValidationError};

const GEOMETRY_EPS: f64 = 1.0e-12;
const CAPACITY_EPS: f64 = 1.0e-6;
const VOLUME_MARGIN: f64 = 8.0 * f32::EPSILON as f64;
const AREA_BALANCE_TOLERANCE: f64 = 1.0e-12;
const AREA_IDENTITY_TOLERANCE: f64 = 1.0e-10;
const CONTINUITY_TARGET: f64 = 2.0 * f32::EPSILON as f64;

pub const CORRECTION_DELTA_OVER_H_HISTOGRAM_BOUNDS: [f64; 10] = [
    1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0,
];

type Point = [f64; 2];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CellwiseClosure {
    /// Measure the receiver certificate without changing any face rate.
    None,
    /// One deterministic SOLA-VOF-style sweep into non-receiver air.
    Local,
    /// Compatibility projection on the conservative pre-trace receiver band.
    #[default]
    BandProjection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CellwiseRemapOptions {
    /// Sample intervals per finest-lattice unit of physical edge (`s` in the
    /// plan).  At `s=1`, a unit edge traces only its endpoints; correction adds
    /// an interpolated midpoint without incrementing the trace counter.
    pub edge_samples: usize,
    /// Fixed RK4 segments over the frame.  This is tracing refinement, not a
    /// transport substep: gather and commit still happen exactly once.
    pub trace_segments: usize,
    pub closure: CellwiseClosure,
    /// False runs M2 geometry and receiver diagnostics without touching fields.
    pub commit_material: bool,
}

impl Default for CellwiseRemapOptions {
    fn default() -> Self {
        Self {
            edge_samples: 1,
            trace_segments: 1,
            closure: CellwiseClosure::BandProjection,
            commit_material: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CellwiseRemapReceipt {
    pub material_committed: bool,
    pub closure: CellwiseClosure,
    /// User-configured spatial edge resolution.  A committing remap may raise
    /// the effective value below to ensure every unit subface has a genuinely
    /// traced interior point.
    pub requested_edge_samples: usize,
    /// Effective sample intervals per finest-lattice unit.
    pub edge_samples: usize,
    pub adaptive_edge_refinement_passes: usize,
    pub adaptive_edge_points_inserted: usize,
    pub adaptive_edge_refinement_exhausted: bool,
    pub adaptive_edge_max_dyadic_depth: usize,
    pub adaptive_edge_max_points_per_subface: usize,
    pub requested_trace_segments: usize,
    /// Effective tracing refinement after enforcing the per-segment Courant
    /// bound. Material gather and commit still occur once per frame.
    pub trace_segments: usize,
    pub trace_extension_generations: u8,
    pub extension_cell_visits: usize,
    pub support_lattice_entries: usize,
    pub projected_subface_rates_preserved: usize,
    pub original_subface_rates_preserved: usize,
    pub extension_subface_rates_republished: usize,
    pub pressure_rate_cleanup_cells: usize,
    pub pressure_rate_cleanup_components: usize,
    pub pressure_rate_cleanup_open_components: usize,
    pub pressure_rate_cleanup_changed_subfaces: usize,
    pub pressure_rate_cleanup_max_normalized_divergence_before: f64,
    pub pressure_rate_cleanup_max_normalized_divergence_after: f64,
    pub pressure_rate_cleanup_max_normalized_roundoff_bound: f64,
    pub pressure_rate_cleanup_max_absolute_rate_change: f64,
    pub pressure_rate_cleanup_max_relative_rate_change: f64,
    pub pressure_rate_cleanup_zero_original_rate_change_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pressure_rate_cleanup_rejection: Option<String>,
    pub traces: usize,
    pub chain_points: usize,
    pub interpolated_correction_points: usize,
    pub rk_evaluations: usize,
    pub support_extrapolation_samples: usize,
    /// RK samples that used a cell-local RT0 profile instead of the compiled
    /// global streamfunction field.
    pub trace_velocity_fallback_samples: usize,
    pub clips: usize,
    pub constrained_pathline_segments: usize,
    /// Changes between successively sampled RK stage region signatures. This
    /// is a discontinuity proxy; neighbouring regions may carry equal values,
    /// and a cross-and-return between stage samples is not observed.
    pub rt0_region_transitions: usize,
    /// Extra accepted-endpoint sampler calls used only to classify RT0 region
    /// transitions. These are separate from `rk_evaluations`.
    pub rt0_region_diagnostic_samples: usize,
    pub pregeometry_receiver_band: usize,
    pub pregeometry_receiver_band_outside_operator: usize,
    pub receiver_band_stabilization_passes: usize,
    pub pregeometry_max_abs_normalized_divergence: f64,
    pub pregeometry_certificate_violations: usize,
    pub pregeometry_certificate_violations_in_operator: usize,
    pub pregeometry_certificate_violations_outside_operator: usize,
    pub pregeometry_max_abs_normalized_divergence_in_operator: f64,
    pub pregeometry_max_abs_normalized_divergence_outside_operator: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pregeometry_max_divergence_cell: Option<u32>,
    pub pregeometry_max_divergence_cell_pressure_member: bool,
    pub pregeometry_max_divergence: f64,
    pub pregeometry_max_divergence_cell_capacity: f64,
    pub pregeometry_max_divergence_cell_density: f64,
    pub pregeometry_max_divergence_cell_liquid_volume: f64,
    pub postgeometry_receivers_outside_pregeometry_band: usize,
    pub closure_accepted: bool,
    pub closure_unknowns: usize,
    pub closure_components: usize,
    pub closure_open_components: usize,
    pub closure_enclosed_components: usize,
    pub closure_infeasible_enclosed_components: usize,
    pub closure_max_enclosed_component_normalized_defect: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub closure_worst_infeasible_component_root: Option<usize>,
    pub closure_worst_infeasible_component_cells: usize,
    pub closure_worst_infeasible_component_liquid_cells: usize,
    pub closure_worst_infeasible_component_capacity: f64,
    pub closure_worst_infeasible_component_signed_defect: f64,
    pub closure_iterations: usize,
    pub closure_initial_normalized_residual: f64,
    pub closure_measured_normalized_residual: f64,
    pub closure_fixed_projected_subfaces: usize,
    pub closure_changed_subfaces: usize,
    pub closure_max_abs_rate_change: f64,
    pub closure_max_relative_rate_change: f64,
    pub closure_zero_original_rate_change_count: usize,
    pub max_courant_before_closure: f64,
    pub max_courant_after_closure: f64,
    pub primary_liquid_velocity_mean: [f64; 2],
    pub primary_full_liquid_velocity_minimum: [f64; 2],
    pub primary_full_liquid_velocity_maximum: [f64; 2],
    pub tracing_full_liquid_velocity_minimum: [f64; 2],
    pub tracing_full_liquid_velocity_maximum: [f64; 2],
    pub tracing_full_liquid_max_velocity_mismatch: f64,
    pub tracing_max_abs_finite_difference_divergence: f64,
    pub tracing_max_velocity_gradient: f64,
    pub tracing_max_velocity_gradient_point: [f64; 2],
    pub streamfunction_max_integrated_flux_residual: f64,
    pub streamfunction_range: f64,
    pub tracing_max_streamfunction_absolute_drift: f64,
    pub tracing_max_streamfunction_relative_drift: f64,
    pub receivers: usize,
    pub receivers_outside_operator: usize,
    pub receiver_band_width: u8,
    pub receiver_band_unextended: usize,
    pub max_courant_liquid: f64,
    pub max_courant_all: f64,
    /// Diagnostic requested by the plan: largest outward-positive `dt*div/C`.
    pub max_dt_div_over_capacity: f64,
    pub max_dt_div_over_capacity_outside_operator: f64,
    /// The actual non-compression certificate: `max(-dt*div/C)`.
    /// A receiver is certified when this is non-positive up to roundoff.
    pub max_compression_ratio: f64,
    pub max_compression_ratio_outside_operator: f64,
    /// Folded pre-images formed directly from the RK4 trajectories, before
    /// discrete-continuity area correction moves chain interiors.
    pub pre_correction_self_intersections: usize,
    pub self_intersections: usize,
    pub pre_correction_liquid_receiver_folds: usize,
    pub corrected_liquid_receiver_folds: usize,
    pub pre_correction_receiver_band_folds: usize,
    pub corrected_receiver_band_folds: usize,
    pub pre_correction_bbox_liquid_folds: usize,
    pub corrected_bbox_liquid_folds: usize,
    /// Folded pre-images whose convex hull intersects an actual donor PLIC.
    /// The hull contains every possible lobe of a self-crossing polygon, so a
    /// miss proves that the fold cannot affect transported material.
    pub pre_correction_convex_hull_liquid_folds: usize,
    pub corrected_convex_hull_liquid_folds: usize,
    pub pre_correction_dry_folds: usize,
    pub corrected_dry_folds: usize,
    /// Exclusive adjacency proxies for folded receiver cells. These do not
    /// prove that a characteristic crossed the named discontinuity.
    pub pre_correction_liquid_fold_locations: FoldLocations,
    pub corrected_liquid_fold_locations: FoldLocations,
    pub pre_correction_fold_details: FoldDetailsByLocation,
    pub corrected_fold_details: FoldDetailsByLocation,
    pub correction_count: usize,
    pub correction_delta_over_h_sum: f64,
    pub correction_delta_over_h_sum_squares: f64,
    pub max_correction_delta_over_h: f64,
    pub correction_delta_over_h_histogram_bounds: [f64; 10],
    pub correction_delta_over_h_histogram: [usize; 11],
    pub receiver_correction_count: usize,
    pub receiver_correction_delta_over_h_sum: f64,
    pub receiver_correction_delta_over_h_sum_squares: f64,
    pub max_receiver_correction_delta_over_h: f64,
    pub gather_clips: usize,
    pub gather_initial_liquid_volume: f64,
    pub gather_final_liquid_volume: f64,
    pub gather_absolute_volume_error: f64,
    pub gather_volume_roundoff_bound: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gather_worst_donor: Option<u32>,
    pub gather_worst_donor_volume_error: f64,
    pub gather_worst_donor_roundoff_bound: f64,
    pub gather_worst_donor_minimum: [f64; 2],
    pub gather_worst_donor_maximum: [f64; 2],
    pub gather_worst_donor_density: f64,
    pub gather_worst_donor_capacity: f64,
    pub gather_worst_donor_gathered_volume: f64,
    pub gather_worst_donor_expected_volume: f64,
    pub gather_worst_donor_interface_normal: [f64; 2],
    pub gather_worst_donor_interface_offset: f64,
    /// Corrected cell pre-images with materially negative signed area. The
    /// count covers every represented cell, including dry diagnostics.
    pub negative_preimage_cells: usize,
    /// Negative pre-images whose footprint can gather old liquid. These are
    /// rejected before gather because no orientation-preserving material map
    /// can have negative area.
    pub negative_liquid_receiver_preimages: usize,
    pub min_preimage_area: f64,
    /// Net outward rate across the represented sparse support boundary.
    pub net_boundary_flux: f64,
    /// Aggregate area identity residual after accounting for that boundary
    /// flux.  This distinguishes a broken shared tiling from an open support.
    pub area_balance_relative_error: f64,
    /// Literal M2 gate: pre-image area versus represented open capacity.
    pub tiling_relative_error: f64,
    pub max_area_identity_error: f64,
    pub closure_adjustments: usize,
    pub closure_unresolved: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FoldLocations {
    pub wall: usize,
    pub seam: usize,
    pub support_boundary: usize,
    pub interior: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FoldDetailsByLocation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wall: Option<FoldCellDetail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seam: Option<FoldCellDetail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub support_boundary: Option<FoldCellDetail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interior: Option<FoldCellDetail>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FoldCellDetail {
    pub cell_id: u32,
    pub pressure_member: bool,
    pub polygon: Vec<[f64; 2]>,
    pub divergence: f64,
    pub capacity: f64,
    pub incident_subfaces: Vec<FoldSubfaceDetail>,
    pub boundary_support_extrapolation_samples: usize,
    pub boundary_constrained_pathline_segments: usize,
    pub boundary_rt0_region_transitions: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FoldSubfaceDetail {
    pub subface_id: u32,
    pub rate: f64,
    pub raw_swept_rate: f64,
    pub area_correction_coefficient: f64,
    pub correction_delta_over_h: f64,
}

#[derive(Clone, Debug)]
struct TracedPoint {
    /// Forward storage along the backward characteristic: fixed point first,
    /// final pre-image point last.
    path: Vec<Point>,
    /// RK4 quadrature of `1/2 integral(x dy - y dx)` along the backward
    /// characteristic. This supplies pathline-sided swept area without the
    /// O(k^-2) chord quadrature error of the retained diagnostic polyline.
    signed_area_integral: f64,
    support_extrapolation_samples: usize,
    constrained_segments: usize,
    rt0_region_transitions: usize,
}

impl TracedPoint {
    fn end(&self) -> Point {
        *self.path.last().expect("a traced path contains its start")
    }
}

#[derive(Clone, Debug)]
struct EdgeChain {
    axis: usize,
    traced: Vec<Point>,
}

#[derive(Clone, Debug)]
struct Geometry {
    uncorrected_chains: Vec<EdgeChain>,
    corrected_chains: Vec<EdgeChain>,
    uncorrected_polygons: Vec<Vec<Point>>,
    polygons: Vec<Vec<Point>>,
    divergence: Vec<f64>,
    identity_error: f64,
    tiling_error: f64,
    pre_correction_fold_mask: Vec<bool>,
    corrected_fold_mask: Vec<bool>,
    correction_deltas_over_h: Vec<f64>,
    raw_swept_rates: Vec<f64>,
    correction_coefficients: Vec<f64>,
    chain_points: usize,
    interpolated_correction_points: usize,
}

/// Trace, correct, diagnose, gather, and commit one whole frame.
///
/// `commit` runs exactly once after density and gamma have passed the volume
/// bounds.  It is not called by geometry-only probe mode.
pub fn transport_volume_cellwise_with_commit(
    graph: &Graph,
    fields: &mut Fields,
    dt: f32,
    options: CellwiseRemapOptions,
    mut commit: impl FnMut(f32, &mut Fields) -> Result<(), ValidationError>,
) -> Result<CellwiseRemapReceipt, ValidationError> {
    validate_inputs(graph, fields, dt, options)?;
    if options.commit_material {
        if fields.solid_motion_active {
            return Err(ValidationError(
                "cellwise remap M7: moving capacity is not implemented".into(),
            ));
        }
        if fields.source_rate.iter().any(|&rate| rate != 0.0) {
            return Err(ValidationError(
                "cellwise remap M7: source transport is not implemented".into(),
            ));
        }
    }

    let mut receipt = CellwiseRemapReceipt {
        closure: options.closure,
        requested_edge_samples: options.edge_samples,
        edge_samples: options.edge_samples,
        requested_trace_segments: options.trace_segments,
        trace_segments: options.trace_segments,
        correction_delta_over_h_histogram_bounds: CORRECTION_DELTA_OVER_H_HISTOGRAM_BOUNDS,
        ..Default::default()
    };
    let base_tracer = FaceConsistentVelocity2d::new(graph, fields, dt)?;
    receipt.trace_extension_generations = base_tracer.extension_generations();
    receipt.extension_cell_visits = base_tracer.extension_cell_visits();
    receipt.support_lattice_entries = base_tracer.support_lattice_entries();
    receipt.projected_subface_rates_preserved = base_tracer.projected_subface_rates_preserved();
    receipt.original_subface_rates_preserved = base_tracer.original_subface_rates_preserved();
    receipt.extension_subface_rates_republished = base_tracer.extension_subface_rates_republished();
    let rates_before_closure = base_tracer.subface_rates().to_vec();
    let mut rates = rates_before_closure.clone();
    let mut receiver_band = conservative_pretrace_receiver_band(graph, fields, &rates, dt as f64)?;
    for pass in 1..=graph.cells.len() + 1 {
        receipt.receiver_band_stabilization_passes = pass;
        rates.clone_from(&rates_before_closure);
        if options.closure == CellwiseClosure::BandProjection {
            let pressure_cleanup = match cleanup_pressure_receiver_rates_2d(
                graph,
                fields,
                &mut rates,
                &receiver_band,
                dt,
            ) {
                Ok(receipt) => receipt,
                Err(error) => {
                    receipt.pressure_rate_cleanup_rejection = Some(error.to_string());
                    receipt.closure_accepted = false;
                    if options.commit_material {
                        fields.fault = Some(NumericalFault {
                            stage: "cellwise-remap-pregeometry-continuity".into(),
                            index: 0,
                            observed: 1.0,
                            expected: CONTINUITY_TARGET as f32,
                        });
                    }
                    return Ok(receipt);
                }
            };
            receipt.pressure_rate_cleanup_cells = pressure_cleanup.cells;
            receipt.pressure_rate_cleanup_components = pressure_cleanup.components;
            receipt.pressure_rate_cleanup_open_components = pressure_cleanup.open_components;
            receipt.pressure_rate_cleanup_changed_subfaces = pressure_cleanup.changed_subfaces;
            receipt.pressure_rate_cleanup_max_normalized_divergence_before =
                pressure_cleanup.max_normalized_divergence_before;
            receipt.pressure_rate_cleanup_max_normalized_divergence_after =
                pressure_cleanup.max_normalized_divergence_after;
            receipt.pressure_rate_cleanup_max_normalized_roundoff_bound =
                pressure_cleanup.max_normalized_roundoff_bound;
            receipt.pressure_rate_cleanup_max_absolute_rate_change =
                pressure_cleanup.max_absolute_rate_change;
            receipt.pressure_rate_cleanup_max_relative_rate_change =
                pressure_cleanup.max_relative_rate_change;
            receipt.pressure_rate_cleanup_zero_original_rate_change_count =
                pressure_cleanup.zero_original_rate_change_count;
        }
        receipt.closure_adjustments = 0;
        receipt.closure_unresolved = 0;
        let initial_divergence = cell_divergence(graph, &rates);
        match options.closure {
            CellwiseClosure::None => receipt.closure_accepted = true,
            CellwiseClosure::Local => {
                local_continuity_closure(
                    graph,
                    fields,
                    &receiver_band,
                    &initial_divergence,
                    dt as f64,
                    &mut rates,
                    &mut receipt,
                );
                receipt.closure_accepted = true;
            }
            CellwiseClosure::BandProjection => {
                streamfunction_extension_rates_2d(
                    graph,
                    fields,
                    &mut rates,
                    &receiver_band,
                    false,
                )?;
                receipt.closure_accepted = true;
                receipt.closure_unknowns = receiver_band
                    .iter()
                    .enumerate()
                    .filter(|&(cell, &receiver)| {
                        receiver && fields.pressure_member.get(cell).copied().unwrap_or(0) == 0
                    })
                    .count();
                receipt.closure_fixed_projected_subfaces = graph
                    .subfaces
                    .iter()
                    .filter(|face| {
                        [face.negative_cell, face.positive_cell]
                            .into_iter()
                            .filter(|&cell| cell >= 0)
                            .any(|cell| {
                                fields
                                    .pressure_member
                                    .get(cell as usize)
                                    .copied()
                                    .unwrap_or(0)
                                    != 0
                            })
                    })
                    .count();
            }
        }
        if !receipt.closure_accepted {
            break;
        }
        let reached = conservative_pretrace_receiver_band(graph, fields, &rates, dt as f64)?;
        let mut expanded = false;
        for (member, reached) in receiver_band.iter_mut().zip(reached) {
            if reached && !*member {
                *member = true;
                expanded = true;
            }
        }
        if !expanded {
            break;
        }
        if pass == graph.cells.len() + 1 {
            return Err(ValidationError(
                "cellwise remap receiver band did not stabilize".into(),
            ));
        }
    }
    receipt.pregeometry_receiver_band = receiver_band.iter().filter(|&&v| v).count();
    receipt.pregeometry_receiver_band_outside_operator = receiver_band
        .iter()
        .enumerate()
        .filter(|&(cell, &receiver)| {
            receiver && fields.pressure_member.get(cell).copied().unwrap_or(0) == 0
        })
        .count();
    if options.closure == CellwiseClosure::BandProjection {
        streamfunction_extension_rates_2d(graph, fields, &mut rates, &receiver_band, true)?;
    }
    publish_rate_change_metrics(
        graph,
        fields,
        &rates_before_closure,
        &rates,
        &receiver_band,
        dt as f64,
        &mut receipt,
    );
    let tracer = if options.closure == CellwiseClosure::BandProjection {
        base_tracer.with_subface_rates(graph, &rates, &receiver_band)?
    } else {
        base_tracer.with_piecewise_subface_rates(graph, &rates)?
    };
    receipt.streamfunction_max_integrated_flux_residual =
        tracer.streamfunction_max_integrated_flux_residual();
    receipt.streamfunction_range = tracer.streamfunction_range();
    publish_liquid_velocity_diagnostics(graph, fields, &tracer, &mut receipt)?;
    publish_tracer_differential_diagnostics(graph, &tracer, &receiver_band, &mut receipt)?;
    // One whole-step gather remains Courant-unlimited, but its RK pathlines
    // must resolve travel across piecewise-RT0 regions. Treat the configured
    // value as a floor and keep each segment at or below Courant 0.25;
    // this is tracing refinement only and does not add material substeps.
    let courant_trace_segments = if options.commit_material {
        let travel = (4.0 * receipt.max_courant_after_closure).ceil().max(1.0) as usize;
        // A fast field can still have little strain, while a near-wall curl
        // patch can stretch adjacent characteristics strongly at modest
        // Courant number. Bound the sampled per-step velocity gradient as
        // well as travel so RK4 does not create a non-injective discrete map.
        let strain = (8.0 * dt as f64 * receipt.tracing_max_velocity_gradient)
            .ceil()
            .max(1.0) as usize;
        travel.max(strain)
    } else {
        options.trace_segments
    };
    if courant_trace_segments > 256 {
        return Err(ValidationError(format!(
            "cellwise remap tracing requires {courant_trace_segments} segments, above the supported 256"
        )));
    }
    let mut effective_options = options;
    // A unit face at s=1 contains only its two shared endpoints.  The old
    // correction then manufactured an untraced midpoint, so its sole area
    // degree of freedom did not belong to the characteristic map.  Keep the
    // requested value as the public configuration while ensuring committing
    // geometry has one actually traced interior point on every unit face.
    effective_options.edge_samples = options.edge_samples.max(2);
    effective_options.trace_segments = options.trace_segments.max(courant_trace_segments);
    receipt.edge_samples = effective_options.edge_samples;
    receipt.trace_segments = effective_options.trace_segments;
    let projected_divergence = cell_divergence(graph, &rates);
    publish_pregeometry_certificate(
        graph,
        fields,
        &receiver_band,
        &projected_divergence,
        dt as f64,
        &mut receipt,
    );
    let enforce_pregeometry_certificate =
        options.commit_material || options.closure != CellwiseClosure::None;
    if !receipt.closure_accepted
        || (enforce_pregeometry_certificate && receipt.pregeometry_certificate_violations != 0)
    {
        if options.commit_material {
            fields.fault = Some(NumericalFault {
                stage: "cellwise-remap-pregeometry-continuity".into(),
                index: 0,
                observed: receipt
                    .pregeometry_max_abs_normalized_divergence
                    .max(receipt.closure_max_enclosed_component_normalized_defect)
                    as f32,
                expected: CONTINUITY_TARGET as f32,
            });
        }
        return Ok(receipt);
    }
    let mut fixed_edges: Vec<Vec<Point>> = graph
        .subfaces
        .iter()
        .map(|face| {
            let touches_receiver_band = [face.negative_cell, face.positive_cell]
                .into_iter()
                .filter(|&cell| cell >= 0)
                .any(|cell| receiver_band[cell as usize]);
            let samples = if options.commit_material && touches_receiver_band {
                effective_options.edge_samples.saturating_mul(2)
            } else {
                effective_options.edge_samples
            };
            fixed_edge(face, samples)
        })
        .collect();
    if options.commit_material {
        receipt.adaptive_edge_refinement_passes = 1;
        receipt.adaptive_edge_points_inserted = graph
            .subfaces
            .iter()
            .zip(&fixed_edges)
            .map(|(face, points)| {
                let base = (face.measure as usize)
                    .saturating_mul(effective_options.edge_samples)
                    .saturating_add(1);
                points.len().saturating_sub(base)
            })
            .sum();
    }
    let mut traced = BTreeMap::new();
    let mut refinement_passes = receipt.adaptive_edge_refinement_passes;
    let (geometry, raw_receivers, receivers) = loop {
        trace_lattice_points(
            graph,
            fields,
            &tracer,
            dt as f64,
            effective_options.trace_segments,
            &fixed_edges,
            &mut traced,
            &mut receipt,
        )?;
        let geometry = build_geometry(
            graph,
            fields,
            dt as f64,
            &fixed_edges,
            &traced,
            &rates,
            &receiver_band,
        )?;
        let raw_receivers = receiver_mask(
            graph,
            fields,
            &geometry.uncorrected_polygons,
            &geometry.pre_correction_fold_mask,
            &mut receipt,
        );
        let receivers = receiver_mask(
            graph,
            fields,
            &geometry.polygons,
            &geometry.corrected_fold_mask,
            &mut receipt,
        );
        let refine_cells: Vec<bool> = (0..graph.cells.len())
            .map(|cell| {
                (geometry.pre_correction_fold_mask[cell]
                    && convex_hull_can_reach_liquid(
                        graph,
                        fields,
                        &geometry.uncorrected_polygons[cell],
                    ))
                    || (geometry.corrected_fold_mask[cell]
                        && convex_hull_can_reach_liquid(
                            graph,
                            fields,
                            &geometry.polygons[cell],
                        ))
            })
            .collect();
        if !options.commit_material || !refine_cells.iter().any(|&folded| folded) {
            break (geometry, raw_receivers, receivers);
        }
        if refinement_passes >= 32 {
            receipt.adaptive_edge_refinement_exhausted = true;
            break (geometry, raw_receivers, receivers);
        }
        let mut intervals = crossing_chain_intervals(
            graph,
            &geometry.uncorrected_chains,
            &refine_cells,
        );
        intervals.extend(crossing_chain_intervals(
            graph,
            &geometry.corrected_chains,
            &refine_cells,
        ));
        intervals.extend(corrected_only_area_intervals(
            graph,
            &geometry,
            &refine_cells,
        ));
        let inserted = refine_crossing_intervals(&mut fixed_edges, &intervals)?;
        if inserted == 0 {
            receipt.adaptive_edge_refinement_exhausted = true;
            break (geometry, raw_receivers, receivers);
        }
        refinement_passes += 1;
        receipt.adaptive_edge_refinement_passes = refinement_passes;
        receipt.adaptive_edge_points_inserted += inserted;
    };
    receipt.edge_samples = fixed_edges
        .iter()
        .map(|points| {
            points
                .windows(2)
                .map(|pair| {
                    let length = distance_squared(pair[0], pair[1]).sqrt();
                    (1.0 / length.max(f64::MIN_POSITIVE)).ceil() as usize
                })
                .max()
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0);
    receipt.adaptive_edge_max_points_per_subface =
        fixed_edges.iter().map(Vec::len).max().unwrap_or(0);
    let initial_samples = effective_options.edge_samples.saturating_mul(2).max(1);
    let ratio = receipt.edge_samples.div_ceil(initial_samples).max(1);
    receipt.adaptive_edge_max_dyadic_depth = ratio.next_power_of_two().ilog2() as usize;
    update_receiver_metrics(
        graph,
        fields,
        tracer.extension_depths(),
        dt as f64,
        &geometry.divergence,
        &rates,
        &receivers,
        &mut receipt,
    );

    publish_geometry_metrics(
        graph,
        fields,
        &geometry,
        &raw_receivers,
        &receivers,
        &traced,
        &rates,
        &fixed_edges,
        &mut receipt,
    );
    receipt.max_area_identity_error = geometry.identity_error;
    receipt.pre_correction_receiver_band_folds = geometry
        .pre_correction_fold_mask
        .iter()
        .zip(&receiver_band)
        .filter(|&(folded, receiver)| *folded && *receiver)
        .count();
    receipt.corrected_receiver_band_folds = geometry
        .corrected_fold_mask
        .iter()
        .zip(&receiver_band)
        .filter(|&(folded, receiver)| *folded && *receiver)
        .count();
    receipt.pre_correction_bbox_liquid_folds = geometry
        .pre_correction_fold_mask
        .iter()
        .zip(&geometry.uncorrected_polygons)
        .filter(|&(folded, polygon)| {
            *folded
                && candidate_donors(graph, polygon)
                    .into_iter()
                    .any(|donor| fields.density[donor] > 0.0)
        })
        .count();
    receipt.corrected_bbox_liquid_folds = geometry
        .corrected_fold_mask
        .iter()
        .zip(&geometry.polygons)
        .filter(|&(folded, polygon)| {
            *folded
                && candidate_donors(graph, polygon)
                    .into_iter()
                    .any(|donor| fields.density[donor] > 0.0)
        })
        .count();
    receipt.pre_correction_convex_hull_liquid_folds = geometry
        .pre_correction_fold_mask
        .iter()
        .zip(&geometry.uncorrected_polygons)
        .filter(|&(folded, polygon)| {
            *folded && convex_hull_can_reach_liquid(graph, fields, polygon)
        })
        .count();
    receipt.corrected_convex_hull_liquid_folds = geometry
        .corrected_fold_mask
        .iter()
        .zip(&geometry.polygons)
        .filter(|&(folded, polygon)| {
            *folded && convex_hull_can_reach_liquid(graph, fields, polygon)
        })
        .count();
    receipt.tiling_relative_error = geometry.tiling_error;
    receipt.net_boundary_flux = geometry.divergence.iter().sum();
    let open_area: f64 = graph
        .cells
        .iter()
        .enumerate()
        .map(|(i, cell)| fields.capacity[i] as f64 * cell.measure as f64)
        .sum();
    let preimage_area: f64 = geometry.polygons.iter().map(|p| signed_area(p)).sum();
    receipt.area_balance_relative_error =
        (preimage_area - (open_area - dt as f64 * receipt.net_boundary_flux)).abs()
            / open_area.max(1.0);
    receipt.postgeometry_receivers_outside_pregeometry_band =
        receivers_outside_band(&raw_receivers, &receivers, &receiver_band);

    receipt.min_preimage_area = f64::INFINITY;
    receipt.negative_preimage_cells = 0;
    receipt.negative_liquid_receiver_preimages = 0;
    for (cell, polygon) in geometry.polygons.iter().enumerate() {
        let area = signed_area(polygon);
        receipt.min_preimage_area = receipt.min_preimage_area.min(area);
        let capacity = fields.capacity[cell] as f64 * graph.cells[cell].measure as f64;
        if preimage_area_is_negative(area, capacity) {
            receipt.negative_preimage_cells += 1;
            if receivers[cell] {
                receipt.negative_liquid_receiver_preimages += 1;
            }
        }
    }
    if !receipt.min_preimage_area.is_finite() {
        receipt.min_preimage_area = 0.0;
    }

    if !options.commit_material {
        return Ok(receipt);
    }
    if reject_incomplete_receiver_band(fields, &receipt) {
        return Ok(receipt);
    }
    if receipt.negative_liquid_receiver_preimages != 0 {
        let (index, observed) = geometry
            .polygons
            .iter()
            .enumerate()
            .find_map(|(cell, polygon)| {
                let area = signed_area(polygon);
                let capacity = fields.capacity[cell] as f64 * graph.cells[cell].measure as f64;
                (receivers[cell] && preimage_area_is_negative(area, capacity))
                    .then_some((cell, area))
            })
            .expect("negative liquid receiver count identifies a cell");
        fields.fault = Some(NumericalFault {
            stage: "cellwise-remap-negative-preimage".into(),
            index: index as u32,
            observed: observed as f32,
            expected: 0.0,
        });
        return Ok(receipt);
    }
    if receipt.pre_correction_convex_hull_liquid_folds != 0
        || receipt.corrected_convex_hull_liquid_folds != 0
    {
        fields.fault = Some(NumericalFault {
            stage: "cellwise-remap-self-intersection".into(),
            index: 0,
            observed: receipt
                .pre_correction_liquid_receiver_folds
                .max(receipt.corrected_liquid_receiver_folds)
                .max(receipt.pre_correction_receiver_band_folds)
                .max(receipt.corrected_receiver_band_folds)
                .max(receipt.pre_correction_convex_hull_liquid_folds)
                .max(receipt.corrected_convex_hull_liquid_folds) as f32,
            expected: 0.0,
        });
        return Ok(receipt);
    }
    if receipt.area_balance_relative_error > AREA_BALANCE_TOLERANCE {
        fields.fault = Some(NumericalFault {
            stage: "cellwise-remap-area-balance".into(),
            index: 0,
            observed: receipt.area_balance_relative_error as f32,
            expected: AREA_BALANCE_TOLERANCE as f32,
        });
        return Ok(receipt);
    }
    if receipt.max_area_identity_error > AREA_IDENTITY_TOLERANCE {
        fields.fault = Some(NumericalFault {
            stage: "cellwise-remap-area-identity".into(),
            index: 0,
            observed: receipt.max_area_identity_error as f32,
            expected: AREA_IDENTITY_TOLERANCE as f32,
        });
        return Ok(receipt);
    }
    if receipt.closure_unresolved != 0 || receipt.max_compression_ratio > VOLUME_MARGIN {
        fields.fault = Some(NumericalFault {
            stage: "cellwise-remap-continuity".into(),
            index: 0,
            observed: receipt.max_compression_ratio as f32,
            expected: 0.0,
        });
        return Ok(receipt);
    }

    let next = gather_material(graph, fields, &geometry.polygons, &mut receipt)?;
    let global_gather_failed =
        receipt.gather_absolute_volume_error > receipt.gather_volume_roundoff_bound;
    let donor_gather_failed =
        receipt.gather_worst_donor_volume_error > receipt.gather_worst_donor_roundoff_bound;
    if global_gather_failed || donor_gather_failed {
        let (index, observed, expected) = if donor_gather_failed {
            (
                receipt.gather_worst_donor.unwrap_or(0),
                receipt.gather_worst_donor_volume_error,
                receipt.gather_worst_donor_roundoff_bound,
            )
        } else {
            (
                0,
                receipt.gather_absolute_volume_error,
                receipt.gather_volume_roundoff_bound,
            )
        };
        fields.fault = Some(NumericalFault {
            stage: "cellwise-remap-gather-conservation".into(),
            index,
            observed: observed as f32,
            expected: expected as f32,
        });
        return Ok(receipt);
    }
    for (i, &volume) in next.iter().enumerate() {
        let capacity = fields.capacity[i] as f64 * graph.cells[i].measure as f64;
        let margin = VOLUME_MARGIN * capacity.max(1.0);
        if !volume.is_finite() || volume < -margin || volume > capacity + margin {
            fields.fault = Some(NumericalFault {
                stage: "cellwise-remap-commit".into(),
                index: i as u32,
                observed: volume as f32,
                expected: capacity as f32,
            });
            return Ok(receipt);
        }
    }
    for (i, &volume) in next.iter().enumerate() {
        fields.density[i] = (volume / graph.cells[i].measure as f64) as f32;
        fields.gamma[i] = 1.0;
    }
    commit(dt, fields)?;
    receipt.material_committed = true;
    Ok(receipt)
}

fn publish_tracer_differential_diagnostics(
    graph: &Graph,
    tracer: &FaceConsistentVelocity2d,
    receiver_band: &[bool],
    receipt: &mut CellwiseRemapReceipt,
) -> Result<(), ValidationError> {
    let epsilon = 1.0e-4;
    for (cell, geometry) in graph.cells.iter().enumerate() {
        if !receiver_band[cell] {
            continue;
        }
        for y in geometry.minimum[1] as usize..geometry.maximum[1] as usize {
            for x in geometry.minimum[0] as usize..geometry.maximum[0] as usize {
                for oy in [0.25, 0.5, 0.75] {
                    for ox in [0.25, 0.5, 0.75] {
                        let point = [x as f64 + ox, y as f64 + oy];
                        let xm = tracer.sample(graph, [point[0] - epsilon, point[1]])?;
                        let xp = tracer.sample(graph, [point[0] + epsilon, point[1]])?;
                        let ym = tracer.sample(graph, [point[0], point[1] - epsilon])?;
                        let yp = tracer.sample(graph, [point[0], point[1] + epsilon])?;
                        let du_dx = (xp[0] - xm[0]) / (2.0 * epsilon);
                        let dv_dx = (xp[1] - xm[1]) / (2.0 * epsilon);
                        let du_dy = (yp[0] - ym[0]) / (2.0 * epsilon);
                        let dv_dy = (yp[1] - ym[1]) / (2.0 * epsilon);
                        receipt.tracing_max_abs_finite_difference_divergence = receipt
                            .tracing_max_abs_finite_difference_divergence
                            .max((du_dx + dv_dy).abs());
                        let gradient =
                            (du_dx * du_dx + dv_dx * dv_dx + du_dy * du_dy + dv_dy * dv_dy).sqrt();
                        if gradient > receipt.tracing_max_velocity_gradient {
                            receipt.tracing_max_velocity_gradient = gradient;
                            receipt.tracing_max_velocity_gradient_point = point;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_inputs(
    graph: &Graph,
    fields: &Fields,
    dt: f32,
    options: CellwiseRemapOptions,
) -> Result<(), ValidationError> {
    graph.validate()?;
    fields.validate_for(graph)?;
    if graph.dimension != 2 {
        return Err(ValidationError(
            "cellwise remap is a gated 2-D experiment".into(),
        ));
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(ValidationError(
            "cellwise remap dt must be finite and positive".into(),
        ));
    }
    if options.edge_samples == 0 || options.edge_samples > 64 {
        return Err(ValidationError(
            "cellwise remap edgeSamples must be in 1..=64".into(),
        ));
    }
    if options.trace_segments == 0 || options.trace_segments > 128 {
        return Err(ValidationError(
            "cellwise remap traceSegments must be in 1..=128".into(),
        ));
    }
    for (i, cell) in graph.cells.iter().enumerate() {
        let cap = fields.capacity[i] as f64;
        if !cap.is_finite() || cap < -CAPACITY_EPS || cap > 1.0 + CAPACITY_EPS {
            return Err(ValidationError(format!(
                "cellwise remap invalid capacity at cell {}",
                cell.id
            )));
        }
        if cap != 1.0 {
            return Err(ValidationError(format!(
                "cellwise remap M7: cut cell {} is not implemented",
                cell.id
            )));
        }
        let volume = fields.density[i] as f64 * cell.measure as f64;
        let capacity = cap * cell.measure as f64;
        let margin = VOLUME_MARGIN * capacity.max(1.0);
        if !volume.is_finite() || volume < -margin || volume > capacity + margin {
            return Err(ValidationError(format!(
                "cellwise remap invalid volume at cell {}",
                cell.id
            )));
        }
    }
    if fields.cell_velocity.iter().any(|v| !v.is_finite())
        || fields.face_velocity.iter().any(|v| !v.is_finite())
        || fields
            .subface_compatibility_rate
            .iter()
            .any(|v| !v.is_finite())
    {
        return Err(ValidationError(
            "cellwise remap velocity field contains a non-finite value".into(),
        ));
    }
    Ok(())
}

fn trace_lattice_points(
    graph: &Graph,
    fields: &Fields,
    tracer: &FaceConsistentVelocity2d,
    dt: f64,
    trace_segments: usize,
    fixed_edges: &[Vec<Point>],
    traced: &mut BTreeMap<(u64, u64), TracedPoint>,
    receipt: &mut CellwiseRemapReceipt,
) -> Result<(), ValidationError> {
    let mut points = BTreeMap::new();
    for fixed in fixed_edges {
        for &point in fixed {
            points.entry(point_key(point)).or_insert(point);
        }
    }
    for (&key, &point) in &points {
        if traced.contains_key(&key) {
            continue;
        }
        let q = trace_rk4(
            graph,
            fields,
            tracer,
            point,
            dt,
            trace_segments,
            receipt,
        )?;
        if q.path.iter().flatten().any(|v| !v.is_finite()) {
            return Err(ValidationError(
                "cellwise remap produced a non-finite characteristic".into(),
            ));
        }
        traced.insert(key, q);
    }
    receipt.traces = traced.len();
    Ok(())
}

fn fixed_edge(face: &crate::types::Subface, samples: usize) -> Vec<Point> {
    let axis = face.axis as usize;
    let tangent = 1 - axis;
    let mut first = [face.center[0] as f64, face.center[1] as f64];
    let length = face.measure as f64;
    let intervals = ((length * samples as f64).round() as usize).max(1);
    first[tangent] -= 0.5 * length;
    (0..=intervals)
        .map(|i| {
            let mut p = first;
            p[tangent] += length * i as f64 / intervals as f64;
            p
        })
        .collect()
}

fn point_key(point: Point) -> (u64, u64) {
    (point[0].to_bits(), point[1].to_bits())
}

fn trace_rk4(
    graph: &Graph,
    fields: &Fields,
    tracer: &FaceConsistentVelocity2d,
    point: Point,
    dt: f64,
    segments: usize,
    receipt: &mut CellwiseRemapReceipt,
) -> Result<TracedPoint, ValidationError> {
    let mut p = point;
    let mut path = Vec::with_capacity(segments + 1);
    let mut signed_area_integral = 0.0;
    let mut support_extrapolation_samples = 0;
    let mut constrained_segments = 0;
    let mut rt0_region_transitions = 0;
    path.push(point);
    let h = -dt / segments as f64;
    for _ in 0..segments {
        let sample1 = sample_tracer(tracer, graph, p, receipt)?;
        support_extrapolation_samples += usize::from(sample1.support_fallback);
        let k1 = sample1.velocity;
        let (p2, constrained2) = projected_stage(graph, fields, p, madd(p, k1, 0.5 * h), receipt);
        let sample2 = sample_tracer(tracer, graph, p2, receipt)?;
        support_extrapolation_samples += usize::from(sample2.support_fallback);
        let k2 = sample2.velocity;
        let (p3, constrained3) = projected_stage(graph, fields, p, madd(p, k2, 0.5 * h), receipt);
        let sample3 = sample_tracer(tracer, graph, p3, receipt)?;
        support_extrapolation_samples += usize::from(sample3.support_fallback);
        let k3 = sample3.velocity;
        let (p4, constrained4) = projected_stage(graph, fields, p, madd(p, k3, h), receipt);
        let sample4 = sample_tracer(tracer, graph, p4, receipt)?;
        support_extrapolation_samples += usize::from(sample4.support_fallback);
        let k4 = sample4.velocity;
        receipt.rk_evaluations += 4;
        let candidate = [
            p[0] + h * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            p[1] + h * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        ];
        let (next, constrained_final) = projected_stage(graph, fields, p, candidate, receipt);
        // Endpoint classification is diagnostic only and does not add an RK
        // evaluation or a support-extrapolation sample to the work counters.
        let next_region = tracer.sample_with_diagnostic(graph, next)?.region;
        receipt.rt0_region_diagnostic_samples += 1;
        let sampled_regions = [
            sample1.region,
            sample2.region,
            sample3.region,
            sample4.region,
            next_region,
        ];
        let transitions = sampled_regions
            .windows(2)
            .filter(|pair| pair[0] != pair[1])
            .count();
        rt0_region_transitions += transitions;
        receipt.rt0_region_transitions += transitions;
        if !(constrained2 || constrained3 || constrained4 || constrained_final) {
            signed_area_integral += h
                * (pathline_area_rate(p, k1)
                    + 2.0 * pathline_area_rate(p2, k2)
                    + 2.0 * pathline_area_rate(p3, k3)
                    + pathline_area_rate(p4, k4))
                / 6.0;
        } else {
            signed_area_integral += 0.5 * cross(p, next);
            receipt.constrained_pathline_segments += 1;
            constrained_segments += 1;
        }
        p = next;
        path.push(p);
    }
    if constrained_segments == 0 {
        if let (Some(initial), Some(final_value)) = (
            tracer.streamfunction_scalar(point)?,
            tracer.streamfunction_scalar(p)?,
        ) {
            let drift = (final_value - initial).abs();
            receipt.tracing_max_streamfunction_absolute_drift = receipt
                .tracing_max_streamfunction_absolute_drift
                .max(drift);
            receipt.tracing_max_streamfunction_relative_drift = receipt
                .tracing_max_streamfunction_relative_drift
                .max(drift / receipt.streamfunction_range.max(f64::MIN_POSITIVE));
        }
    }
    Ok(TracedPoint {
        path,
        signed_area_integral,
        support_extrapolation_samples,
        constrained_segments,
        rt0_region_transitions,
    })
}

fn pathline_area_rate(point: Point, velocity: Point) -> f64 {
    0.5 * cross(point, velocity)
}

fn sample_tracer(
    tracer: &FaceConsistentVelocity2d,
    graph: &Graph,
    point: Point,
    receipt: &mut CellwiseRemapReceipt,
) -> Result<FaceVelocitySample2d, ValidationError> {
    let sample = tracer.sample_with_diagnostic(graph, point)?;
    if sample.support_fallback {
        receipt.support_extrapolation_samples += 1;
    }
    if sample.velocity_model_fallback {
        receipt.trace_velocity_fallback_samples += 1;
    }
    Ok(sample)
}

fn madd(point: Point, velocity: Point, scale: f64) -> Point {
    [
        point[0] + scale * velocity[0],
        point[1] + scale * velocity[1],
    ]
}

fn projected_stage(
    graph: &Graph,
    fields: &Fields,
    start: Point,
    candidate: Point,
    receipt: &mut CellwiseRemapReceipt,
) -> (Point, bool) {
    let bounded = [
        candidate[0].clamp(0.0, graph.dimensions[0] as f64),
        candidate[1].clamp(0.0, graph.dimensions[1] as f64),
    ];
    let domain_clamped = candidate != bounded;
    let bounded_f32 = [bounded[0] as f32, bounded[1] as f32];
    let clipped = clip_segment(
        graph,
        fields,
        [start[0] as f32, start[1] as f32],
        bounded_f32,
    );
    let segment_clamped = clipped != bounded_f32;
    let same_voxel = bounded
        .iter()
        .zip(bounded_f32)
        .all(|(&value, rounded)| value.floor() == f64::from(rounded).floor());
    let result = if segment_clamped || !same_voxel {
        [clipped[0] as f64, clipped[1] as f64]
    } else {
        bounded
    };
    let constrained = domain_clamped || segment_clamped;
    if constrained {
        receipt.clips += 1;
    }
    (result, constrained)
}

fn build_geometry(
    graph: &Graph,
    fields: &Fields,
    dt: f64,
    fixed_edges: &[Vec<Point>],
    traced_points: &BTreeMap<(u64, u64), TracedPoint>,
    rates: &[f64],
    receiver_band: &[bool],
) -> Result<Geometry, ValidationError> {
    let mut chains = Vec::with_capacity(graph.subfaces.len());
    let mut uncorrected_chains = Vec::with_capacity(graph.subfaces.len());
    let mut fixed_chains = Vec::with_capacity(graph.subfaces.len());
    let mut raw_swept_rates = Vec::with_capacity(graph.subfaces.len());
    let mut correction_deltas_over_h = Vec::with_capacity(graph.subfaces.len());
    let mut correction_coefficients = Vec::with_capacity(graph.subfaces.len());
    let mut chain_points = 0;
    let mut interpolated_correction_points = 0;
    for (face, source_fixed) in graph.subfaces.iter().zip(fixed_edges) {
        let mut fixed = source_fixed.clone();
        let mut traced: Vec<_> = fixed
            .iter()
            .map(|&p| traced_points[&point_key(p)].end())
            .collect();
        let first_path = &traced_points[&point_key(fixed[0])];
        let last_path = &traced_points[&point_key(*fixed.last().unwrap())];
        if insert_correction_midpoint(&mut fixed, &mut traced) {
            interpolated_correction_points += 1;
        }
        chain_points += traced.len();
        uncorrected_chains.push(EdgeChain {
            axis: face.axis as usize,
            traced: traced.clone(),
        });
        let canonical_sign = if face.axis == 0 { 1.0 } else { -1.0 };
        raw_swept_rates
            .push(canonical_sign * swept_area(&fixed, &traced, first_path, last_path) / dt);
        fixed_chains.push(fixed);
    }
    let measured_swept_rates = raw_swept_rates.clone();
    // Correct only faces that can bound material receivers. Remote dry cells
    // still use the global flow map, but retain their measured swept rates;
    // forcing irrelevant dry edges to extension targets can fold harmless
    // geometry. This mixed rate vector is also the exact area-identity RHS.
    for (i, face) in graph.subfaces.iter().enumerate() {
        let touches_receiver = [face.negative_cell, face.positive_cell]
            .into_iter()
            .filter(|&cell| cell >= 0)
            .any(|cell| receiver_band[cell as usize]);
        if touches_receiver {
            raw_swept_rates[i] = rates[i];
        }
    }
    for (i, face) in graph.subfaces.iter().enumerate() {
        let fixed = &fixed_chains[i];
        let mut traced = uncorrected_chains[i].traced.clone();
        let first_path = &traced_points[&point_key(fixed[0])];
        let last_path = &traced_points[&point_key(*fixed.last().unwrap())];
        let canonical_sign = if face.axis == 0 { 1.0 } else { -1.0 };
        let delta = correct_chain_area(
            fixed,
            &mut traced,
            first_path,
            last_path,
            canonical_sign * dt * raw_swept_rates[i],
        )
        .map_err(|error| {
            ValidationError(format!(
                "cellwise remap face {} geometry correction failed: {}",
                face.id, error.0
            ))
        })?;
        let target_area = canonical_sign * dt * raw_swept_rates[i];
        let measured_area = canonical_sign * dt * measured_swept_rates[i];
        correction_coefficients.push(if delta.abs() > GEOMETRY_EPS {
            (target_area - measured_area) / delta
        } else {
            0.0
        });
        correction_deltas_over_h.push(delta.abs());
        chains.push(EdgeChain {
            axis: face.axis as usize,
            traced,
        });
    }

    let uncorrected_polygons = cell_polygons(graph, &uncorrected_chains)?;
    let pre_correction_fold_mask = uncorrected_polygons
        .iter()
        .map(|polygon| polygon_self_intersects(polygon))
        .collect();
    let polygons = cell_polygons(graph, &chains)?;
    let corrected_fold_mask = polygons
        .iter()
        .map(|polygon| polygon_self_intersects(polygon))
        .collect();
    let divergence = cell_divergence(graph, &raw_swept_rates);
    let mut identity_error = 0.0_f64;
    let mut total_area = 0.0;
    let mut open_area = 0.0;
    for (i, polygon) in polygons.iter().enumerate() {
        let area = signed_area(polygon);
        let capacity = fields.capacity[i] as f64 * graph.cells[i].measure as f64;
        let expected = capacity - dt * divergence[i];
        identity_error = identity_error.max((area - expected).abs());
        total_area += area;
        open_area += capacity;
    }
    let scale = open_area.abs().max(1.0);
    Ok(Geometry {
        uncorrected_chains,
        corrected_chains: chains,
        uncorrected_polygons,
        polygons,
        divergence,
        identity_error,
        tiling_error: (total_area - open_area).abs() / scale,
        pre_correction_fold_mask,
        corrected_fold_mask,
        correction_deltas_over_h,
        raw_swept_rates: measured_swept_rates,
        correction_coefficients,
        chain_points,
        interpolated_correction_points,
    })
}

fn swept_area(
    fixed: &[Point],
    traced: &[Point],
    first_path: &TracedPoint,
    last_path: &TracedPoint,
) -> f64 {
    0.5 * polyline_cross_sum(fixed) + last_path.signed_area_integral
        - 0.5 * polyline_cross_sum(traced)
        - first_path.signed_area_integral
}

fn polyline_cross_sum(points: &[Point]) -> f64 {
    points.windows(2).map(|pair| cross(pair[0], pair[1])).sum()
}

fn cross(a: Point, b: Point) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn insert_correction_midpoint(fixed: &mut Vec<Point>, traced: &mut Vec<Point>) -> bool {
    if fixed.len() != 2 || traced.len() != 2 {
        return false;
    }
    fixed.insert(
        1,
        [
            0.5 * (fixed[0][0] + fixed[1][0]),
            0.5 * (fixed[0][1] + fixed[1][1]),
        ],
    );
    traced.insert(
        1,
        [
            0.5 * (traced[0][0] + traced[1][0]),
            0.5 * (traced[0][1] + traced[1][1]),
        ],
    );
    true
}

fn correct_chain_area(
    fixed: &[Point],
    traced: &mut [Point],
    first_path: &TracedPoint,
    last_path: &TracedPoint,
    target: f64,
) -> Result<f64, ValidationError> {
    if fixed.len() != traced.len() || fixed.len() < 3 {
        return Err(ValidationError(
            "cellwise remap edge chain has no correctable interior point".into(),
        ));
    }
    let tangent = [
        fixed[fixed.len() - 1][0] - fixed[0][0],
        fixed[fixed.len() - 1][1] - fixed[0][1],
    ];
    let length = tangent[0].hypot(tangent[1]);
    if length <= GEOMETRY_EPS {
        return Err(ValidationError(
            "cellwise remap encountered a zero-length subface".into(),
        ));
    }
    // Move the chain along its traced-chord normal. Using the fixed-face
    // normal becomes singular when the flow rotates an edge close to ninety
    // degrees: the correction direction is then almost parallel to the traced
    // chord and an ordinary area defect demands an unbounded displacement.
    // Endpoints remain fixed, so adjacent cells still share exactly one chain.
    let traced_tangent = [
        traced[traced.len() - 1][0] - traced[0][0],
        traced[traced.len() - 1][1] - traced[0][1],
    ];
    let traced_length = traced_tangent[0].hypot(traced_tangent[1]);
    let normal = if traced_length > GEOMETRY_EPS {
        [
            -traced_tangent[1] / traced_length,
            traced_tangent[0] / traced_length,
        ]
    } else {
        [-tangent[1] / length, tangent[0] / length]
    };
    let original = swept_area(fixed, traced, first_path, last_path);
    let tangent_squared = tangent[0] * tangent[0] + tangent[1] * tangent[1];
    let source_parameter = |point: Point| {
        ((point[0] - fixed[0][0]) * tangent[0]
            + (point[1] - fixed[0][1]) * tangent[1])
            / tangent_squared
    };
    let mut unit = traced.to_vec();
    for (i, point) in unit[1..traced.len() - 1].iter_mut().enumerate() {
        let t = source_parameter(fixed[i + 1]);
        let weight = 4.0 * t * (1.0 - t);
        point[0] += weight * normal[0];
        point[1] += weight * normal[1];
    }
    let coefficient = swept_area(fixed, &unit, first_path, last_path) - original;
    if coefficient.abs() <= GEOMETRY_EPS {
        if (target - original).abs() <= GEOMETRY_EPS {
            return Ok(0.0);
        }
        return Err(ValidationError(
            format!(
                "cellwise remap edge area correction is singular: target={target:.17e} original={original:.17e} coefficient={coefficient:.17e} fixed=({:.17e},{:.17e})->({:.17e},{:.17e}) traced=({:.17e},{:.17e})->({:.17e},{:.17e})",
                fixed[0][0], fixed[0][1], fixed[fixed.len() - 1][0], fixed[fixed.len() - 1][1],
                traced[0][0], traced[0][1], traced[traced.len() - 1][0], traced[traced.len() - 1][1],
            ),
        ));
    }
    let delta = (target - original) / coefficient;
    if !delta.is_finite() {
        return Err(ValidationError(
            "cellwise remap edge area correction is non-finite".into(),
        ));
    }
    for (i, point) in traced[1..fixed.len() - 1].iter_mut().enumerate() {
        let t = source_parameter(fixed[i + 1]);
        let weight = 4.0 * t * (1.0 - t);
        point[0] += delta * weight * normal[0];
        point[1] += delta * weight * normal[1];
    }
    Ok(delta)
}

fn cell_polygons(graph: &Graph, chains: &[EdgeChain]) -> Result<Vec<Vec<Point>>, ValidationError> {
    let mut faces_by_cell = vec![Vec::<(usize, i8)>::new(); graph.cells.len()];
    for face in &graph.subfaces {
        if face.negative_cell >= 0 {
            faces_by_cell[face.negative_cell as usize].push((face.id as usize, -1));
        }
        if face.positive_cell >= 0 {
            faces_by_cell[face.positive_cell as usize].push((face.id as usize, 1));
        }
    }
    faces_by_cell
        .iter_mut()
        .enumerate()
        .map(|(cell, faces)| {
            faces.sort_by(|&(a, ao), &(b, bo)| {
                boundary_order(graph, cell, a, ao)
                    .partial_cmp(&boundary_order(graph, cell, b, bo))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            let mut polygon = Vec::new();
            for &(face, orientation) in faces.iter() {
                let chain = &chains[face];
                let ccw_forward = match (chain.axis, orientation) {
                    (0, -1) | (1, 1) => true,
                    (0, 1) | (1, -1) => false,
                    _ => unreachable!(),
                };
                let iterator: Box<dyn Iterator<Item = Point>> = if ccw_forward {
                    Box::new(chain.traced.iter().copied())
                } else {
                    Box::new(chain.traced.iter().rev().copied())
                };
                for point in iterator {
                    if polygon
                        .last()
                        .is_none_or(|last| distance_squared(*last, point) > 1e-24)
                    {
                        polygon.push(point);
                    }
                }
            }
            if polygon.len() > 1 && distance_squared(polygon[0], *polygon.last().unwrap()) <= 1e-24
            {
                polygon.pop();
            }
            if polygon.len() < 3 {
                Err(ValidationError(format!(
                    "cellwise remap could not assemble cell {cell} boundary"
                )))
            } else {
                Ok(polygon)
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
struct ChainSegment {
    face: usize,
    interval: usize,
    first: Point,
    second: Point,
}

/// Return the canonical source intervals whose straight mapped chords create
/// a proper crossing, together with their immediate boundary neighbours. The
/// chains are stored once per physical subface and merely reversed for the
/// adjacent cell, so splitting the returned interval keeps both cell polygons
/// conforming without tracing either side independently.
fn crossing_chain_intervals(
    graph: &Graph,
    chains: &[EdgeChain],
    refine_cells: &[bool],
) -> BTreeSet<(usize, usize)> {
    let mut faces_by_cell = vec![Vec::<(usize, i8)>::new(); graph.cells.len()];
    for face in &graph.subfaces {
        if face.negative_cell >= 0 {
            faces_by_cell[face.negative_cell as usize].push((face.id as usize, -1));
        }
        if face.positive_cell >= 0 {
            faces_by_cell[face.positive_cell as usize].push((face.id as usize, 1));
        }
    }
    let mut marked = BTreeSet::new();
    for (cell, faces) in faces_by_cell.iter_mut().enumerate() {
        if !refine_cells[cell] {
            continue;
        }
        faces.sort_by(|&(a, ao), &(b, bo)| {
            boundary_order(graph, cell, a, ao)
                .partial_cmp(&boundary_order(graph, cell, b, bo))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut boundary = Vec::new();
        for &(face, orientation) in faces.iter() {
            let chain = &chains[face];
            let forward = matches!((chain.axis, orientation), (0, -1) | (1, 1));
            if forward {
                for interval in 0..chain.traced.len() - 1 {
                    boundary.push(ChainSegment {
                        face,
                        interval,
                        first: chain.traced[interval],
                        second: chain.traced[interval + 1],
                    });
                }
            } else {
                for interval in (0..chain.traced.len() - 1).rev() {
                    boundary.push(ChainSegment {
                        face,
                        interval,
                        first: chain.traced[interval + 1],
                        second: chain.traced[interval],
                    });
                }
            }
        }
        for i in 0..boundary.len() {
            for j in i + 2..boundary.len() {
                if i == 0 && j + 1 == boundary.len() {
                    continue;
                }
                if !segments_cross(
                    boundary[i].first,
                    boundary[i].second,
                    boundary[j].first,
                    boundary[j].second,
                ) {
                    continue;
                }
                for at in [i, j] {
                    for neighbour in [
                        (at + boundary.len() - 1) % boundary.len(),
                        at,
                        (at + 1) % boundary.len(),
                    ] {
                        let segment = boundary[neighbour];
                        marked.insert((segment.face, segment.interval));
                    }
                }
            }
        }
    }
    marked
}

/// A corrected-only fold means that the characteristic chain itself is
/// simple but its face-flux area correction is not. Refining only the tiny
/// crossing chords cannot change the broad line-integral quadrature error
/// that demanded that correction. Split every interval on the folded cell's
/// incident faces whose measured swept rate still needs a nonzero correction;
/// the next build retraces those shared midpoints once for both neighbours.
fn corrected_only_area_intervals(
    graph: &Graph,
    geometry: &Geometry,
    refine_cells: &[bool],
) -> BTreeSet<(usize, usize)> {
    let mut marked = BTreeSet::new();
    for (cell, &refine) in refine_cells.iter().enumerate() {
        if !refine
            || !geometry.corrected_fold_mask[cell]
            || geometry.pre_correction_fold_mask[cell]
        {
            continue;
        }
        for face in &graph.subfaces {
            let face_index = face.id as usize;
            if (face.negative_cell != cell as i32 && face.positive_cell != cell as i32)
                || geometry.correction_deltas_over_h[face_index] <= GEOMETRY_EPS
            {
                continue;
            }
            for interval in 0..geometry.corrected_chains[face_index]
                .traced
                .len()
                .saturating_sub(1)
            {
                marked.insert((face_index, interval));
            }
        }
    }
    marked
}

fn refine_crossing_intervals(
    fixed_edges: &mut [Vec<Point>],
    intervals: &BTreeSet<(usize, usize)>,
) -> Result<usize, ValidationError> {
    const MAX_INTERVALS_PER_SUBFACE: usize = 4096;
    let mut by_face = BTreeMap::<usize, Vec<usize>>::new();
    for &(face, interval) in intervals {
        by_face.entry(face).or_default().push(interval);
    }
    let mut inserted = 0;
    for (face, mut selected) in by_face {
        selected.sort_unstable();
        selected.dedup();
        for interval in selected.into_iter().rev() {
            let points = &mut fixed_edges[face];
            if points.len() - 1 >= MAX_INTERVALS_PER_SUBFACE || interval + 1 >= points.len() {
                continue;
            }
            let first = points[interval];
            let second = points[interval + 1];
            let midpoint = [
                0.5 * (first[0] + second[0]),
                0.5 * (first[1] + second[1]),
            ];
            if midpoint == first || midpoint == second {
                return Err(ValidationError(format!(
                    "cellwise remap face {face} crossing interval is below f64 resolution"
                )));
            }
            points.insert(interval + 1, midpoint);
            inserted += 1;
        }
    }
    Ok(inserted)
}

fn boundary_order(graph: &Graph, cell: usize, face: usize, orientation: i8) -> f64 {
    let subface = &graph.subfaces[face];
    let c = &graph.cells[cell];
    match (subface.axis, orientation) {
        (1, 1) => subface.center[0] as f64 - c.minimum[0] as f64,
        (0, -1) => c.widths[0] as f64 + subface.center[1] as f64 - c.minimum[1] as f64,
        (1, -1) => {
            c.widths[0] as f64 + c.widths[1] as f64 + c.maximum[0] as f64 - subface.center[0] as f64
        }
        (0, 1) => {
            2.0 * c.widths[0] as f64 + c.widths[1] as f64 + c.maximum[1] as f64
                - subface.center[1] as f64
        }
        _ => f64::INFINITY,
    }
}

fn cell_divergence(graph: &Graph, rates: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; graph.cells.len()];
    for (i, face) in graph.subfaces.iter().enumerate() {
        if face.negative_cell >= 0 {
            result[face.negative_cell as usize] += rates[i];
        }
        if face.positive_cell >= 0 {
            result[face.positive_cell as usize] -= rates[i];
        }
    }
    result
}

/// Conservative, non-circular receiver discovery for the pre-trace closure.
///
/// RT0 component values are bounded by their incident face-normal values, so
/// the largest component rate gives an axis-aligned whole-step travel bound.
/// Dilating finest-lattice liquid occupancy crosses sparse support holes and
/// avoids treating an adaptive graph hop as a unit of physical distance.
fn conservative_pretrace_receiver_band(
    graph: &Graph,
    fields: &Fields,
    rates: &[f64],
    dt: f64,
) -> Result<Vec<bool>, ValidationError> {
    let width = graph.dimensions[0] as usize;
    let height = graph.dimensions[1] as usize;
    if width == 0
        || height == 0
        || width as f32 != graph.dimensions[0]
        || height as f32 != graph.dimensions[1]
    {
        return Err(ValidationError(
            "cellwise remap receiver band requires integral 2-D dimensions".into(),
        ));
    }
    let mut minimum_velocity = [0.0_f64; 2];
    let mut maximum_velocity = [0.0_f64; 2];
    for (face, &rate) in graph.subfaces.iter().zip(rates) {
        let velocity = rate / face.measure.max(1.0e-8) as f64;
        let axis = face.axis as usize;
        minimum_velocity[axis] = minimum_velocity[axis].min(velocity);
        maximum_velocity[axis] = maximum_velocity[axis].max(velocity);
    }
    // A receiver at x traces backward into
    // [x-dt*max(u), x-dt*min(u)]. Retain those signed one-frame reach bounds
    // instead of symmetric |u| padding: symmetric padding can include a
    // coarse wall cell that material cannot reach, and its zero wall flux then
    // bends the private air extension before physical contact.
    let minimum_displacement = [
        (dt * minimum_velocity[0]).floor() as i64,
        (dt * minimum_velocity[1]).floor() as i64,
    ];
    let maximum_displacement = [
        (dt * maximum_velocity[0]).ceil() as i64,
        (dt * maximum_velocity[1]).ceil() as i64,
    ];
    let stride = width
        .checked_add(1)
        .ok_or_else(|| ValidationError("cellwise remap receiver lattice overflow".into()))?;
    let rows = height
        .checked_add(1)
        .ok_or_else(|| ValidationError("cellwise remap receiver lattice overflow".into()))?;
    let lattice_len = stride
        .checked_mul(rows)
        .ok_or_else(|| ValidationError("cellwise remap receiver lattice overflow".into()))?;
    let mut liquid = vec![0_usize; lattice_len];
    for (cell, &density) in graph.cells.iter().zip(&fields.density) {
        if density <= 0.0 {
            continue;
        }
        let x0 = (cell.minimum[0].floor() as usize).min(width);
        let y0 = (cell.minimum[1].floor() as usize).min(height);
        let x1 = (cell.maximum[0].ceil() as usize).min(width);
        let y1 = (cell.maximum[1].ceil() as usize).min(height);
        for y in y0..y1 {
            for x in x0..x1 {
                liquid[(x + 1) + stride * (y + 1)] = 1;
            }
        }
    }
    for y in 1..=height {
        for x in 1..=width {
            let at = x + stride * y;
            liquid[at] += liquid[at - 1] + liquid[at - stride] - liquid[at - stride - 1];
        }
    }
    let rectangle_sum = |x0: usize, y0: usize, x1: usize, y1: usize| {
        liquid[x1 + stride * y1] + liquid[x0 + stride * y0]
            - liquid[x0 + stride * y1]
            - liquid[x1 + stride * y0]
    };
    Ok(graph
        .cells
        .iter()
        .map(|cell| {
            let clamp = |value: i64, upper: usize| value.clamp(0, upper as i64) as usize;
            let x0 = clamp(
                cell.minimum[0].floor() as i64 - maximum_displacement[0],
                width,
            );
            let y0 = clamp(
                cell.minimum[1].floor() as i64 - maximum_displacement[1],
                height,
            );
            let x1 = clamp(
                cell.maximum[0].ceil() as i64 - minimum_displacement[0],
                width,
            );
            let y1 = clamp(
                cell.maximum[1].ceil() as i64 - minimum_displacement[1],
                height,
            );
            rectangle_sum(x0, y0, x1, y1) != 0
        })
        .collect())
}

fn publish_geometry_metrics(
    graph: &Graph,
    fields: &Fields,
    geometry: &Geometry,
    raw_receivers: &[bool],
    corrected_receivers: &[bool],
    traced: &BTreeMap<(u64, u64), TracedPoint>,
    rates: &[f64],
    fixed_edges: &[Vec<Point>],
    receipt: &mut CellwiseRemapReceipt,
) {
    receipt.chain_points = geometry.chain_points;
    receipt.interpolated_correction_points = geometry.interpolated_correction_points;
    receipt.pre_correction_self_intersections = geometry
        .pre_correction_fold_mask
        .iter()
        .filter(|&&v| v)
        .count();
    receipt.self_intersections = geometry.corrected_fold_mask.iter().filter(|&&v| v).count();
    receipt.pre_correction_liquid_receiver_folds = geometry
        .pre_correction_fold_mask
        .iter()
        .zip(raw_receivers)
        .filter(|&(folded, receiver)| *folded && *receiver)
        .count();
    receipt.corrected_liquid_receiver_folds = geometry
        .corrected_fold_mask
        .iter()
        .zip(corrected_receivers)
        .filter(|&(folded, receiver)| *folded && *receiver)
        .count();
    receipt.pre_correction_dry_folds = receipt
        .pre_correction_self_intersections
        .saturating_sub(receipt.pre_correction_liquid_receiver_folds);
    receipt.corrected_dry_folds = receipt
        .self_intersections
        .saturating_sub(receipt.corrected_liquid_receiver_folds);
    receipt.pre_correction_liquid_fold_locations =
        classify_folds(graph, &geometry.pre_correction_fold_mask, raw_receivers);
    receipt.corrected_liquid_fold_locations =
        classify_folds(graph, &geometry.corrected_fold_mask, corrected_receivers);
    receipt.pre_correction_fold_details = fold_details_by_location(
        graph,
        fields,
        &geometry.uncorrected_polygons,
        &geometry.pre_correction_fold_mask,
        raw_receivers,
        &geometry.divergence,
        &geometry.correction_deltas_over_h,
        &geometry.raw_swept_rates,
        &geometry.correction_coefficients,
        traced,
        rates,
        fixed_edges,
    );
    receipt.corrected_fold_details = fold_details_by_location(
        graph,
        fields,
        &geometry.polygons,
        &geometry.corrected_fold_mask,
        corrected_receivers,
        &geometry.divergence,
        &geometry.correction_deltas_over_h,
        &geometry.raw_swept_rates,
        &geometry.correction_coefficients,
        traced,
        rates,
        fixed_edges,
    );

    receipt.correction_count = geometry.correction_deltas_over_h.len();
    receipt.correction_delta_over_h_sum = 0.0;
    receipt.correction_delta_over_h_sum_squares = 0.0;
    receipt.max_correction_delta_over_h = 0.0;
    receipt.correction_delta_over_h_histogram = [0; 11];
    for &value in &geometry.correction_deltas_over_h {
        receipt.correction_delta_over_h_sum += value;
        receipt.correction_delta_over_h_sum_squares += value * value;
        receipt.max_correction_delta_over_h = receipt.max_correction_delta_over_h.max(value);
        let bucket = CORRECTION_DELTA_OVER_H_HISTOGRAM_BOUNDS
            .iter()
            .position(|&bound| value <= bound)
            .unwrap_or(CORRECTION_DELTA_OVER_H_HISTOGRAM_BOUNDS.len());
        receipt.correction_delta_over_h_histogram[bucket] += 1;
    }
    let mut receiver_faces = BTreeSet::new();
    for face in &graph.subfaces {
        let touches_receiver = (face.negative_cell >= 0
            && corrected_receivers[face.negative_cell as usize])
            || (face.positive_cell >= 0 && corrected_receivers[face.positive_cell as usize]);
        if touches_receiver {
            receiver_faces.insert(face.id as usize);
        }
    }
    receipt.receiver_correction_count = receiver_faces.len();
    receipt.receiver_correction_delta_over_h_sum = 0.0;
    receipt.receiver_correction_delta_over_h_sum_squares = 0.0;
    receipt.max_receiver_correction_delta_over_h = 0.0;
    for face in receiver_faces {
        let value = geometry.correction_deltas_over_h[face];
        receipt.receiver_correction_delta_over_h_sum += value;
        receipt.receiver_correction_delta_over_h_sum_squares += value * value;
        receipt.max_receiver_correction_delta_over_h =
            receipt.max_receiver_correction_delta_over_h.max(value);
    }
}

/// Fold-location adjacency proxies are exclusive to make frame reductions
/// mergeable. A cell touching several categories is assigned wall, then seam,
/// then represented support boundary, then interior. This is not causal trace
/// provenance; `support_extrapolation_samples` separately records actual
/// samples outside represented support but is frame-global.
fn classify_folds(graph: &Graph, folded: &[bool], receivers: &[bool]) -> FoldLocations {
    let mut result = FoldLocations::default();
    for (cell, (&is_folded, &receiver)) in folded.iter().zip(receivers).enumerate() {
        if !is_folded || !receiver {
            continue;
        }
        match fold_proxy(graph, cell) {
            FoldProxy::Wall => result.wall += 1,
            FoldProxy::Seam => result.seam += 1,
            FoldProxy::SupportBoundary => result.support_boundary += 1,
            FoldProxy::Interior => result.interior += 1,
        }
    }
    result
}

#[derive(Clone, Copy)]
enum FoldProxy {
    Wall,
    Seam,
    SupportBoundary,
    Interior,
}

fn fold_proxy(graph: &Graph, cell: usize) -> FoldProxy {
    let mut seam = false;
    let mut support = false;
    for &row in &graph.incidences[cell] {
        match graph.rows[row as usize].kind {
            crate::RowKind::ClosedWorld => return FoldProxy::Wall,
            crate::RowKind::MixedSeam => seam = true,
            crate::RowKind::SparseAir => support = true,
            _ => {}
        }
    }
    if seam {
        FoldProxy::Seam
    } else if support {
        FoldProxy::SupportBoundary
    } else {
        FoldProxy::Interior
    }
}

#[allow(clippy::too_many_arguments)]
fn fold_details_by_location(
    graph: &Graph,
    fields: &Fields,
    polygons: &[Vec<Point>],
    folded: &[bool],
    receivers: &[bool],
    divergence: &[f64],
    correction_deltas: &[f64],
    raw_swept_rates: &[f64],
    correction_coefficients: &[f64],
    traced: &BTreeMap<(u64, u64), TracedPoint>,
    rates: &[f64],
    fixed_edges: &[Vec<Point>],
) -> FoldDetailsByLocation {
    let mut result = FoldDetailsByLocation::default();
    for cell in 0..graph.cells.len() {
        if !folded[cell] || !receivers[cell] {
            continue;
        }
        let slot = match fold_proxy(graph, cell) {
            FoldProxy::Wall => &mut result.wall,
            FoldProxy::Seam => &mut result.seam,
            FoldProxy::SupportBoundary => &mut result.support_boundary,
            FoldProxy::Interior => &mut result.interior,
        };
        if slot.is_some() {
            continue;
        }
        let incident: Vec<_> = graph
            .subfaces
            .iter()
            .filter(|face| face.negative_cell == cell as i32 || face.positive_cell == cell as i32)
            .collect();
        let mut point_keys = BTreeSet::new();
        for face in &incident {
            for &point in &fixed_edges[face.id as usize] {
                point_keys.insert(point_key(point));
            }
        }
        let mut boundary_support_extrapolation_samples = 0;
        let mut boundary_constrained_pathline_segments = 0;
        let mut boundary_rt0_region_transitions = 0;
        for key in point_keys {
            if let Some(path) = traced.get(&key) {
                boundary_support_extrapolation_samples += path.support_extrapolation_samples;
                boundary_constrained_pathline_segments += path.constrained_segments;
                boundary_rt0_region_transitions += path.rt0_region_transitions;
            }
        }
        *slot = Some(FoldCellDetail {
            cell_id: cell as u32,
            pressure_member: fields.pressure_member.get(cell).copied().unwrap_or(0) != 0,
            polygon: polygons[cell].clone(),
            divergence: divergence[cell],
            capacity: fields.capacity[cell] as f64 * graph.cells[cell].measure as f64,
            incident_subfaces: incident
                .iter()
                .map(|face| FoldSubfaceDetail {
                    subface_id: face.id,
                    rate: rates[face.id as usize],
                    raw_swept_rate: raw_swept_rates[face.id as usize],
                    area_correction_coefficient: correction_coefficients[face.id as usize],
                    correction_delta_over_h: correction_deltas[face.id as usize],
                })
                .collect(),
            boundary_support_extrapolation_samples,
            boundary_constrained_pathline_segments,
            boundary_rt0_region_transitions,
        });
    }
    result
}

fn receiver_mask(
    graph: &Graph,
    fields: &Fields,
    polygons: &[Vec<Point>],
    folded: &[bool],
    receipt: &mut CellwiseRemapReceipt,
) -> Vec<bool> {
    let mut result = vec![false; graph.cells.len()];
    for (i, polygon) in polygons.iter().enumerate() {
        for donor in candidate_donors(graph, polygon) {
            if fields.density[donor] <= 0.0 {
                continue;
            }
            let c = &graph.cells[donor];
            let liquid = donor_liquid_polygon(graph, fields, donor);
            if folded.get(i).copied().unwrap_or(false) {
                if polygon_intersects_region(polygon, &liquid) {
                    result[i] = true;
                    break;
                }
                continue;
            }
            let clipped = clip_rectangle(
                polygon,
                [c.minimum[0] as f64, c.minimum[1] as f64],
                [c.maximum[0] as f64, c.maximum[1] as f64],
            );
            receipt.gather_clips += 1;
            let clipped = if liquid.len() == 4
                && fields.density[donor] as f64 * c.measure as f64
                    >= fields.capacity[donor] as f64 * c.measure as f64
                        - VOLUME_MARGIN * c.measure as f64
            {
                clipped
            } else {
                let normal = [
                    fields.interface_normal[2 * donor] as f64,
                    fields.interface_normal[2 * donor + 1] as f64,
                ];
                let offset = fields.interface_offset[donor] as f64
                    + normal[0] * c.center[0] as f64
                    + normal[1] * c.center[1] as f64;
                receipt.gather_clips += 1;
                clip_half_plane(&clipped, normal, offset)
            };
            if signed_area(&clipped).abs() > GEOMETRY_EPS {
                result[i] = true;
                break;
            }
        }
    }
    result
}

fn donor_liquid_polygon(graph: &Graph, fields: &Fields, donor: usize) -> Vec<Point> {
    let c = &graph.cells[donor];
    let rectangle = vec![
        [c.minimum[0] as f64, c.minimum[1] as f64],
        [c.maximum[0] as f64, c.minimum[1] as f64],
        [c.maximum[0] as f64, c.maximum[1] as f64],
        [c.minimum[0] as f64, c.maximum[1] as f64],
    ];
    let volume = fields.density[donor] as f64 * c.measure as f64;
    let capacity = fields.capacity[donor] as f64 * c.measure as f64;
    if volume >= capacity - VOLUME_MARGIN * capacity.max(1.0) {
        return rectangle;
    }
    let normal = [
        fields.interface_normal[2 * donor] as f64,
        fields.interface_normal[2 * donor + 1] as f64,
    ];
    if normal[0].hypot(normal[1]) <= GEOMETRY_EPS {
        // Missing PLIC is invalid for gather, but the fold gate must remain
        // conservative and treat the entire nonempty donor as potential liquid.
        return rectangle;
    }
    let offset = fields.interface_offset[donor] as f64
        + normal[0] * c.center[0] as f64
        + normal[1] * c.center[1] as f64;
    clip_half_plane(&rectangle, normal, offset)
}

fn convex_hull_can_reach_liquid(graph: &Graph, fields: &Fields, polygon: &[Point]) -> bool {
    let hull = convex_hull(polygon);
    if hull.len() < 3 {
        return false;
    }
    candidate_donors(graph, &hull).into_iter().any(|donor| {
        let volume = fields.density[donor] as f64 * graph.cells[donor].measure as f64;
        volume > GEOMETRY_EPS
            && polygon_intersects_region(&hull, &donor_liquid_polygon(graph, fields, donor))
    })
}

/// Monotone-chain hull used as a conservative envelope for a folded
/// pre-image. Testing the folded polygon itself with parity can miss an
/// even-winding lobe; testing only its bounding box admits unrelated liquid.
fn convex_hull(polygon: &[Point]) -> Vec<Point> {
    let mut points = polygon.to_vec();
    points.sort_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then_with(|| a[1].total_cmp(&b[1]))
    });
    points.dedup();
    if points.len() <= 2 {
        return points;
    }
    let mut lower = Vec::new();
    for &point in &points {
        while lower.len() >= 2 {
            let (determinant, bound) = robust_orient(
                lower[lower.len() - 2],
                lower[lower.len() - 1],
                point,
            );
            if determinant > bound {
                break;
            }
            lower.pop();
        }
        lower.push(point);
    }
    let mut upper = Vec::new();
    for &point in points.iter().rev() {
        while upper.len() >= 2 {
            let (determinant, bound) = robust_orient(
                upper[upper.len() - 2],
                upper[upper.len() - 1],
                point,
            );
            if determinant > bound {
                break;
            }
            upper.pop();
        }
        upper.push(point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn polygon_intersects_region(polygon: &[Point], region: &[Point]) -> bool {
    if polygon.len() < 3 || region.len() < 3 {
        return false;
    }
    if polygon.iter().any(|&point| point_in_polygon(point, region))
        || region.iter().any(|&point| point_in_polygon(point, polygon))
    {
        return true;
    }
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        for j in 0..region.len() {
            let c = region[j];
            let d = region[(j + 1) % region.len()];
            if segments_intersect_inclusive(a, b, c, d) {
                return true;
            }
        }
    }
    false
}

fn point_in_polygon(point: Point, polygon: &[Point]) -> bool {
    let mut inside = false;
    let mut previous = *polygon.last().unwrap();
    for &current in polygon {
        if orient(previous, current, point).abs() <= GEOMETRY_EPS
            && point[0] >= previous[0].min(current[0]) - GEOMETRY_EPS
            && point[0] <= previous[0].max(current[0]) + GEOMETRY_EPS
            && point[1] >= previous[1].min(current[1]) - GEOMETRY_EPS
            && point[1] <= previous[1].max(current[1]) + GEOMETRY_EPS
        {
            return true;
        }
        if (current[1] > point[1]) != (previous[1] > point[1]) {
            let crossing = (previous[0] - current[0]) * (point[1] - current[1])
                / (previous[1] - current[1])
                + current[0];
            if point[0] < crossing {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

fn segments_intersect_inclusive(a: Point, b: Point, c: Point, d: Point) -> bool {
    let o1 = orient(a, b, c);
    let o2 = orient(a, b, d);
    let o3 = orient(c, d, a);
    let o4 = orient(c, d, b);
    if segments_cross(a, b, c, d) {
        return true;
    }
    [(c, o1), (d, o2)].into_iter().any(|(point, orientation)| {
        orientation.abs() <= GEOMETRY_EPS
            && point[0] >= a[0].min(b[0]) - GEOMETRY_EPS
            && point[0] <= a[0].max(b[0]) + GEOMETRY_EPS
            && point[1] >= a[1].min(b[1]) - GEOMETRY_EPS
            && point[1] <= a[1].max(b[1]) + GEOMETRY_EPS
    }) || [(a, o3), (b, o4)].into_iter().any(|(point, orientation)| {
        orientation.abs() <= GEOMETRY_EPS
            && point[0] >= c[0].min(d[0]) - GEOMETRY_EPS
            && point[0] <= c[0].max(d[0]) + GEOMETRY_EPS
            && point[1] >= c[1].min(d[1]) - GEOMETRY_EPS
            && point[1] <= c[1].max(d[1]) + GEOMETRY_EPS
    })
}

fn update_receiver_metrics(
    graph: &Graph,
    fields: &Fields,
    extension_depths: &[u8],
    dt: f64,
    divergence: &[f64],
    rates: &[f64],
    receivers: &[bool],
    receipt: &mut CellwiseRemapReceipt,
) {
    receipt.receivers = receivers.iter().filter(|&&v| v).count();
    receipt.receivers_outside_operator = receivers
        .iter()
        .enumerate()
        .filter(|&(i, &v)| v && fields.pressure_member.get(i).copied().unwrap_or(0) == 0)
        .count();
    receipt.receiver_band_width = 0;
    receipt.receiver_band_unextended = 0;
    for (i, &receiver) in receivers.iter().enumerate() {
        if !receiver || fields.pressure_member.get(i).copied().unwrap_or(0) != 0 {
            continue;
        }
        let depth = extension_depths.get(i).copied().unwrap_or(255);
        if depth == 255 {
            receipt.receiver_band_unextended += 1;
        } else {
            receipt.receiver_band_width = receipt.receiver_band_width.max(depth);
        }
    }
    receipt.max_dt_div_over_capacity = f64::NEG_INFINITY;
    receipt.max_dt_div_over_capacity_outside_operator = f64::NEG_INFINITY;
    receipt.max_compression_ratio = f64::NEG_INFINITY;
    receipt.max_compression_ratio_outside_operator = f64::NEG_INFINITY;
    receipt.max_courant_all = 0.0;
    receipt.max_courant_liquid = 0.0;
    let mut outgoing = vec![0.0_f64; graph.cells.len()];
    for (face, &rate) in graph.subfaces.iter().zip(rates) {
        if face.negative_cell >= 0 {
            outgoing[face.negative_cell as usize] += rate.max(0.0);
        }
        if face.positive_cell >= 0 {
            outgoing[face.positive_cell as usize] += (-rate).max(0.0);
        }
    }
    for (i, cell) in graph.cells.iter().enumerate() {
        let capacity = fields.capacity[i] as f64 * cell.measure as f64;
        if capacity <= GEOMETRY_EPS {
            continue;
        }
        let courant = dt * outgoing[i] / capacity;
        receipt.max_courant_all = receipt.max_courant_all.max(courant);
        if fields.density[i] > 0.0 {
            receipt.max_courant_liquid = receipt.max_courant_liquid.max(courant);
        }
        if receivers[i] {
            let ratio = dt * divergence[i] / capacity;
            receipt.max_dt_div_over_capacity = receipt.max_dt_div_over_capacity.max(ratio);
            receipt.max_compression_ratio = receipt.max_compression_ratio.max(-ratio);
            if fields.pressure_member.get(i).copied().unwrap_or(0) == 0 {
                receipt.max_dt_div_over_capacity_outside_operator =
                    receipt.max_dt_div_over_capacity_outside_operator.max(ratio);
                receipt.max_compression_ratio_outside_operator =
                    receipt.max_compression_ratio_outside_operator.max(-ratio);
            }
        }
    }
    if receipt.max_dt_div_over_capacity == f64::NEG_INFINITY {
        receipt.max_dt_div_over_capacity = 0.0;
        receipt.max_compression_ratio = 0.0;
    }
    if receipt.max_dt_div_over_capacity_outside_operator == f64::NEG_INFINITY {
        receipt.max_dt_div_over_capacity_outside_operator = 0.0;
        receipt.max_compression_ratio_outside_operator = 0.0;
    }
}

fn publish_liquid_velocity_diagnostics(
    graph: &Graph,
    fields: &Fields,
    tracer: &FaceConsistentVelocity2d,
    receipt: &mut CellwiseRemapReceipt,
) -> Result<(), ValidationError> {
    let mut weight = 0.0_f64;
    let mut mean = [0.0_f64; 2];
    let mut primary_min = [f64::INFINITY; 2];
    let mut primary_max = [f64::NEG_INFINITY; 2];
    let mut tracing_min = [f64::INFINITY; 2];
    let mut tracing_max = [f64::NEG_INFINITY; 2];
    let mut mismatch = 0.0_f64;
    let mut full = 0_usize;
    for (cell, geometry) in graph.cells.iter().enumerate() {
        if fields.density[cell] <= 0.0 {
            continue;
        }
        let primary = [
            fields.cell_velocity[2 * cell] as f64,
            fields.cell_velocity[2 * cell + 1] as f64,
        ];
        let volume = fields.density[cell] as f64 * geometry.measure as f64;
        weight += volume;
        for axis in 0..2 {
            mean[axis] += volume * primary[axis];
        }
        if fields.density[cell] < 1.0 - 8.0 * f32::EPSILON {
            continue;
        }
        full += 1;
        let tracing = tracer.sample(
            graph,
            [geometry.center[0] as f64, geometry.center[1] as f64],
        )?;
        for axis in 0..2 {
            primary_min[axis] = primary_min[axis].min(primary[axis]);
            primary_max[axis] = primary_max[axis].max(primary[axis]);
            tracing_min[axis] = tracing_min[axis].min(tracing[axis]);
            tracing_max[axis] = tracing_max[axis].max(tracing[axis]);
        }
        mismatch = mismatch.max((tracing[0] - primary[0]).hypot(tracing[1] - primary[1]));
    }
    if weight > 0.0 {
        receipt.primary_liquid_velocity_mean = [mean[0] / weight, mean[1] / weight];
    }
    if full != 0 {
        receipt.primary_full_liquid_velocity_minimum = primary_min;
        receipt.primary_full_liquid_velocity_maximum = primary_max;
        receipt.tracing_full_liquid_velocity_minimum = tracing_min;
        receipt.tracing_full_liquid_velocity_maximum = tracing_max;
        receipt.tracing_full_liquid_max_velocity_mismatch = mismatch;
    }
    Ok(())
}

fn maximum_receiver_courant(
    graph: &Graph,
    fields: &Fields,
    rates: &[f64],
    receivers: &[bool],
    dt: f64,
) -> f64 {
    let mut outgoing = vec![0.0_f64; graph.cells.len()];
    for (face, &rate) in graph.subfaces.iter().zip(rates) {
        if face.negative_cell >= 0 {
            outgoing[face.negative_cell as usize] += rate.max(0.0);
        }
        if face.positive_cell >= 0 {
            outgoing[face.positive_cell as usize] += (-rate).max(0.0);
        }
    }
    graph
        .cells
        .iter()
        .enumerate()
        .filter_map(|(cell, geometry)| {
            let capacity = fields.capacity[cell] as f64 * geometry.measure as f64;
            (receivers[cell] && capacity > GEOMETRY_EPS).then_some(dt * outgoing[cell] / capacity)
        })
        .fold(0.0, f64::max)
}

fn publish_rate_change_metrics(
    graph: &Graph,
    fields: &Fields,
    before: &[f64],
    after: &[f64],
    receivers: &[bool],
    dt: f64,
    receipt: &mut CellwiseRemapReceipt,
) {
    receipt.max_courant_before_closure =
        maximum_receiver_courant(graph, fields, before, receivers, dt);
    receipt.max_courant_after_closure =
        maximum_receiver_courant(graph, fields, after, receivers, dt);
    receipt.closure_changed_subfaces = 0;
    receipt.closure_max_abs_rate_change = 0.0;
    receipt.closure_max_relative_rate_change = 0.0;
    receipt.closure_zero_original_rate_change_count = 0;
    for (&old, &new) in before.iter().zip(after) {
        let change = (new - old).abs();
        if change == 0.0 {
            continue;
        }
        receipt.closure_changed_subfaces += 1;
        receipt.closure_max_abs_rate_change = receipt.closure_max_abs_rate_change.max(change);
        if old == 0.0 {
            receipt.closure_zero_original_rate_change_count += 1;
        } else {
            receipt.closure_max_relative_rate_change = receipt
                .closure_max_relative_rate_change
                .max(change / old.abs());
        }
    }
}

fn publish_pregeometry_certificate(
    graph: &Graph,
    fields: &Fields,
    receiver_band: &[bool],
    divergence: &[f64],
    dt: f64,
    receipt: &mut CellwiseRemapReceipt,
) {
    receipt.pregeometry_max_abs_normalized_divergence = 0.0;
    receipt.pregeometry_certificate_violations = 0;
    receipt.pregeometry_certificate_violations_in_operator = 0;
    receipt.pregeometry_certificate_violations_outside_operator = 0;
    receipt.pregeometry_max_abs_normalized_divergence_in_operator = 0.0;
    receipt.pregeometry_max_abs_normalized_divergence_outside_operator = 0.0;
    receipt.pregeometry_max_divergence_cell = None;
    receipt.pregeometry_max_divergence_cell_pressure_member = false;
    receipt.pregeometry_max_divergence = 0.0;
    receipt.pregeometry_max_divergence_cell_capacity = 0.0;
    receipt.pregeometry_max_divergence_cell_density = 0.0;
    receipt.pregeometry_max_divergence_cell_liquid_volume = 0.0;
    for (cell, &receiver) in receiver_band.iter().enumerate() {
        if !receiver {
            continue;
        }
        let capacity = fields.capacity[cell] as f64 * graph.cells[cell].measure as f64;
        let normalized = if capacity > GEOMETRY_EPS {
            dt * divergence[cell].abs() / capacity
        } else {
            f64::INFINITY
        };
        receipt.pregeometry_max_abs_normalized_divergence = receipt
            .pregeometry_max_abs_normalized_divergence
            .max(normalized);
        let pressure_member = fields.pressure_member.get(cell).copied().unwrap_or(0) != 0;
        if pressure_member {
            receipt.pregeometry_max_abs_normalized_divergence_in_operator = receipt
                .pregeometry_max_abs_normalized_divergence_in_operator
                .max(normalized);
        } else {
            receipt.pregeometry_max_abs_normalized_divergence_outside_operator = receipt
                .pregeometry_max_abs_normalized_divergence_outside_operator
                .max(normalized);
        }
        if receipt.pregeometry_max_divergence_cell.is_none()
            || normalized >= receipt.pregeometry_max_abs_normalized_divergence
        {
            receipt.pregeometry_max_divergence_cell = Some(cell as u32);
            receipt.pregeometry_max_divergence_cell_pressure_member = pressure_member;
            receipt.pregeometry_max_divergence = divergence[cell];
            receipt.pregeometry_max_divergence_cell_capacity = capacity;
            receipt.pregeometry_max_divergence_cell_density = fields.density[cell] as f64;
            receipt.pregeometry_max_divergence_cell_liquid_volume =
                fields.density[cell] as f64 * graph.cells[cell].measure as f64;
        }
        if !normalized.is_finite() || normalized > CONTINUITY_TARGET {
            receipt.pregeometry_certificate_violations += 1;
            if pressure_member {
                receipt.pregeometry_certificate_violations_in_operator += 1;
            } else {
                receipt.pregeometry_certificate_violations_outside_operator += 1;
            }
        }
    }
}

fn local_continuity_closure(
    graph: &Graph,
    fields: &Fields,
    receivers: &[bool],
    initial_divergence: &[f64],
    dt: f64,
    rates: &mut [f64],
    receipt: &mut CellwiseRemapReceipt,
) {
    let mut divergence = initial_divergence.to_vec();
    let mut order: Vec<_> = (0..graph.cells.len())
        .filter(|&i| {
            receivers[i]
                && fields.pressure_member.get(i).copied().unwrap_or(0) == 0
                && divergence[i].abs() > GEOMETRY_EPS
        })
        .collect();
    order.sort_by(|&a, &b| {
        divergence[b]
            .abs()
            .total_cmp(&divergence[a].abs())
            .then(a.cmp(&b))
    });
    receipt.closure_unknowns = order.len();
    receipt.closure_iterations = usize::from(!order.is_empty());
    for cell in order {
        let capacity = fields.capacity[cell] as f64 * graph.cells[cell].measure as f64;
        if capacity <= GEOMETRY_EPS || dt * divergence[cell].abs() / capacity <= CONTINUITY_TARGET {
            continue;
        }
        let free: Vec<_> = graph
            .subfaces
            .iter()
            .filter(|face| {
                let row = &graph.rows[face.row_id as usize];
                let other = if face.negative_cell == cell as i32 {
                    face.positive_cell
                } else if face.positive_cell == cell as i32 {
                    face.negative_cell
                } else {
                    return false;
                };
                face.aperture > 1.0e-8
                    && row.open_fraction > 1.0e-8
                    && row.kind != crate::RowKind::ClosedWorld
                    && fields
                        .pressure_row_member
                        .get(row.id as usize)
                        .copied()
                        .unwrap_or(0)
                        == 0
                    && (other < 0 || !receivers[other as usize])
                    && (other < 0
                        || fields
                            .pressure_member
                            .get(other as usize)
                            .copied()
                            .unwrap_or(0)
                            == 0)
            })
            .map(|face| face.id as usize)
            .collect();
        let weight: f64 = free
            .iter()
            .map(|&i| graph.subfaces[i].measure as f64 * graph.subfaces[i].aperture as f64)
            .sum();
        if weight <= GEOMETRY_EPS {
            receipt.closure_unresolved += 1;
            continue;
        }
        // Two-sided comparison arm: positive divergence decreases outflow (or
        // increases inflow) just as negative divergence does the converse.
        let needed = -divergence[cell];
        for face_index in free {
            let face = &graph.subfaces[face_index];
            let share = needed * face.measure as f64 * face.aperture as f64 / weight;
            if face.negative_cell == cell as i32 {
                rates[face_index] += share;
                divergence[cell] += share;
                if face.positive_cell >= 0 {
                    divergence[face.positive_cell as usize] -= share;
                }
            } else {
                rates[face_index] -= share;
                divergence[cell] += share;
                if face.negative_cell >= 0 {
                    divergence[face.negative_cell as usize] -= share;
                }
            }
            receipt.closure_adjustments += 1;
        }
    }
    receipt.closure_unresolved = receivers
        .iter()
        .enumerate()
        .filter(|&(cell, &receiver)| {
            if !receiver || fields.pressure_member.get(cell).copied().unwrap_or(0) != 0 {
                return false;
            }
            let capacity = fields.capacity[cell] as f64 * graph.cells[cell].measure as f64;
            capacity <= GEOMETRY_EPS || dt * divergence[cell].abs() / capacity > CONTINUITY_TARGET
        })
        .count();
}

fn gather_material(
    graph: &Graph,
    fields: &Fields,
    polygons: &[Vec<Point>],
    receipt: &mut CellwiseRemapReceipt,
) -> Result<Vec<f64>, ValidationError> {
    let mut next = vec![0.0; graph.cells.len()];
    let mut gathered_by_donor = vec![0.0; graph.cells.len()];
    for (receiver, polygon) in polygons.iter().enumerate() {
        let mut volume = 0.0;
        for donor in candidate_donors(graph, polygon) {
            let donor_volume = fields.density[donor] as f64 * graph.cells[donor].measure as f64;
            if donor_volume <= GEOMETRY_EPS {
                continue;
            }
            let c = &graph.cells[donor];
            let mut clipped = clip_rectangle(
                polygon,
                [c.minimum[0] as f64, c.minimum[1] as f64],
                [c.maximum[0] as f64, c.maximum[1] as f64],
            );
            receipt.gather_clips += 1;
            if clipped.len() < 3 {
                continue;
            }
            let capacity = fields.capacity[donor] as f64 * c.measure as f64;
            if (donor_volume - capacity).abs() <= VOLUME_MARGIN * capacity.max(1.0) {
                let contribution = signed_area(&clipped).abs();
                volume += contribution;
                gathered_by_donor[donor] += contribution;
                continue;
            }
            let normal = [
                fields.interface_normal[2 * donor] as f64,
                fields.interface_normal[2 * donor + 1] as f64,
            ];
            if normal[0].hypot(normal[1]) <= GEOMETRY_EPS {
                return Err(ValidationError(format!(
                    "cellwise remap partial donor {donor} has no PLIC plane"
                )));
            }
            let offset = fields.interface_offset[donor] as f64
                + normal[0] * c.center[0] as f64
                + normal[1] * c.center[1] as f64;
            clipped = clip_half_plane(&clipped, normal, offset);
            receipt.gather_clips += 1;
            let contribution = signed_area(&clipped).abs();
            volume += contribution;
            gathered_by_donor[donor] += contribution;
        }
        next[receiver] = volume;
    }
    receipt.gather_initial_liquid_volume = graph
        .cells
        .iter()
        .enumerate()
        .map(|(cell, geometry)| fields.density[cell] as f64 * geometry.measure as f64)
        .sum();
    receipt.gather_final_liquid_volume = next.iter().sum();
    receipt.gather_absolute_volume_error =
        (receipt.gather_final_liquid_volume - receipt.gather_initial_liquid_volume).abs();
    receipt.gather_volume_roundoff_bound =
        VOLUME_MARGIN * receipt.gather_initial_liquid_volume.max(1.0);
    for (donor, (&gathered, geometry)) in gathered_by_donor.iter().zip(&graph.cells).enumerate() {
        let expected = fields.density[donor] as f64 * geometry.measure as f64;
        let error = (gathered - expected).abs();
        if error > receipt.gather_worst_donor_volume_error {
            receipt.gather_worst_donor_volume_error = error;
            receipt.gather_worst_donor = Some(donor as u32);
            receipt.gather_worst_donor_minimum = [
                geometry.minimum[0] as f64,
                geometry.minimum[1] as f64,
            ];
            receipt.gather_worst_donor_maximum = [
                geometry.maximum[0] as f64,
                geometry.maximum[1] as f64,
            ];
            receipt.gather_worst_donor_density = fields.density[donor] as f64;
            receipt.gather_worst_donor_capacity = fields.capacity[donor] as f64;
            receipt.gather_worst_donor_gathered_volume = gathered;
            receipt.gather_worst_donor_expected_volume = expected;
            receipt.gather_worst_donor_interface_normal = [
                fields.interface_normal[2 * donor] as f64,
                fields.interface_normal[2 * donor + 1] as f64,
            ];
            receipt.gather_worst_donor_interface_offset = fields.interface_offset[donor] as f64;
            receipt.gather_worst_donor_roundoff_bound =
                VOLUME_MARGIN * (fields.capacity[donor] as f64 * geometry.measure as f64).max(1.0);
        }
    }
    Ok(next)
}

fn candidate_donors(graph: &Graph, polygon: &[Point]) -> BTreeSet<usize> {
    let (minimum, maximum) = bounds(polygon);
    let x0 = minimum[0].floor().max(0.0) as i32;
    let y0 = minimum[1].floor().max(0.0) as i32;
    let x1 = maximum[0].ceil().min(graph.dimensions[0] as f64) as i32;
    let y1 = maximum[1].ceil().min(graph.dimensions[1] as f64) as i32;
    let mut result = BTreeSet::new();
    for y in y0..y1 {
        for x in x0..x1 {
            if let Some(owner) =
                crate::numerics::owner_at(graph, [x as f32 + 0.5, y as f32 + 0.5, 0.0])
            {
                result.insert(owner);
            }
        }
    }
    result
}

fn clip_rectangle(polygon: &[Point], minimum: Point, maximum: Point) -> Vec<Point> {
    let mut result = polygon.to_vec();
    result = clip_axis(&result, 0, minimum[0], true);
    result = clip_axis(&result, 0, maximum[0], false);
    result = clip_axis(&result, 1, minimum[1], true);
    clip_axis(&result, 1, maximum[1], false)
}

fn clip_axis(polygon: &[Point], axis: usize, bound: f64, keep_greater: bool) -> Vec<Point> {
    clip_predicate(polygon, |p| {
        if keep_greater {
            p[axis] - bound
        } else {
            bound - p[axis]
        }
    })
}

fn clip_half_plane(polygon: &[Point], normal: Point, offset: f64) -> Vec<Point> {
    clip_predicate(polygon, |p| offset - normal[0] * p[0] - normal[1] * p[1])
}

fn clip_predicate(polygon: &[Point], signed_inside: impl Fn(Point) -> f64) -> Vec<Point> {
    if polygon.len() < 3 {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(polygon.len() + 2);
    let mut previous = *polygon.last().unwrap();
    let mut previous_value = signed_inside(previous);
    for &current in polygon {
        let current_value = signed_inside(current);
        let previous_inside = previous_value >= -GEOMETRY_EPS;
        let current_inside = current_value >= -GEOMETRY_EPS;
        if previous_inside != current_inside {
            let denominator = previous_value - current_value;
            if denominator.abs() > GEOMETRY_EPS {
                let t = previous_value / denominator;
                output.push([
                    previous[0] + t * (current[0] - previous[0]),
                    previous[1] + t * (current[1] - previous[1]),
                ]);
            }
        }
        if current_inside {
            output.push(current);
        }
        previous = current;
        previous_value = current_value;
    }
    output
}

fn signed_area(polygon: &[Point]) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }
    // Translate before taking cross products so grid-scale world coordinates
    // do not cancel the unit-scale polygon area. Neumaier compensation then
    // keeps the result stable as spatial edge refinement adds vertices.
    let origin = polygon[0];
    let mut twice = 0.0;
    let mut correction = 0.0;
    for i in 0..polygon.len() {
        let a = [polygon[i][0] - origin[0], polygon[i][1] - origin[1]];
        let next = polygon[(i + 1) % polygon.len()];
        let b = [next[0] - origin[0], next[1] - origin[1]];
        let term = cross(a, b);
        let sum = twice + term;
        if twice.abs() >= term.abs() {
            correction += (twice - sum) + term;
        } else {
            correction += (term - sum) + twice;
        }
        twice = sum;
    }
    0.5 * (twice + correction)
}

fn preimage_area_is_negative(area: f64, capacity: f64) -> bool {
    area < -VOLUME_MARGIN * capacity.max(1.0)
}

fn receivers_outside_band(raw: &[bool], corrected: &[bool], band: &[bool]) -> usize {
    raw.iter()
        .zip(corrected)
        .zip(band)
        .filter(|&((&raw, &corrected), &band)| (raw || corrected) && !band)
        .count()
}

fn reject_incomplete_receiver_band(fields: &mut Fields, receipt: &CellwiseRemapReceipt) -> bool {
    if receipt.postgeometry_receivers_outside_pregeometry_band == 0 {
        return false;
    }
    fields.fault = Some(NumericalFault {
        stage: "cellwise-remap-receiver-band".into(),
        index: 0,
        observed: receipt.postgeometry_receivers_outside_pregeometry_band as f32,
        expected: 0.0,
    });
    true
}

fn bounds(polygon: &[Point]) -> (Point, Point) {
    polygon.iter().fold(
        ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
        |(mut lo, mut hi), &p| {
            for a in 0..2 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
            (lo, hi)
        },
    )
}

fn polygon_self_intersects(polygon: &[Point]) -> bool {
    for i in 0..polygon.len() {
        let i_next = (i + 1) % polygon.len();
        for j in i + 1..polygon.len() {
            let j_next = (j + 1) % polygon.len();
            if i == j || i_next == j || j_next == i {
                continue;
            }
            if segments_cross(polygon[i], polygon[i_next], polygon[j], polygon[j_next]) {
                return true;
            }
        }
    }
    false
}

fn segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool {
    let (o1, b1) = robust_orient(a, b, c);
    let (o2, b2) = robust_orient(a, b, d);
    let (o3, b3) = robust_orient(c, d, a);
    let (o4, b4) = robust_orient(c, d, b);
    o1.abs() > b1
        && o2.abs() > b2
        && o3.abs() > b3
        && o4.abs() > b4
        && o1.signum() != o2.signum()
        && o3.signum() != o4.signum()
}

/// Return the 2-D orientation determinant and a forward-error bound derived
/// from the two products that form it. Comparing products of two orientation
/// determinants to a length-independent epsilon is dimensionally wrong and
/// hid small, well-resolved crossings in Figure 7 after impact.
fn robust_orient(a: Point, b: Point, c: Point) -> (f64, f64) {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let acx = c[0] - a[0];
    let acy = c[1] - a[1];
    let first = abx * acy;
    let second = aby * acx;
    let determinant = first - second;
    // Subtraction to form each coordinate difference, two products and their
    // subtraction are covered by this conservative Higham-style census.
    let operations = 8.0;
    let gamma = operations * f64::EPSILON / (1.0 - operations * f64::EPSILON);
    (determinant, gamma * (first.abs() + second.abs()))
}

fn orient(a: Point, b: Point, c: Point) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn distance_squared(a: Point, b: Point) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::BoundaryMode;
    use crate::topology::{compile_topology, BrickSeed, TopologySeed};

    fn polyline_path(path: Vec<Point>) -> TracedPoint {
        TracedPoint {
            signed_area_integral: 0.5 * polyline_cross_sum(&path),
            path,
            support_extrapolation_samples: 0,
            constrained_segments: 0,
            rt0_region_transitions: 0,
        }
    }

    fn uniform_test_graph(resolution: u8) -> Graph {
        compile_topology::<2>(TopologySeed {
            dimensions: [resolution as u32, resolution as u32, 1],
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Closed; 6],
            bricks: vec![BrickSeed {
                id: 0,
                key: 0,
                coordinate: [0, 0, 0],
                span_bricks: 1,
                resolution,
                active: true,
                density: Vec::new(),
                gamma: Vec::new(),
                refinement_region_scale: None,
            }],
        })
        .unwrap()
        .graph
    }

    #[test]
    fn edge_area_correction_is_affine_and_keeps_endpoints() {
        let fixed = vec![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]];
        let mut traced = vec![[0.1, 0.2], [0.6, 0.3], [1.1, 0.2]];
        let first_path = polyline_path(vec![fixed[0], traced[0]]);
        let last_path = polyline_path(vec![fixed[2], traced[2]]);
        let endpoints = [traced[0], traced[2]];
        correct_chain_area(&fixed, &mut traced, &first_path, &last_path, 0.75).unwrap();
        assert!((swept_area(&fixed, &traced, &first_path, &last_path) - 0.75).abs() < 1e-12);
        assert_eq!([traced[0], traced[2]], endpoints);
    }

    #[test]
    fn one_sample_unit_edge_gets_an_untraced_correction_node() {
        let mut fixed = vec![[0.0, 0.0], [1.0, 0.0]];
        let mut traced = vec![[0.25, 0.0], [1.25, 0.0]];
        assert!(insert_correction_midpoint(&mut fixed, &mut traced));
        assert_eq!(fixed, vec![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]);
        assert_eq!(traced[1], [0.75, 0.0]);
    }

    #[test]
    fn swept_area_includes_curved_endpoint_pathlines_instead_of_their_chords() {
        let fixed = vec![[0.0, 0.0], [1.0, 0.0]];
        let traced = vec![[0.0, 1.0], [1.0, 1.0]];
        let first_path = polyline_path(vec![[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]]);
        let last_path = polyline_path(vec![[1.0, 0.0], [1.5, 0.5], [1.0, 1.0]]);

        // The right endpoint's two path segments enclose a triangle of area
        // 1/4 outside the unit chord-sided swept rectangle.
        let pathline_area = swept_area(&fixed, &traced, &first_path, &last_path);
        let chord_area = signed_area(&[fixed[0], fixed[1], traced[1], traced[0]]);
        assert!((pathline_area - 1.25).abs() < 1.0e-12);
        assert!((chord_area - 1.0).abs() < 1.0e-12);
        assert!((pathline_area - chord_area).abs() > 0.2);
    }

    #[test]
    fn divergence_free_saddle_path_integrals_equal_time_integrated_face_flux() {
        let a = 0.7_f64;
        let dt = 1.3_f64;
        let x = 2.0_f64;
        let (y0, y1) = (1.0_f64, 2.0_f64);
        let steps = 512;
        let path = |y: f64| {
            (0..=steps)
                .map(|i| {
                    let t = dt * i as f64 / steps as f64;
                    [x * (-a * t).exp(), y * (a * t).exp()]
                })
                .collect::<Vec<_>>()
        };
        let first_path = TracedPoint {
            path: path(y0),
            signed_area_integral: a * x * y0 * dt,
            support_extrapolation_samples: 0,
            constrained_segments: 0,
            rt0_region_transitions: 0,
        };
        let last_path = TracedPoint {
            path: path(y1),
            signed_area_integral: a * x * y1 * dt,
            support_extrapolation_samples: 0,
            constrained_segments: 0,
            rt0_region_transitions: 0,
        };
        let fixed = vec![[x, y0], [x, y1]];
        let traced = vec![first_path.end(), last_path.end()];
        let integrated_flux = dt * a * x * (y1 - y0);
        let pathline_area = swept_area(&fixed, &traced, &first_path, &last_path);
        let chord_area = signed_area(&[fixed[0], fixed[1], traced[1], traced[0]]);

        assert!((pathline_area - integrated_flux).abs() < 1.0e-12);
        assert!((chord_area - integrated_flux).abs() > 1.0e-2);
    }

    #[test]
    fn rk4_augmented_path_area_converges_for_face_consistent_saddle() {
        let graph = compile_topology::<2>(TopologySeed {
            dimensions: [8, 8, 1],
            generation: 1,
            sparse_air_phi: 0.5,
            boundaries: [BoundaryMode::Closed; 6],
            bricks: vec![BrickSeed {
                id: 0,
                key: 0,
                coordinate: [0, 0, 0],
                span_bricks: 1,
                resolution: 8,
                active: true,
                density: Vec::new(),
                gamma: Vec::new(),
                refinement_region_scale: None,
            }],
        })
        .unwrap()
        .graph;
        let a = 0.25_f64;
        let dt = 0.5_f32;
        let n = graph.cells.len();
        let r = graph.rows.len();
        let mut cell_velocity = Vec::with_capacity(2 * n);
        for cell in &graph.cells {
            cell_velocity.extend([
                (a * cell.center[0] as f64) as f32,
                (-a * cell.center[1] as f64) as f32,
            ]);
        }
        let mut fields = Fields {
            density: vec![1.0; n],
            gamma: vec![1.0; n],
            capacity: vec![1.0; n],
            capacity_before: vec![1.0; n],
            capacity_after: vec![1.0; n],
            cell_velocity,
            face_velocity: graph
                .rows
                .iter()
                .map(|row| {
                    if row.axis == 0 {
                        (a * row.center[0] as f64) as f32
                    } else {
                        (-a * row.center[1] as f64) as f32
                    }
                })
                .collect(),
            pressure: vec![0.0; n],
            pressure_rhs: vec![0.0; n],
            pressure_diagonal: vec![1.0; n],
            pressure_member: vec![1; n],
            pressure_row_member: vec![1; r],
            extension_depth: vec![0; n],
            interface_normal: vec![0.0; 2 * n],
            interface_offset: vec![0.0; n],
            ..Fields::default()
        };
        let tracer = FaceConsistentVelocity2d::new(&graph, &fields, dt).unwrap();
        let start = [2.0, 2.0];
        let exact = a * start[0] * start[1] * dt as f64;
        let mut errors = Vec::new();
        for segments in [1, 2, 4, 8] {
            let mut receipt = CellwiseRemapReceipt::default();
            let traced = trace_rk4(
                &graph,
                &fields,
                &tracer,
                start,
                dt as f64,
                segments,
                &mut receipt,
            )
            .unwrap();
            assert_eq!(receipt.constrained_pathline_segments, 0);
            assert_eq!(receipt.clips, 0);
            errors.push((traced.signed_area_integral - exact).abs());
        }
        assert!(errors[1] < errors[0] / 8.0, "{errors:?}");
        assert!(errors[2] < errors[1] / 8.0, "{errors:?}");
        assert!(errors[3] < errors[2], "{errors:?}");
        assert!(errors[3] < 5.0e-9, "{errors:?}");

        let mut projection_receipt = CellwiseRemapReceipt::default();
        let (_, interior_constrained) = projected_stage(
            &graph,
            &fields,
            [2.0, 2.0],
            [2.125, 2.25],
            &mut projection_receipt,
        );
        assert!(!interior_constrained);
        assert_eq!(projection_receipt.clips, 0);
        for candidate in [[8.0, 4.125000000123], [8.0, 3.875000000123]] {
            let (wall_tangent, constrained) = projected_stage(
                &graph,
                &fields,
                [8.0, 4.0],
                candidate,
                &mut projection_receipt,
            );
            assert!(!constrained);
            assert_eq!(wall_tangent, candidate);
        }
        for (start, candidate) in [
            ([0.25, 2.0], [-0.25, 2.0]),
            ([7.75, 2.0], [8.25, 2.0]),
            ([2.0, 0.25], [2.0, -0.25]),
            ([2.0, 7.75], [2.0, 8.25]),
        ] {
            let (_, constrained) =
                projected_stage(&graph, &fields, start, candidate, &mut projection_receipt);
            assert!(constrained, "outward wall segment {start:?}->{candidate:?}");
        }
        assert!(projection_receipt.clips > 0);

        // A represented zero-capacity cell is an internal solid. Its lower and
        // upper faces admit tangential points but symmetrically stop normal
        // penetration from the fluid on either side.
        let solid = graph
            .cells
            .iter()
            .position(|cell| cell.center[..2] == [4.5, 4.5])
            .unwrap();
        fields.capacity[solid] = 0.0;
        let (below, below_constrained) = projected_stage(
            &graph,
            &fields,
            [4.5, 3.75],
            [4.5, 4.25],
            &mut projection_receipt,
        );
        let (above, above_constrained) = projected_stage(
            &graph,
            &fields,
            [4.5, 5.25],
            [4.5, 4.75],
            &mut projection_receipt,
        );
        assert!(below_constrained && above_constrained);
        assert!(below[1] <= 4.0 && above[1] >= 5.0, "{below:?} {above:?}");
        // Eight coarse probes plus eight bisections locate the two surfaces
        // within one 0.5/2048 interval; both sides also apply the same 1e-4
        // parametric clearance.
        assert!((below[1] + above[1] - 9.0).abs() <= 3.0e-4, "{below:?} {above:?}");
    }

    #[test]
    fn folded_bow_tie_cannot_cancel_a_potential_liquid_intersection() {
        let bow_tie = vec![[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]];
        let liquid = vec![[0.25, 0.25], [1.75, 0.25], [1.75, 1.75], [0.25, 1.75]];
        assert_eq!(signed_area(&bow_tie), 0.0);
        assert!(polygon_intersects_region(&bow_tie, &liquid));
    }

    #[test]
    fn folded_hull_relevance_uses_the_donor_plic_not_its_bounding_cell() {
        let graph = uniform_test_graph(8);
        let mut fields = Fields {
            density: vec![0.0; graph.cells.len()],
            capacity: vec![1.0; graph.cells.len()],
            interface_normal: vec![0.0; 2 * graph.cells.len()],
            interface_offset: vec![0.0; graph.cells.len()],
            ..Fields::default()
        };
        let donor = graph
            .cells
            .iter()
            .position(|cell| cell.center[..2] == [1.5, 1.5])
            .unwrap();
        fields.density[donor] = 0.25;
        fields.interface_normal[2 * donor] = 1.0;
        // The represented quarter-cell liquid is x <= 1.25.
        fields.interface_offset[donor] = -0.25;

        // Both folded hulls lie in the donor's cell-sized bounding box. The
        // first is wholly in its empty side, matching the irrelevant frame-17
        // case; the second reaches the actual PLIC and must remain protected.
        let dry_side = vec![[1.75, 1.2], [1.9, 1.8], [1.75, 1.8], [1.9, 1.2]];
        let liquid_side = vec![[1.1, 1.2], [1.4, 1.8], [1.1, 1.8], [1.4, 1.2]];
        assert!(polygon_self_intersects(&dry_side));
        assert!(polygon_self_intersects(&liquid_side));
        assert!(!convex_hull_can_reach_liquid(&graph, &fields, &dry_side));
        assert!(convex_hull_can_reach_liquid(
            &graph,
            &fields,
            &liquid_side,
        ));
    }

    #[test]
    fn clipping_preserves_concave_signed_area() {
        let polygon = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0], [0.0, 2.0]];
        let clipped = clip_rectangle(&polygon, [0.5, 0.5], [1.5, 1.5]);
        assert!((signed_area(&clipped).abs() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn proper_segment_crossing_uses_orientation_signs_not_their_tiny_product() {
        // Exact frame-41 receiver-1999 edges. Their intersection generated a
        // three-cell overlap, but multiplying the four orientation values
        // before comparing with an area epsilon hid this proper crossing.
        let a = [127.12139322141157, 15.301301877582288];
        let b = [127.12814163049019, 15.32010228561595];
        let c = [127.12244762334737, 15.304262035131734];
        let d = [127.11995478947227, 15.296588451279513];
        assert!(segments_cross(a, b, c, d));

        // Classification must be independent of world origin and ordinary
        // unit changes. This transform keeps enough f64 separation while
        // reducing every determinant, and their product, by many orders.
        let transform = |point: Point| {
            [
                4096.0 + 0.01 * (point[0] - 127.0),
                -8192.0 + 0.01 * (point[1] - 15.0),
            ]
        };
        assert!(segments_cross(
            transform(a),
            transform(b),
            transform(c),
            transform(d),
        ));

        // Close and almost parallel is not enough: separated segments must
        // remain disjoint under the same robust predicate.
        assert!(!segments_cross(
            [0.0, 0.0],
            [1.0, 1.0e-8],
            [0.0, 2.0e-8],
            [1.0, 3.1e-8],
        ));
    }

    #[test]
    fn clockwise_preimage_is_rejected_beyond_the_volume_margin() {
        let clockwise = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        let area = signed_area(&clockwise);
        assert_eq!(area, -1.0);
        assert!(preimage_area_is_negative(area, 1.0));
        assert!(!preimage_area_is_negative(-0.5 * VOLUME_MARGIN, 1.0));
    }

    #[test]
    fn incomplete_receiver_band_sets_named_fault() {
        let raw = [false, true, false];
        let corrected = [false, false, true];
        let band = [true, false, true];
        let mut receipt = CellwiseRemapReceipt::default();
        receipt.postgeometry_receivers_outside_pregeometry_band =
            receivers_outside_band(&raw, &corrected, &band);
        assert_eq!(receipt.postgeometry_receivers_outside_pregeometry_band, 1);
        let mut fields = Fields::default();
        assert!(reject_incomplete_receiver_band(&mut fields, &receipt));
        assert_eq!(
            fields.fault.as_ref().unwrap().stage,
            "cellwise-remap-receiver-band"
        );
    }

    #[test]
    fn local_closure_is_two_sided_and_never_uses_closed_world_as_a_vent() {
        let graph = uniform_test_graph(4);
        let mut fields = Fields {
            capacity: vec![1.0; graph.cells.len()],
            pressure_member: vec![0; graph.cells.len()],
            ..Fields::default()
        };
        let face = graph
            .subfaces
            .iter()
            .find(|face| face.negative_cell >= 0 && face.positive_cell >= 0)
            .unwrap();
        let cell = face.negative_cell as usize;
        let outward_sign = if face.negative_cell == cell as i32 {
            1.0
        } else {
            -1.0
        };
        let mut receiver = vec![false; graph.cells.len()];
        receiver[cell] = true;
        for divergence_sign in [-1.0, 1.0] {
            let mut rates = vec![0.0; graph.subfaces.len()];
            rates[face.id as usize] = divergence_sign * outward_sign;
            let initial = cell_divergence(&graph, &rates);
            assert_eq!(initial[cell], divergence_sign);
            let mut receipt = CellwiseRemapReceipt::default();
            local_continuity_closure(
                &graph,
                &fields,
                &receiver,
                &initial,
                1.0,
                &mut rates,
                &mut receipt,
            );
            assert!(cell_divergence(&graph, &rates)[cell].abs() <= GEOMETRY_EPS);
            assert!(receipt.closure_adjustments > 0);
            assert_eq!(receipt.closure_unresolved, 0);
        }

        fields.pressure_row_member = vec![0; graph.rows.len()];
        fields.pressure_row_member[face.row_id as usize] = 1;
        let mut rates = vec![0.0; graph.subfaces.len()];
        rates[face.id as usize] = outward_sign;
        let initial = cell_divergence(&graph, &rates);
        let fixed_rate = rates[face.id as usize];
        let mut receipt = CellwiseRemapReceipt::default();
        local_continuity_closure(
            &graph,
            &fields,
            &receiver,
            &initial,
            1.0,
            &mut rates,
            &mut receipt,
        );
        assert_eq!(rates[face.id as usize], fixed_rate);
        assert!(cell_divergence(&graph, &rates)[cell].abs() <= GEOMETRY_EPS);
        fields.pressure_row_member.fill(0);

        // With every represented cell marked as a receiver, the only possible
        // exterior vents are ClosedWorld rows. They must remain unchanged and
        // the two-cell defect must be reported unresolved.
        let receiver = vec![true; graph.cells.len()];
        let mut rates = vec![0.0; graph.subfaces.len()];
        rates[face.id as usize] = outward_sign;
        let initial = cell_divergence(&graph, &rates);
        let original = rates.clone();
        let mut receipt = CellwiseRemapReceipt::default();
        local_continuity_closure(
            &graph,
            &fields,
            &receiver,
            &initial,
            1.0,
            &mut rates,
            &mut receipt,
        );
        assert_eq!(rates, original);
        assert_eq!(receipt.closure_adjustments, 0);
        assert!(receipt.closure_unresolved >= 2);
        fields.fault = None;
    }
}
