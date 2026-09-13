//! Native production-world runner used by tools/wasm/world-parity.ts.
use fluid_core::initial_scene::SceneDocument;
use fluid_core::production_scene::ProductionSceneOptions;
use fluid_core::world::{TransportExperiment, World, WorldOptions};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeSet;
use std::io::{self, Read};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Input {
    scene: SceneDocument,
    #[serde(default)]
    production_options: ProductionSceneOptions,
    #[serde(default)]
    world_options: WorldOptions,
    #[serde(default = "frames")]
    frames: u32,
    /// Keep long experimental censuses small while exercising the same World
    /// advance path. The default output remains the full parity dump.
    #[serde(default, alias = "receipts_only")]
    receipts_only: bool,
    /// Reject a frame unless the opt-in cellwise path completed a real
    /// material commit. This keeps long no-UI regressions from treating a
    /// fail-closed, unchanged frame as a successful simulation step.
    #[serde(default, alias = "require_cellwise_commit")]
    require_cellwise_commit: bool,
    /// Record compact volume-weighted velocity populations at each observer
    /// stage without serializing full graphs and fields.
    #[serde(default, alias = "observe_stage_metrics")]
    observe_stage_metrics: bool,
    /// Diagnostic mode: serialize the mutated World and completed observer
    /// stages at the first rejected frame instead of discarding that evidence.
    #[serde(default, alias = "capture_failure")]
    capture_failure: bool,
    /// Keep the compact history but serialize the final accepted World. This
    /// is useful for auditing which concrete bricks own an aggregate receipt
    /// without retaining every intermediate graph and field array.
    #[serde(default, alias = "capture_final_state")]
    capture_final_state: bool,
    /// Diagnostic-only final-frame override. This permits multiple geometry
    /// resolutions to consume the exact same evolved pre-transport state.
    #[serde(default, alias = "final_frame_transport_experiment")]
    final_frame_transport_experiment: Option<TransportExperiment>,
}
fn frames() -> u32 {
    3
}
fn capture(world: &World) -> serde_json::Value {
    let embedding_fault = world
        .embedding
        .as_ref()
        .and_then(|e| e.mapping_fault)
        .map(|f| json!({"kind":format!("{:?}",f.kind).to_lowercase(),"id":f.id}));
    let embedding = world.embedding.as_ref().map(|e| {
        json!({
            "centreCell": e.centre_cell,
            "centreRow": e.centre_row,
            "pressureMember": e.pressure_member,
            "rowActive": e.row_active,
            "rowTheta": e.row_theta,
        })
    });
    json!({"graph":world.state.topology.graph,"fields":world.state.fields,
        "receipt":world.receipt(),"resolution":world.resolution_receipt,
        "rigid":world.physical.as_ref().map(|p|&p.coupling_receipts),
        "pressureAuthority":world.embedding.as_ref().map(|e|&e.pressure_authority.receipt).unwrap_or(&world.pressure_authority.receipt),
        "scalarAuthority":world.scalar_authority.receipt,
        "embeddingFault":embedding_fault,"embedding":embedding})
}
fn capture_receipt(world: &World) -> serde_json::Value {
    let density_range = if world.state.fields.density.is_empty() {
        None
    } else {
        Some([
            world
                .state
                .fields
                .density
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min),
            world
                .state
                .fields
                .density
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
        ])
    };
    let mut liquid_minimum = [f32::INFINITY; 2];
    let mut liquid_maximum = [f32::NEG_INFINITY; 2];
    let mut represented_minimum = [f32::INFINITY; 2];
    let mut represented_maximum = [f32::NEG_INFINITY; 2];
    let mut represented_cell_count = 0;
    let mut liquid_cell_count = 0;
    let mut cell_widths = BTreeSet::new();
    let mut cells_by_width = std::collections::BTreeMap::<u32, [f64; 7]>::new();
    let mut liquid_weighted = [0.0_f64; 2];
    let mut liquid_velocity_weighted = [0.0_f64; 2];
    let mut liquid_velocity_second_moment = [[0.0_f64; 2]; 2];
    let mut liquid_measure = 0.0_f64;
    let mut liquid_measure_by_fine_y =
        vec![0.0_f64; world.state.topology.graph.dimensions[1] as usize];
    for cell in &world.state.topology.graph.cells {
        let width = cell.widths[0] as u32;
        cell_widths.insert(width);
        let density = world.state.fields.density[cell.id as usize] as f64;
        let capacity = world.state.fields.capacity[cell.id as usize] as f64;
        let volume = density * cell.measure as f64;
        let width_counts = cells_by_width.entry(width).or_default();
        width_counts[0] += 1.0;
        if density <= 1.0e-5 {
            continue;
        }
        width_counts[1] += 1.0;
        width_counts[4] += volume;
        if density >= capacity - 1.0e-5 {
            width_counts[2] += 1.0;
            width_counts[5] += volume;
        } else {
            width_counts[3] += 1.0;
            width_counts[6] += volume;
        }
        liquid_cell_count += 1;
        liquid_measure += volume;
        let velocity = [
            world.state.fields.cell_velocity[2 * cell.id as usize] as f64,
            world.state.fields.cell_velocity[2 * cell.id as usize + 1] as f64,
        ];
        for a in 0..2 {
            liquid_velocity_weighted[a] += volume * velocity[a];
            for b in 0..2 {
                liquid_velocity_second_moment[a][b] += volume * velocity[a] * velocity[b];
            }
        }
        let y0 = cell.minimum[1].max(0.0) as usize;
        let y1 = (cell.maximum[1] as usize).min(liquid_measure_by_fine_y.len());
        for bin in &mut liquid_measure_by_fine_y[y0..y1] {
            *bin += volume / (y1 - y0) as f64;
        }
        if density / capacity.max(1.0e-6) >= 0.5 {
            represented_cell_count += 1;
            for axis in 0..2 {
                represented_minimum[axis] = represented_minimum[axis].min(cell.minimum[axis]);
                represented_maximum[axis] = represented_maximum[axis].max(cell.maximum[axis]);
            }
        }
        for axis in 0..2 {
            liquid_minimum[axis] = liquid_minimum[axis].min(cell.minimum[axis]);
            liquid_maximum[axis] = liquid_maximum[axis].max(cell.maximum[axis]);
            liquid_weighted[axis] += volume * cell.center[axis] as f64;
        }
    }
    let liquid_bounds = if liquid_cell_count != 0 {
        Some(json!({"minimum":liquid_minimum,"maximum":liquid_maximum}))
    } else {
        None
    };
    let represented_bounds = if represented_cell_count != 0 {
        Some(json!({"minimum":represented_minimum,"maximum":represented_maximum}))
    } else {
        None
    };
    let liquid_centroid = (liquid_measure > 0.0).then(|| {
        [
            liquid_weighted[0] / liquid_measure,
            liquid_weighted[1] / liquid_measure,
        ]
    });
    let liquid_mean_velocity = (liquid_measure > 0.0).then(|| {
        [
            liquid_velocity_weighted[0] / liquid_measure,
            liquid_velocity_weighted[1] / liquid_measure,
        ]
    });
    let mut liquid_velocity_covariance = [[0.0_f64; 2]; 2];
    if let Some(mean) = liquid_mean_velocity {
        for a in 0..2 {
            for b in 0..2 {
                liquid_velocity_covariance[a][b] =
                    liquid_velocity_second_moment[a][b] / liquid_measure - mean[a] * mean[b];
            }
        }
    }
    let mut liquid_covariance = [[0.0_f64; 2]; 2];
    if let Some(centroid) = liquid_centroid {
        for cell in &world.state.topology.graph.cells {
            let volume = world.state.fields.density[cell.id as usize] as f64 * cell.measure as f64;
            let delta = [
                cell.center[0] as f64 - centroid[0],
                cell.center[1] as f64 - centroid[1],
            ];
            for a in 0..2 {
                for b in 0..2 {
                    liquid_covariance[a][b] += volume * delta[a] * delta[b] / liquid_measure;
                }
                liquid_covariance[a][a] += volume
                    * (cell.widths[a] as f64 * cell.widths[a] as f64 / 12.0)
                    / liquid_measure;
            }
        }
    }
    let width_counts: std::collections::BTreeMap<_, _> = cells_by_width
        .into_iter()
        .map(|(width, values)| {
            (
                width,
                json!({
                    "cells": values[0] as usize,
                    "wetCells": values[1] as usize,
                    "fullCells": values[2] as usize,
                    "partialCells": values[3] as usize,
                    "liquidMeasure": values[4],
                    "fullLiquidMeasure": values[5],
                    "partialLiquidMeasure": values[6],
                }),
            )
        })
        .collect();
    let mixed_seam_count = world
        .state
        .topology
        .graph
        .rows
        .iter()
        .filter(|row| row.kind == fluid_core::RowKind::MixedSeam)
        .count();
    let wet_mixed_seam_count = world
        .state
        .topology
        .graph
        .rows
        .iter()
        .filter(|row| row.kind == fluid_core::RowKind::MixedSeam)
        .filter(|row| {
            row.terms
                .iter()
                .any(|term| world.state.fields.density[term.cell_id as usize] > 1.0e-5)
        })
        .count();
    json!({
        "cellCount": world.state.topology.graph.cells.len(),
        "rowCount": world.state.topology.graph.rows.len(),
        "subfaceCount": world.state.topology.graph.subfaces.len(),
        "densityRange": density_range,
        "liquidCellCount": liquid_cell_count,
        "liquidBounds": liquid_bounds,
        "representedBounds": represented_bounds,
        "liquidCentroid": liquid_centroid,
        "liquidCovariance": liquid_covariance,
        "liquidMeanVelocity": liquid_mean_velocity,
        "liquidVelocityCovariance": liquid_velocity_covariance,
        "liquidMeasureByFineY": liquid_measure_by_fine_y,
        "cellWidths": cell_widths,
        "cellsByWidth": width_counts,
        "mixedSeamCount": mixed_seam_count,
        "wetMixedSeamCount": wet_mixed_seam_count,
        "receipt": world.receipt(),
        "resolution": world.resolution_receipt,
    })
}
fn capture_velocity_population(
    graph: &fluid_core::Graph,
    fields: &fluid_core::Fields,
    population: u8,
) -> serde_json::Value {
    let mut measure = 0.0_f64;
    let mut weighted = [0.0_f64; 2];
    let mut second = [0.0_f64; 2];
    let mut minimum = [f64::INFINITY; 2];
    let mut maximum = [f64::NEG_INFINITY; 2];
    let mut cells = 0_usize;
    for cell in &graph.cells {
        let id = cell.id as usize;
        let density = fields.density[id] as f64;
        let capacity = (fields.capacity[id] as f64).max(1.0e-6);
        let fill = density / capacity;
        let selected = match population {
            1 => fill >= 1.0 - 1.0e-5,
            2 => density > 1.0e-5 && fill < 1.0 - 1.0e-5,
            _ => density > 1.0e-5,
        };
        if !selected {
            continue;
        }
        let volume = density * cell.measure as f64;
        cells += 1;
        measure += volume;
        for axis in 0..2 {
            let velocity = fields.cell_velocity[2 * id + axis] as f64;
            weighted[axis] += volume * velocity;
            second[axis] += volume * velocity * velocity;
            minimum[axis] = minimum[axis].min(velocity);
            maximum[axis] = maximum[axis].max(velocity);
        }
    }
    if measure == 0.0 {
        return json!({"cells":0,"liquidMeasure":0.0});
    }
    let mean = [weighted[0] / measure, weighted[1] / measure];
    let rms_deviation = [
        (second[0] / measure - mean[0] * mean[0]).max(0.0).sqrt(),
        (second[1] / measure - mean[1] * mean[1]).max(0.0).sqrt(),
    ];
    json!({
        "cells": cells,
        "liquidMeasure": measure,
        "mean": mean,
        "rmsDeviation": rms_deviation,
        "minimum": minimum,
        "maximum": maximum,
    })
}
fn capture_row_state(
    graph: &fluid_core::Graph,
    fields: &fluid_core::Fields,
    row_id: u32,
) -> serde_json::Value {
    let row = &graph.rows[row_id as usize];
    let cells: Vec<_> = row
        .terms
        .iter()
        .map(|term| {
            let cell = term.cell_id as usize;
            json!({
                "id": cell,
                "coefficient": term.coefficient,
                "geometry": graph.cells[cell],
                "density": fields.density[cell],
                "capacity": fields.capacity[cell],
                "pressureMember": fields.pressure_member.get(cell).copied().unwrap_or(0),
                "velocity": &fields.cell_velocity[2 * cell..2 * cell + 2],
            })
        })
        .collect();
    json!({"row":row,"faceVelocity":fields.face_velocity[row_id as usize],"cells":cells})
}
fn capture_stage_velocity(
    name: &str,
    graph: &fluid_core::Graph,
    fields: &fluid_core::Fields,
) -> serde_json::Value {
    let mut maximum_normalized_divergence = 0.0_f64;
    let mut maximum_full_liquid_normalized_divergence = 0.0_f64;
    let mut maximum_absolute_pressure_rhs = 0.0_f64;
    let mut maximum_divergence_cell = 0_u32;
    let mut maximum_rhs_cell = 0_u32;
    for cell in &graph.cells {
        let id = cell.id as usize;
        if fields.pressure_member.get(id).copied().unwrap_or(0) == 0 {
            continue;
        }
        let mut divergence = 0.0_f64;
        for &row_id in &graph.incidences[id] {
            let row = &graph.rows[row_id as usize];
            if fields.pressure_row_member.len() == graph.rows.len()
                && fields.pressure_row_member[row.id as usize] == 0
            {
                continue;
            }
            let Some(own) = row.terms.iter().find(|term| term.cell_id as usize == id) else {
                continue;
            };
            let stored = fields
                .face_velocity
                .get(row.id as usize)
                .copied()
                .unwrap_or(0.0) as f64;
            let fluid = stored - (1.0 - row.open_fraction as f64) * row.solid_velocity as f64;
            divergence += own.coefficient as f64
                * row.static_dual_weight.unwrap_or(row.dual_weight) as f64
                * fluid;
        }
        let normalized_divergence = divergence.abs() / (cell.measure as f64).max(1.0e-12);
        if normalized_divergence > maximum_normalized_divergence {
            maximum_normalized_divergence = normalized_divergence;
            maximum_divergence_cell = cell.id;
        }
        let fill = fields.density[id] as f64 / (fields.capacity[id] as f64).max(1.0e-6);
        if fill >= 1.0 - 1.0e-5 {
            maximum_full_liquid_normalized_divergence = maximum_full_liquid_normalized_divergence
                .max(divergence.abs() / (cell.measure as f64).max(1.0e-12));
        }
        let absolute_rhs = fields.pressure_rhs.get(id).copied().unwrap_or(0.0).abs() as f64;
        if absolute_rhs > maximum_absolute_pressure_rhs {
            maximum_absolute_pressure_rhs = absolute_rhs;
            maximum_rhs_cell = cell.id;
        }
    }
    let mut face_measure = [[0.0_f64; 3]; 2];
    let mut face_weighted = [[0.0_f64; 3]; 2];
    let mut face_second = [[0.0_f64; 3]; 2];
    let mut face_minimum = [[f64::INFINITY; 3]; 2];
    let mut face_maximum = [[f64::NEG_INFINITY; 3]; 2];
    let mut face_minimum_row = [[0_u32; 3]; 2];
    let mut face_maximum_row = [[0_u32; 3]; 2];
    let mut face_count = [[0_usize; 3]; 2];
    for row in &graph.rows {
        let axis = row.axis as usize;
        if axis >= 2 || row.open_fraction <= 1.0e-6 {
            continue;
        }
        let full_pressure_incident = row.terms.iter().any(|term| {
            let id = term.cell_id as usize;
            let density = fields.density[id] as f64;
            let capacity = (fields.capacity[id] as f64).max(1.0e-6);
            fields.pressure_member.get(id).copied().unwrap_or(0) != 0
                && density / capacity >= 1.0 - 1.0e-5
        });
        if !full_pressure_incident {
            continue;
        }
        let velocity = (fields.face_velocity[row.id as usize] as f64
            - (1.0 - row.open_fraction as f64) * row.solid_velocity as f64)
            / row.open_fraction as f64;
        let weight = row.measure as f64;
        let kind = match row.kind {
            fluid_core::RowKind::MixedSeam => 1,
            fluid_core::RowKind::SparseAir => 2,
            _ => 0,
        };
        face_count[axis][kind] += 1;
        face_measure[axis][kind] += weight;
        face_weighted[axis][kind] += weight * velocity;
        face_second[axis][kind] += weight * velocity * velocity;
        if velocity < face_minimum[axis][kind] {
            face_minimum[axis][kind] = velocity;
            face_minimum_row[axis][kind] = row.id;
        }
        if velocity > face_maximum[axis][kind] {
            face_maximum[axis][kind] = velocity;
            face_maximum_row[axis][kind] = row.id;
        }
    }
    let incident_faces: Vec<_> = (0..2)
        .flat_map(|axis| (0..3).map(move |kind| (axis, kind)))
        .map(|(axis, kind)| {
            let measure = face_measure[axis][kind];
            let kind_name = ["same-level", "mixed-seam", "sparse-air"][kind];
            if measure == 0.0 {
                return json!({
                    "axis": axis,
                    "kind": kind_name,
                    "count": 0,
                });
            }
            let mean = face_weighted[axis][kind] / measure;
            json!({
                "axis": axis,
                "kind": kind_name,
                "count": face_count[axis][kind],
                "mean": mean,
                "rmsDeviation": (face_second[axis][kind] / measure - mean * mean).max(0.0).sqrt(),
                "minimum": face_minimum[axis][kind],
                "minimumRow": face_minimum_row[axis][kind],
                "maximum": face_maximum[axis][kind],
                "maximumRow": face_maximum_row[axis][kind],
                "maximumRowState": capture_row_state(graph, fields, face_maximum_row[axis][kind]),
            })
        })
        .collect();
    json!({
        "name": name,
        "topologyGeneration": graph.topology_generation,
        "pressure": {
            "maximumNormalizedDivergence": maximum_normalized_divergence,
            "maximumDivergenceCell": maximum_divergence_cell,
            "maximumFullLiquidNormalizedDivergence": maximum_full_liquid_normalized_divergence,
            "maximumAbsoluteRhs": maximum_absolute_pressure_rhs,
            "maximumRhsCell": maximum_rhs_cell,
            "fullLiquidIncidentFaces": incident_faces,
        },
        "allLiquid": capture_velocity_population(graph, fields, 0),
        "fullLiquid": capture_velocity_population(graph, fields, 1),
        "partialLiquid": capture_velocity_population(graph, fields, 2),
    })
}
fn require_cellwise_commit(world: &World, frame: u32) -> Result<(), String> {
    let Some(receipt) = world.cellwise_remap_receipt.as_ref() else {
        return Err(format!("frame {frame}: no cellwise remap receipt"));
    };
    if let Some(fault) = world.state.fields.fault.as_ref() {
        return Err(format!("frame {frame}: numerical fault {fault:?}"));
    }
    if !receipt.material_committed
        || !receipt.closure_accepted
        || receipt.pregeometry_certificate_violations != 0
        || receipt.traces == 0
        || receipt.gather_clips == 0
        || receipt.postgeometry_receivers_outside_pregeometry_band != 0
        || receipt.negative_liquid_receiver_preimages != 0
        || receipt.pre_correction_liquid_receiver_folds != 0
        || receipt.corrected_liquid_receiver_folds != 0
        || receipt.pre_correction_convex_hull_liquid_folds != 0
        || receipt.corrected_convex_hull_liquid_folds != 0
        || receipt.adaptive_edge_refinement_exhausted
        || receipt.gather_absolute_volume_error > receipt.gather_volume_roundoff_bound
        || receipt.gather_worst_donor_volume_error > receipt.gather_worst_donor_roundoff_bound
        || receipt.support_extrapolation_samples != 0
        || receipt.trace_velocity_fallback_samples != 0
        || receipt.area_balance_relative_error > 1.0e-12
        || receipt.max_area_identity_error > 1.0e-10
        || receipt.closure_unresolved != 0
        || receipt.max_compression_ratio > 8.0 * f32::EPSILON as f64
    {
        return Err(format!(
            "frame {frame}: cellwise material commit rejected: materialCommitted={}, closureAccepted={}, certificateViolations={}, traces={}, gatherClips={}, receiversOutsideBand={}, negativeLiquidPreimages={}, liquidFolds={}/{}, receiverBandFolds={}/{}, bboxLiquidFolds={}/{}, convexHullLiquidFolds={}/{}, refinementExhausted={}, gatherError={}/{}, worstDonor={:?}:{}/{}, supportExtrapolationSamples={}, traceVelocityFallbackSamples={}, areaBalance={}, areaIdentity={}, closureUnresolved={}, compression={}",
            receipt.material_committed,
            receipt.closure_accepted,
            receipt.pregeometry_certificate_violations,
            receipt.traces,
            receipt.gather_clips,
            receipt.postgeometry_receivers_outside_pregeometry_band,
            receipt.negative_liquid_receiver_preimages,
            receipt.pre_correction_liquid_receiver_folds,
            receipt.corrected_liquid_receiver_folds,
            receipt.pre_correction_receiver_band_folds,
            receipt.corrected_receiver_band_folds,
            receipt.pre_correction_bbox_liquid_folds,
            receipt.corrected_bbox_liquid_folds,
            receipt.pre_correction_convex_hull_liquid_folds,
            receipt.corrected_convex_hull_liquid_folds,
            receipt.adaptive_edge_refinement_exhausted,
            receipt.gather_absolute_volume_error,
            receipt.gather_volume_roundoff_bound,
            receipt.gather_worst_donor,
            receipt.gather_worst_donor_volume_error,
            receipt.gather_worst_donor_roundoff_bound,
            receipt.support_extrapolation_samples,
            receipt.trace_velocity_fallback_samples,
            receipt.area_balance_relative_error,
            receipt.max_area_identity_error,
            receipt.closure_unresolved,
            receipt.max_compression_ratio,
        ));
    }
    if world.microsteps != 0 {
        return Err(format!(
            "frame {frame}: baseline transport ran {} microsteps",
            world.microsteps
        ));
    }
    if let Some((cell, density)) = world
        .state
        .fields
        .density
        .iter()
        .copied()
        .enumerate()
        .find(|(_, density)| !density.is_finite() || !(-1.0e-6..=1.000001).contains(density))
    {
        return Err(format!(
            "frame {frame}: cell {cell} has unbounded density {density}"
        ));
    }
    if world
        .state
        .fields
        .face_velocity
        .iter()
        .any(|velocity| !velocity.is_finite())
    {
        return Err(format!("frame {frame}: face velocity is non-finite"));
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    let input: Input = serde_json::from_str(&text)?;
    let mut world =
        World::from_document(input.scene, input.production_options, input.world_options)?;
    let mut output = vec![if input.receipts_only {
        capture_receipt(&world)
    } else {
        capture(&world)
    }];
    let mut stages = Vec::new();
    let mut failure = None;
    for frame in 0..input.frames {
        if frame + 1 == input.frames {
            if let Some(experiment) = input.final_frame_transport_experiment {
                world.options.transport_experiment = experiment;
            }
        }
        let mut observed = Vec::new();
        let advance = if input.receipts_only && input.observe_stage_metrics {
            world.advance_with_observer(frame + 1, world.timestep_s, |name, graph, fields| {
                observed.push(capture_stage_velocity(name, graph, fields));
            })
        } else if input.receipts_only {
            world.advance(frame + 1, world.timestep_s)
        } else {
            world.advance_with_observer(frame + 1, world.timestep_s, |name, graph, fields| {
                observed.push(json!({"name":name,"graph":graph,"fields":fields}));
            })
        };
        if let Err(error) = advance {
            if !input.capture_failure {
                return Err(error.into());
            }
            failure = Some(json!({"frame":frame + 1,"error":error.to_string()}));
            stages.push(observed);
            output.push(capture(&world));
            break;
        }
        if input.require_cellwise_commit {
            if let Err(error) = require_cellwise_commit(&world, frame + 1) {
                if !input.capture_failure {
                    return Err(error.into());
                }
                failure = Some(json!({"frame":frame + 1,"error":error}));
                stages.push(observed);
                output.push(capture(&world));
                break;
            }
        }
        stages.push(observed);
        output.push(if input.receipts_only
            && !(input.capture_final_state && frame + 1 == input.frames)
        {
            capture_receipt(&world)
        } else {
            capture(&world)
        });
    }
    println!(
        "{}",
        serde_json::to_string(&json!({"frames":output,"stages":stages,"failure":failure}))?
    );
    Ok(())
}
