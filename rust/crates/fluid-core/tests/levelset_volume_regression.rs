//! Independent numerical regressions for the level-set-plus-volume lab path.

use fluid_core::geometry::BoundaryMode;
use fluid_core::levelset_surface;
use fluid_core::levelset_volume::advance as advance_levelset_volume;
use fluid_core::presentation::{RdfSupport, RdfSurface, RdfTopology};
use fluid_core::topology::{compile_topology, BrickSeed, TopologySeed};
use fluid_core::transfer::{transfer_fields, transfer_fields_allow_overcapacity};
use fluid_core::{Fields, Graph};

fn brick(key: u32, coordinate: [i32; 3], resolution: u8) -> BrickSeed {
    BrickSeed {
        id: key,
        key,
        coordinate,
        span_bricks: 1,
        resolution,
        active: true,
        density: Vec::new(),
        gamma: Vec::new(),
        refinement_region_scale: None,
    }
}

fn seam_topology() -> Graph {
    compile_topology::<2>(TopologySeed {
        dimensions: [16, 8, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![brick(0, [0, 0, 0], 4), brick(1, [1, 0, 0], 8)],
    })
    .unwrap()
    .graph
}

fn uniform_topology() -> Graph {
    compile_topology::<2>(TopologySeed {
        dimensions: [8, 8, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![brick(0, [0, 0, 0], 8)],
    })
    .unwrap()
    .graph
}

fn fields(graph: &Graph, density: impl Fn(&fluid_core::Cell) -> f32) -> Fields {
    let n = graph.cells.len();
    Fields {
        density: graph.cells.iter().map(density).collect(),
        gamma: vec![1.0; n],
        capacity: vec![1.0; n],
        capacity_before: vec![1.0; n],
        capacity_after: vec![1.0; n],
        cell_velocity: vec![0.0; 2 * n],
        face_velocity: vec![0.0; graph.rows.len()],
        pressure: vec![0.0; n],
        pressure_rhs: vec![0.0; n],
        pressure_diagonal: vec![1.0; n],
        pressure_member: vec![1; n],
        pressure_row_member: vec![1; graph.rows.len()],
        extension_depth: vec![0; n],
        interface_normal: vec![0.0; 2 * n],
        interface_offset: vec![0.0; n],
        ..Fields::default()
    }
}

fn surface(graph: &Graph, fields: &Fields) -> (RdfTopology, RdfSupport, RdfSurface) {
    let topology = RdfTopology::compile(graph).unwrap();
    let support = RdfSupport::default();
    let rdf = levelset_surface::initialize_from_volume(graph, fields).unwrap();
    (topology, support, rdf)
}

fn physical_volume(graph: &Graph, fields: &Fields) -> f64 {
    graph
        .cells
        .iter()
        .map(|cell| fields.density[cell.id as usize] as f64 * cell.measure as f64)
        .sum()
}

#[test]
fn zero_velocity_is_identity_on_mixed_adaptive_cells() {
    let graph = seam_topology();
    let mut fields = fields(&graph, |cell| {
        let x = cell.center[0];
        let y = cell.center[1];
        (0.15 + 0.03 * x + 0.02 * y).clamp(0.0, 0.9)
    });
    let (rdf_topology, rdf_support, previous_surface) = surface(&graph, &mut fields);
    let before = fields.density.clone();
    let mut phi = levelset_surface::cell_phi(&graph, &previous_surface).unwrap();

    let (_, receipt) = advance_levelset_volume(
        &graph,
        &mut fields,
        &previous_surface,
        &rdf_topology,
        &rdf_support,
        &mut phi,
        1.0 / 30.0,
    )
    .unwrap();

    for (actual, expected) in fields.density.iter().zip(before) {
        assert!((actual - expected).abs() <= 2.0e-6, "{actual} != {expected}");
    }
    assert!(receipt.absolute_volume_drift <= receipt.volume_roundoff_bound);
    assert!(receipt.maximum_donor_residual <= 1.0e-12);
}

#[test]
fn high_courant_uncovered_donors_remain_conservative() {
    let graph = uniform_topology();
    let mut fields = fields(&graph, |cell| {
        if cell.center[0] < 3.0 && (2.0..6.0).contains(&cell.center[1]) {
            0.8
        } else {
            0.0
        }
    });
    for row in &graph.rows {
        fields.face_velocity[row.id as usize] = if row.axis == 0 { 20.0 } else { 0.0 };
    }
    let (rdf_topology, rdf_support, previous_surface) = surface(&graph, &mut fields);
    let before = physical_volume(&graph, &fields);
    let mut phi = levelset_surface::cell_phi(&graph, &previous_surface).unwrap();

    let (_, receipt) = advance_levelset_volume(
        &graph,
        &mut fields,
        &previous_surface,
        &rdf_topology,
        &rdf_support,
        &mut phi,
        0.5,
    )
    .unwrap();

    let after = physical_volume(&graph, &fields);
    assert!(receipt.maximum_trace_courant >= 5.0, "{receipt:?}");
    assert!(receipt.zero_weight_donors > 0, "missing-donor path was not exercised");
    assert!((after - before).abs() <= receipt.volume_roundoff_bound, "{receipt:?}");
    assert!(fields.density.iter().all(|value| value.is_finite() && *value >= 0.0));
}

#[test]
fn positive_liquid_in_zero_capacity_cell_is_rejected_without_mutation() {
    let graph = uniform_topology();
    let mut fields = fields(&graph, |_| 0.0);
    fields.density[0] = 0.25;
    fields.capacity[0] = 0.0;
    let before = fields.density.clone();
    let rdf_topology = RdfTopology::compile(&graph).unwrap();
    let rdf_support = RdfSupport::default();
    let previous_surface = RdfSurface {
        dimensions: [8, 8],
        vertex_phi_fine: vec![1.0; 9 * 9],
        segments_fine: Vec::new(),
        receipt: Default::default(),
    };
    let mut phi = vec![1.0; graph.cells.len()];

    let error = advance_levelset_volume(
        &graph,
        &mut fields,
        &previous_surface,
        &rdf_topology,
        &rdf_support,
        &mut phi,
        1.0 / 30.0,
    )
    .unwrap_err();

    assert!(error.0.contains("zero open capacity"), "{error}");
    assert_eq!(fields.density, before);
}

#[test]
fn direct_surface_rejects_nonfinite_shared_scalar_instead_of_using_volume_phase() {
    let graph = uniform_topology();
    let fields = fields(&graph, |cell| if cell.center[0] < 4.0 { 1.0 } else { 0.0 });
    assert!(levelset_surface::publish([8, 8], vec![f32::NAN; 9 * 9],
        physical_volume(&graph, &fields)).is_err());
}

#[test]
fn affine_shared_phi_has_one_zero_contour_across_two_to_one_subfaces() {
    let graph = seam_topology();
    let interface_y = 4.25_f32;
    let vertices = (0..=8)
        .flat_map(|y| (0..=16).map(move |_| y as f32 - interface_y))
        .collect();
    let surface = levelset_surface::publish([16, 8], vertices, 0.0).unwrap();
    assert!(!surface.segments_fine.is_empty());
    assert!(surface.segments_fine.chunks_exact(4).all(|segment| {
        (segment[1] - interface_y).abs() <= 1e-6
            && (segment[3] - interface_y).abs() <= 1e-6
    }));
    let cell_phi = levelset_surface::cell_phi(&graph, &surface).unwrap();
    for cell in &graph.cells {
        assert!((cell_phi[cell.id as usize] - (cell.center[1] - interface_y)).abs() <= 1e-6);
    }
}

#[test]
fn topology_transfer_preserves_overcapacity_only_for_level_set_mode() {
    let source = compile_topology::<2>(TopologySeed {
        dimensions: [8, 8, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![brick(0, [0, 0, 0], 4)],
    })
    .unwrap()
    .graph;
    let target = compile_topology::<2>(TopologySeed {
        dimensions: [8, 8, 1],
        generation: 2,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![brick(0, [0, 0, 0], 8)],
    })
    .unwrap()
    .graph;
    let mut source_fields = fields(&source, |_| 0.0);
    source_fields.density[5] = 1.375;
    let before = physical_volume(&source, &source_fields);
    let target_capacity = vec![1.0; target.cells.len()];

    assert!(transfer_fields(&source, &target, &source_fields, &target_capacity, &[]).is_err());
    let moved = transfer_fields_allow_overcapacity(
        &source,
        &target,
        &source_fields,
        &target_capacity,
        &[],
    )
    .unwrap();
    let after: f64 = target
        .cells
        .iter()
        .map(|cell| moved.density[cell.id as usize] as f64 * cell.measure as f64)
        .sum();
    assert!((after - before).abs() <= 2.0 * f32::EPSILON as f64 * before.max(1.0));
    assert!(moved.density.iter().any(|&density| density > 1.0));
}

#[test]
fn overcapacity_topology_transfer_keeps_solid_children_empty() {
    let source = compile_topology::<2>(TopologySeed {
        dimensions: [8, 8, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![brick(0, [0, 0, 0], 4)],
    })
    .unwrap()
    .graph;
    let target = compile_topology::<2>(TopologySeed {
        dimensions: [8, 8, 1],
        generation: 2,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![brick(0, [0, 0, 0], 8)],
    })
    .unwrap()
    .graph;
    let mut source_fields = fields(&source, |_| 0.0);
    source_fields.density[0] = 0.25;
    source_fields.interface_normal[0] = 1.0;
    let before = physical_volume(&source, &source_fields);
    let mut target_capacity = vec![1.0; target.cells.len()];
    target_capacity[0] = 0.0;

    let moved = transfer_fields_allow_overcapacity(
        &source,
        &target,
        &source_fields,
        &target_capacity,
        &[],
    )
    .unwrap();
    let after: f64 = target
        .cells
        .iter()
        .map(|cell| moved.density[cell.id as usize] as f64 * cell.measure as f64)
        .sum();

    assert_eq!(moved.density[0], 0.0);
    assert!((after - before).abs() <= 2.0 * f32::EPSILON as f64 * before.max(1.0));
}

#[test]
fn level_set_topology_transfer_uses_phi_plane_for_cut_donor_split() {
    let source = compile_topology::<2>(TopologySeed {
        dimensions: [8, 8, 1], generation: 1, sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![brick(0, [0, 0, 0], 4)],
    }).unwrap().graph;
    let target = compile_topology::<2>(TopologySeed {
        dimensions: [8, 8, 1], generation: 2, sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![brick(0, [0, 0, 0], 8)],
    }).unwrap().graph;
    let mut source_fields = fields(&source, |_| 0.0);
    source_fields.capacity[0] = 0.5;
    source_fields.density[0] = 0.5;
    source_fields.interface_normal[0] = 1.0;
    source_fields.interface_offset[0] = 0.0;
    let before = physical_volume(&source, &source_fields);

    let moved = transfer_fields_allow_overcapacity(
        &source, &target, &source_fields, &vec![1.0; target.cells.len()], &[],
    ).unwrap();
    let children: Vec<_> = target.cells.iter().filter(|cell| {
        cell.minimum[0] < 2.0 && cell.minimum[1] < 2.0
    }).map(|cell| moved.density[cell.id as usize]).collect();
    let after: f64 = target.cells.iter().map(|cell| {
        moved.density[cell.id as usize] as f64 * cell.measure as f64
    }).sum();

    assert_eq!(children.iter().filter(|&&density| density == 0.0).count(), 2);
    assert_eq!(children.iter().filter(|&&density| density == 1.0).count(), 2);
    assert!((after - before).abs() <= 2.0 * f32::EPSILON as f64 * before.max(1.0));
}

#[test]
fn redistance_uses_one_fine_coordinate_metric_across_mixed_widths() {
    let graph = seam_topology();
    let surface = RdfSurface {
        dimensions: [16, 8],
        vertex_phi_fine: (0..=8)
            .flat_map(|_| (0..=16).map(|x| x as f32 - 8.0))
            .collect(),
        segments_fine: vec![8.0, 0.0, 8.0, 8.0],
        receipt: Default::default(),
    };
    let phi = levelset_surface::cell_phi(&graph, &surface).unwrap();
    for cell in &graph.cells {
        let expected = cell.center[0] - 8.0;
        assert!((phi[cell.id as usize] - expected).abs() < 1e-6,
            "width {:?}: {} != {expected}", cell.widths, phi[cell.id as usize]);
    }
}

#[test]
fn contour_empty_finite_scalar_is_preserved_without_volume_phase_fallback() {
    let graph = uniform_topology();
    let surface = levelset_surface::publish([8, 8], vec![2.0; 9 * 9], 17.0).unwrap();
    assert!(surface.segments_fine.is_empty());
    let phi = levelset_surface::cell_phi(&graph, &surface).unwrap();
    assert!(phi.iter().all(|value| *value == 2.0));
}

#[test]
fn identical_shared_phi_with_different_volume_and_poisoned_plic_has_identical_future_surface() {
    let graph = uniform_topology();
    let vertices: Vec<f32> = (0..=8).flat_map(|y| (0..=8).map(move |x| {
        (x as f32 - 4.0).hypot(y as f32 - 4.0) - 2.25
    })).collect();
    let previous = levelset_surface::publish([8, 8], vertices, 0.0).unwrap();
    let topology = RdfTopology::compile(&graph).unwrap();
    let support = RdfSupport::default();
    let mut a = fields(&graph, |cell| if cell.center[0] < 4.0 { 0.75 } else { 0.1 });
    let mut b = fields(&graph, |cell| if cell.center[1] < 4.0 { 0.2 } else { 0.9 });
    for values in [&mut a, &mut b] {
        values.interface_normal.fill(f32::NAN);
        values.interface_offset.fill(f32::NAN);
        for row in &graph.rows {
            values.face_velocity[row.id as usize] = [0.3, -0.2][row.axis as usize];
        }
    }
    let mut phi_a = levelset_surface::cell_phi(&graph, &previous).unwrap();
    let mut phi_b = phi_a.clone();
    let (surface_a, _) = advance_levelset_volume(
        &graph, &mut a, &previous, &topology, &support, &mut phi_a, 0.1,
    ).unwrap();
    let (surface_b, _) = advance_levelset_volume(
        &graph, &mut b, &previous, &topology, &support, &mut phi_b, 0.1,
    ).unwrap();
    assert_eq!(surface_a.vertex_phi_fine, surface_b.vertex_phi_fine);
    assert_eq!(surface_a.segments_fine, surface_b.segments_fine);
    assert_eq!(phi_a, phi_b);
}

#[test]
fn small_translations_cross_fine_coarse_seam_in_both_directions() {
    const DT: f32 = 0.2;
    const PATCH_HEIGHT: f64 = 4.0;
    const EXPECTED_CROSSING: f64 = DT as f64 * PATCH_HEIGHT;

    for &(name, vx, vy) in &[
        ("fine-to-coarse axis", -1.0_f32, 0.0_f32),
        ("fine-to-coarse diagonal", -1.0, 0.5),
        ("coarse-to-fine axis", 1.0, 0.0),
        ("coarse-to-fine diagonal", 1.0, 0.5),
    ] {
        let graph = seam_topology();
        let source_on_fine = vx < 0.0;
        let mut fields = fields(&graph, |cell| {
            let in_x = if source_on_fine {
                cell.minimum[0] >= 8.0 && cell.maximum[0] <= 12.0
            } else {
                cell.minimum[0] >= 4.0 && cell.maximum[0] <= 8.0
            };
            if in_x && cell.minimum[1] >= 2.0 && cell.maximum[1] <= 6.0 { 1.0 } else { 0.0 }
        });
        for row in &graph.rows {
            fields.face_velocity[row.id as usize] = [vx, vy][row.axis as usize];
        }
        let destination_volume = |values: &Fields| -> f64 {
            graph.cells.iter().filter(|cell| if source_on_fine {
                cell.maximum[0] <= 8.0
            } else {
                cell.minimum[0] >= 8.0
            }).map(|cell| values.density[cell.id as usize] as f64 * cell.measure as f64).sum()
        };
        let before_total = physical_volume(&graph, &fields);
        let before_destination = destination_volume(&fields);
        assert_eq!(before_destination, 0.0, "{name}: destination must begin dry");
        let (rdf_topology, rdf_support, previous_surface) = surface(&graph, &mut fields);
        let mut phi = levelset_surface::cell_phi(&graph, &previous_surface).unwrap();

        let (_, receipt) = advance_levelset_volume(
            &graph, &mut fields, &previous_surface, &rdf_topology, &rdf_support, &mut phi, DT,
        ).unwrap();

        let after_total = physical_volume(&graph, &fields);
        let crossing = destination_volume(&fields) - before_destination;
        assert!((after_total - before_total).abs() <= receipt.volume_roundoff_bound,
            "{name}: mass changed: {receipt:?}");
        assert!(fields.density.iter().all(|value| value.is_finite() && *value >= 0.0),
            "{name}: transport produced invalid fill");
        // The patch and seam are remote from the closed domain boundary. The
        // remaining 4.4e-4 worst-case difference is the bounded marginal
        // balancing perturbation, rather than missing cross-rung support.
        assert!((crossing - EXPECTED_CROSSING).abs() <= 5.0e-4,
            "{name}: seam crossing {crossing} did not reproduce geometric overlap {EXPECTED_CROSSING}");
    }
}
