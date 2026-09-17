//! Independent invariants for conservative volume sharpening around fixed phi.

use fluid_core::geometry::BoundaryMode;
use fluid_core::levelset_sharpening::sharpen_volume;
use fluid_core::levelset_surface::{self, implied_fill_fine_cells};
use fluid_core::levelset_volume::publish_pressure_geometry_from_phi;
use fluid_core::presentation::RdfSurface;
use fluid_core::topology::{compile_topology, BrickSeed, TopologySeed};
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

fn graph(mixed: bool) -> Graph {
    compile_topology::<2>(TopologySeed {
        dimensions: [16, 8, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks: vec![
            brick(0, [0, 0, 0], if mixed { 4 } else { 8 }),
            brick(1, [1, 0, 0], 8),
        ],
    })
    .unwrap()
    .graph
}

fn surface(function: impl Fn(f32, f32) -> f32) -> RdfSurface {
    let function = &function;
    let vertices = (0..=8)
        .flat_map(|y| (0..=16).map(move |x| function(x as f32, y as f32)))
        .collect();
    levelset_surface::publish([16, 8], vertices, 0.0).unwrap()
}

fn target_fields(
    graph: &Graph,
    surface: &RdfSurface,
    fine_capacity: &[f32],
) -> (Fields, Vec<f64>) {
    let fill = implied_fill_fine_cells(surface).unwrap();
    let mut capacity = vec![0.0; graph.cells.len()];
    let mut density = vec![0.0; graph.cells.len()];
    let mut volume = vec![0.0; graph.cells.len()];
    for cell in &graph.cells {
        let id = cell.id as usize;
        let mut open = 0.0_f64;
        let mut liquid = 0.0_f64;
        for y in cell.minimum[1] as usize..cell.maximum[1] as usize {
            for x in cell.minimum[0] as usize..cell.maximum[0] as usize {
                let index = x + 16 * y;
                open += fine_capacity[index] as f64;
                liquid += fine_capacity[index] as f64 * fill[index] as f64;
            }
        }
        capacity[id] = (open / cell.measure as f64) as f32;
        density[id] = (liquid / cell.measure as f64) as f32;
        volume[id] = liquid;
    }
    let n = graph.cells.len();
    let mut fields = Fields {
            density,
            gamma: vec![1.0; n],
            capacity: capacity.clone(),
            capacity_before: capacity.clone(),
            capacity_after: capacity,
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
        };
    let phi = levelset_surface::cell_phi(graph, surface).unwrap();
    publish_pressure_geometry_from_phi(graph, &mut fields, &phi).unwrap();
    (fields, volume)
}

fn cell_at(graph: &Graph, point: [f32; 2]) -> usize {
    graph
        .cells
        .iter()
        .find(|cell| {
            cell.minimum[0] <= point[0]
                && point[0] < cell.maximum[0]
                && cell.minimum[1] <= point[1]
                && point[1] < cell.maximum[1]
        })
        .unwrap()
        .id as usize
}

fn total(values: &[f64]) -> f64 {
    values.iter().sum()
}

#[test]
fn open_weighted_target_is_a_fixed_point_across_terrain_and_two_to_one_seam() {
    let graph = graph(true);
    let surface = surface(|_, y| y - 4.25);
    let fine_capacity: Vec<_> = (0..8)
        .flat_map(|y| {
            (0..16).map(move |x| {
                if y as f32 + 0.5 > 0.18 * x as f32 {
                    1.0
                } else {
                    0.0
                }
            })
        })
        .collect();
    let (fields, mut volume) = target_fields(&graph, &surface, &fine_capacity);
    let before = volume.clone();
    let vertices = surface.vertex_phi_fine.clone();
    let segments = surface.segments_fine.clone();
    let phi = levelset_surface::cell_phi(&graph, &surface).unwrap();

    let receipt = sharpen_volume(
        &graph,
        &fields,
        &fine_capacity,
        &surface,
        &phi,
        &mut volume,
    )
    .unwrap();

    assert_eq!(surface.vertex_phi_fine, vertices);
    assert_eq!(surface.segments_fine, segments);
    assert_eq!(volume, before);
    assert_eq!(receipt.relocated_volume, 0.0);
    assert!(receipt.initial_band_absolute_mismatch <= 1.0e-6, "{receipt:?}");
    assert!(receipt.global_conservation_residual <= 1.0e-12, "{receipt:?}");
    assert_eq!(receipt.bound_violation_count, 0);
}

#[test]
fn diffuse_volume_moves_toward_one_contour_without_cross_component_transfer() {
    let graph = graph(false);
    let surface = surface(|x, y| {
        let left = (x - 4.0).hypot(y - 4.0) - 2.5;
        let right = (x - 12.0).hypot(y - 4.0) - 2.5;
        left.min(right)
    });
    let fine_capacity = vec![1.0; 16 * 8];
    let (fields, mut volume) = target_fields(&graph, &surface, &fine_capacity);
    let donor = cell_at(&graph, [7.5, 4.5]);
    let receiver = cell_at(&graph, [5.5, 4.5]);
    let moved = 0.6;
    volume[donor] += moved;
    volume[receiver] -= moved;
    let before = volume.clone();
    let mut poisoned_fields = fields.clone();
    poisoned_fields.interface_normal.fill(0.0);
    for normal in poisoned_fields.interface_normal.chunks_exact_mut(2) {
        normal[1] = 1.0;
    }
    let mut poisoned_volume = before.clone();
    let before_total = total(&volume);
    let right_before: Vec<_> = graph
        .cells
        .iter()
        .filter(|cell| cell.center[0] >= 9.0)
        .map(|cell| (cell.id as usize, volume[cell.id as usize]))
        .collect();
    let vertices = surface.vertex_phi_fine.clone();
    let segments = surface.segments_fine.clone();
    let phi = levelset_surface::cell_phi(&graph, &surface).unwrap();

    let receipt = sharpen_volume(
        &graph,
        &fields,
        &fine_capacity,
        &surface,
        &phi,
        &mut volume,
    )
    .unwrap();
    let poisoned_receipt = sharpen_volume(
        &graph,
        &poisoned_fields,
        &fine_capacity,
        &surface,
        &phi,
        &mut poisoned_volume,
    )
    .unwrap();

    assert_eq!(surface.vertex_phi_fine, vertices);
    assert_eq!(surface.segments_fine, segments);
    assert_eq!(receipt.component_count, 2, "{receipt:?}");
    assert_eq!(receipt.cross_component_pair_count, 0);
    assert!(receipt.relocated_volume > 0.0, "{receipt:?}");
    assert!(volume[donor] < before[donor]);
    assert!(volume[receiver] > before[receiver]);
    assert!(receipt.final_band_absolute_mismatch < receipt.initial_band_absolute_mismatch);
    assert!(receipt.final_distance_weighted_mismatch < receipt.initial_distance_weighted_mismatch);
    assert!((total(&volume) - before_total).abs() <= 1.0e-12, "{receipt:?}");
    assert!(right_before
        .into_iter()
        .all(|(id, expected)| volume[id] == expected));
    assert_eq!(receipt.bound_violation_count, 0);
    assert_eq!(poisoned_volume, volume, "sharpening must derive direction from current phi");
    assert_eq!(poisoned_receipt.relocated_volume, receipt.relocated_volume);
}

#[test]
fn subcell_diffuse_island_is_retained_when_it_does_not_pass_the_gate() {
    let graph = graph(false);
    let surface = surface(|x, y| (x - 4.0).hypot(y - 4.0) - 2.5);
    let fine_capacity = vec![1.0; 16 * 8];
    let (fields, mut volume) = target_fields(&graph, &surface, &fine_capacity);
    let donor = cell_at(&graph, [7.5, 4.5]);
    let receiver = cell_at(&graph, [5.5, 4.5]);
    volume[donor] += 0.25;
    volume[receiver] -= 0.25;
    let before = volume.clone();
    let phi = levelset_surface::cell_phi(&graph, &surface).unwrap();

    let receipt = sharpen_volume(
        &graph,
        &fields,
        &fine_capacity,
        &surface,
        &phi,
        &mut volume,
    )
    .unwrap();

    assert_eq!(volume, before, "subcell material must not be erased or spread");
    assert_eq!(receipt.relocated_volume, 0.0);
    assert!(receipt.unresolved_eligible_residual >= 0.25 - 1.0e-12, "{receipt:?}");
    assert!(receipt.global_conservation_residual <= 1.0e-12, "{receipt:?}");
}

#[test]
fn fine_contour_component_survives_when_every_cell_centre_is_air() {
    let graph = graph(false);
    let surface = surface(|x, y| (x - 4.0).hypot(y - 4.0) - 0.45);
    let fine_capacity = vec![1.0; 16 * 8];
    let (fields, mut volume) = target_fields(&graph, &surface, &fine_capacity);
    let before = volume.clone();
    let phi = levelset_surface::cell_phi(&graph, &surface).unwrap();
    assert!(phi.iter().all(|value| *value > 0.0));

    let receipt = sharpen_volume(
        &graph,
        &fields,
        &fine_capacity,
        &surface,
        &phi,
        &mut volume,
    )
    .unwrap();

    assert_eq!(receipt.component_count, 1, "{receipt:?}");
    assert_eq!(receipt.ambiguous_cell_count, 0, "{receipt:?}");
    assert_eq!(volume, before);
    assert_eq!(receipt.relocated_volume, 0.0);
}

#[test]
fn adaptive_distance_returns_far_residue_across_a_two_to_one_seam() {
    for mixed in [false, true] {
        let graph = graph(mixed);
        let surface = surface(|x, _| x-6.0);
        let capacity = vec![1.0;128];
        let (fields, mut volume) = target_fields(&graph,&surface,&capacity);
        let donor = cell_at(&graph,[11.5,3.5]);
        let receiver = cell_at(&graph,[5.5,3.5]);
        volume[donor] += 0.75; volume[receiver] -= 0.75;
        let before = total(&volume);
        let phi = levelset_surface::cell_phi(&graph,&surface).unwrap();
        let receipt = sharpen_volume(&graph,&fields,&capacity,&surface,&phi,&mut volume).unwrap();
        assert!(receipt.far_relocated_volume > 0.7, "mixed={mixed}: {receipt:?}");
        assert!(volume[donor] < 0.05);
        assert!((total(&volume)-before).abs()<1e-12);
        assert_eq!(receipt.bound_violation_count,0);
        assert!(receipt.maximum_relocation_distance <= 8.0);
    }
}

#[test]
fn adaptive_return_cannot_cross_a_closed_wall() {
    let mut graph = graph(false);
    let surface = surface(|x,_| x-6.0);
    let capacity: Vec<_> = (0..8).flat_map(|_| (0..16).map(|x| if x==8 {0.0} else {1.0})).collect();
    let (fields, mut volume) = target_fields(&graph,&surface,&capacity);
    let donor = cell_at(&graph,[11.5,3.5]);
    let receiver = cell_at(&graph,[5.5,3.5]);
    volume[donor] += 0.75; volume[receiver] -= 0.75;
    // Also exercise the row-aperture barrier, independently of capacity tests.
    for row in &mut graph.rows { if row.axis==0 && row.center[0]==8.0 {row.open_fraction=0.0;} }
    let before = volume.clone();
    let phi = levelset_surface::cell_phi(&graph,&surface).unwrap();
    let receipt = sharpen_volume(&graph,&fields,&capacity,&surface,&phi,&mut volume).unwrap();
    assert_eq!(volume,before);
    assert!(receipt.unassigned_volume >= 0.75);
}

#[test]
fn equidistant_far_residue_does_not_choose_between_components() {
    let graph = graph(false);
    let surface = surface(|x,_| (x-3.5).min(11.5-x));
    let capacity=vec![1.0;128];
    let (fields,mut volume)=target_fields(&graph,&surface,&capacity);
    let donor=cell_at(&graph,[7.5,3.5]);
    volume[donor]=0.75;
    volume[cell_at(&graph,[2.5,3.5])]-=0.375;
    volume[cell_at(&graph,[12.5,3.5])]-=0.375;
    let before=volume.clone();
    let phi=levelset_surface::cell_phi(&graph,&surface).unwrap();
    let receipt=sharpen_volume(&graph,&fields,&capacity,&surface,&phi,&mut volume).unwrap();
    assert_eq!(volume,before);
    assert!(receipt.ambiguous_cell_count>0);
}

#[test]
fn symmetric_donors_share_limited_receiver_capacity_without_directional_bias() {
    let graph = graph(false);
    let surface = surface(|_, y| y - 4.0);
    let capacity = vec![1.0; 16*8];
    let (fields, mut volume) = target_fields(&graph, &surface, &capacity);
    for cell in &graph.cells {
        if cell.center[0] == 7.5 || cell.center[0] == 8.5 {
            if cell.center[1] == 4.5 {volume[cell.id as usize] = 0.75;}
            if cell.center[1] == 3.5 {volume[cell.id as usize] = 0.5;}
        }
    }
    let initial:f64 = volume.iter().sum();
    let phi = levelset_surface::cell_phi(&graph, &surface).unwrap();
    let receipt = sharpen_volume(&graph, &fields, &capacity, &surface, &phi, &mut volume).unwrap();
    assert!((volume.iter().sum::<f64>()-initial).abs()<1e-12);
    assert_eq!(receipt.bound_violation_count,0);
    assert!((receipt.relocated_volume-1.0).abs()<1e-12);
    for cell in &graph.cells {
        let mirror=graph.cells.iter().find(|c|c.center[0]==16.0-cell.center[0] && c.center[1]==cell.center[1]).unwrap();
        let (a,b)=(volume[cell.id as usize],volume[mirror.id as usize]);
        assert!((a-b).abs()<1e-12,"{:?}: {a} vs reflected {b}",cell.center);
    }
}
