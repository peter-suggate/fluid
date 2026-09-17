use fluid_core::geometry::BoundaryMode;
use fluid_core::presentation::{RdfSupport, RdfTopology};
use fluid_core::topology::{compile_topology, BrickSeed, TopologySeed};
use fluid_core::{levelset_surface, levelset_volume};
use fluid_core::{Fields, Graph};
fn graph(width: u8) -> Graph {
    let bricks = (0..2)
        .flat_map(|y| {
            (0..4).map(move |x| {
                let key = x + 4 * y;
                BrickSeed {
                    id: key,
                    key,
                    coordinate: [x as i32, y as i32, 0],
                    span_bricks: 1,
                    resolution: 8 / width,
                    active: true,
                    density: vec![],
                    gamma: vec![],
                    refinement_region_scale: None,
                }
            })
        })
        .collect();
    compile_topology::<2>(TopologySeed {
        dimensions: [32, 16, 1],
        generation: 1,
        sparse_air_phi: 0.5,
        boundaries: [BoundaryMode::Closed; 6],
        bricks,
    })
    .unwrap()
    .graph
}
fn main() {
    for width in [1, 2, 4] {
        let g = graph(width);
        let n = g.cells.len();
        // Slab is separated from every moving-direction wall. Exact phi and V agree initially.
        let vertices = (0..=16)
            .flat_map(|y| (0..=32).map(move |_| (4.25 - y as f32).max(y as f32 - 8.25)))
            .collect();
        let surface = levelset_surface::publish([32, 16], vertices, 128.0).unwrap();
        let fill = levelset_surface::implied_fill_fine_cells(&surface).unwrap();
        let rho = g
            .cells
            .iter()
            .map(|c| {
                let mut v = 0.0;
                for y in c.minimum[1] as usize..c.maximum[1] as usize {
                    for x in c.minimum[0] as usize..c.maximum[0] as usize {
                        v += fill[x + 32 * y];
                    }
                }
                v / c.measure
            })
            .collect();
        let mut f = Fields {
            density: rho,
            gamma: vec![1.0; n],
            capacity: vec![1.0; n],
            capacity_before: vec![1.0; n],
            capacity_after: vec![1.0; n],
            cell_velocity: vec![0.0; 2 * n],
            face_velocity: g
                .rows
                .iter()
                .map(|r| if r.axis == 1 { 0.1 } else { 0.0 })
                .collect(),
            pressure: vec![0.0; n],
            pressure_rhs: vec![0.0; n],
            pressure_diagonal: vec![1.0; n],
            pressure_member: vec![1; n],
            pressure_row_member: vec![1; g.rows.len()],
            extension_depth: vec![0; n],
            interface_normal: vec![0.0; 2 * n],
            interface_offset: vec![0.0; n],
            ..Default::default()
        };
        let topo = RdfTopology::compile(&g).unwrap();
        let mut phi = levelset_surface::cell_phi(&g, &surface).unwrap();
        eprintln!("CASE width={width}");
        let (_, r) = levelset_volume::advance(
            &g,
            &mut f,
            &surface,
            &topo,
            &RdfSupport::default(),
            &mut phi,
            1.0,
        )
        .unwrap();
        println!("{}", serde_json::json!({"width":width,"receipt":r}));
    }
}
