//! Independent contracts for the authored-geometry level-set initializer.

use fluid_core::initial_liquid::initial_liquid_surface_scalar;
use fluid_core::initial_scene::{lattice_dimensions, InitialLiquidHeightField, InitialLiquidVolume, SceneDocument};
use fluid_core::levelset_surface;
use fluid_core::scene_model::Vec3;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Golden {
    cases: Vec<GoldenCase>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoldenCase {
    scene: SceneDocument,
}

fn fixture(id: &str) -> SceneDocument {
    let golden: Golden = serde_json::from_str(include_str!(
        "../../../core/testdata/initial-liquid-golden.json"
    ))
    .unwrap();
    golden.cases.into_iter().find(|case| case.scene.scene_id == id).unwrap().scene
}

fn clean_scene() -> SceneDocument {
    let mut scene = fixture("initial-liquid-all-volume-arms");
    scene.fluid.initial_liquid_volumes.clear();
    scene.fluid.initial_brick_seeds_m = None;
    scene.fluid.initial_brick_seeds_additive = false;
    scene.fluid.initial_height_field = None;
    scene.container.fill_fraction = 0.0;
    scene
}

fn value_at(surface: &fluid_core::presentation::RdfSurface, x: usize, y: usize) -> f32 {
    surface.vertex_phi_fine[x + (surface.dimensions[0] as usize + 1) * y]
}

#[test]
fn cell_cut_tank_surface_is_at_authored_height_and_has_no_wall_contour() {
    let mut scene = clean_scene();
    scene.fluid.initial_condition = "tank-fill".into();
    scene.container.width_m = 1.6;
    scene.container.height_m = 1.2;
    scene.container.depth_m = 0.8;
    scene.container.fill_fraction = 61.0 / 96.0;
    scene.voxel_domain.finest_cell_size_m = 0.05;
    let surface = levelset_surface::initialize_from_document(&scene, 488.0).unwrap();
    assert_eq!(surface.dimensions, [32, 24]);
    for x in 0..=32 {
        assert!((value_at(&surface, x, 15) + 0.25).abs() < 1e-6);
        assert!((value_at(&surface, x, 16) - 0.75).abs() < 1e-6);
    }
    for segment in surface.segments_fine.chunks_exact(4) {
        assert!((segment[1] - 15.25).abs() < 1e-6);
        assert!((segment[3] - 15.25).abs() < 1e-6);
    }
}

#[test]
fn sphere_cylinder_and_box_use_the_production_cell_centre_z_slice() {
    let mut scene = clean_scene();
    let h = scene.voxel_domain.finest_cell_size_m;
    let dims = lattice_dimensions(&scene);
    assert_eq!(dims, [24, 16, 16]);
    // Production selects source z-cell nz/2, whose centre is z=+h/2.
    scene.fluid.initial_liquid_volumes = vec![InitialLiquidVolume::Sphere {
        center_m: Vec3 { x: 0.0, y: 0.4, z: 0.0 }, radius_m: 0.2,
    }];
    let sphere = levelset_surface::initialize_from_document(&scene, 0.0).unwrap();
    assert!((value_at(&sphere, 12, 8) - ((0.5 * h - 0.2) / h) as f32).abs() < 1e-6);

    scene.fluid.initial_liquid_volumes = vec![InitialLiquidVolume::Cylinder {
        center_m: Vec3 { x: 0.0, y: 0.4, z: 0.0 }, radius_m: 0.2, half_height_m: 0.1,
    }];
    let cylinder = levelset_surface::initialize_from_document(&scene, 0.0).unwrap();
    assert!((value_at(&cylinder, 12, 8) + 1.5).abs() < 1e-6);

    scene.fluid.initial_liquid_volumes = vec![InitialLiquidVolume::Box {
        min_m: Vec3 { x: -0.2, y: 0.2, z: -0.2 },
        max_m: Vec3 { x: 0.2, y: 0.6, z: 0.2 },
    }];
    let box_surface = levelset_surface::initialize_from_document(&scene, 0.0).unwrap();
    assert!((value_at(&box_surface, 12, 8) + 3.5).abs() < 1e-6);
}

#[test]
fn height_field_vertices_match_the_authored_function_at_selected_z() {
    let mut scene = clean_scene();
    scene.fluid.initial_height_field = Some(InitialLiquidHeightField::Quadratic {
        base_height_m: 0.17,
        center_m: fluid_core::initial_scene::HeightFieldCenter { x: 0.08, z: -0.06 },
        curvature_x_m_inv: 0.71,
        curvature_z_m_inv: 0.39,
    });
    let dims = lattice_dimensions(&scene);
    let surface = levelset_surface::initialize_from_document(&scene, 0.0).unwrap();
    let x = 7_usize;
    let y = 9_usize;
    let h = scene.voxel_domain.finest_cell_size_m;
    let point = Vec3 {
        x: -0.5 * scene.container.width_m + x as f64 * h,
        y: y as f64 * h,
        z: 0.5 * h,
    };
    let expected = initial_liquid_surface_scalar(&scene, point, dims) / h;
    assert!((value_at(&surface, x, y) as f64 - expected).abs() < 1e-6);
}

#[test]
fn painted_bricks_replace_or_add_to_the_base_liquid_as_authored() {
    let replacement = fixture("initial-liquid-replacement-seeds");
    let additive = fixture("initial-liquid-additive-seeds");
    let dims = lattice_dimensions(&replacement);
    let point = Vec3 { x: -0.1, y: 0.2, z: 0.0 };
    assert!(initial_liquid_surface_scalar(&replacement, point, dims) > 0.0);
    assert!(initial_liquid_surface_scalar(&additive, point, dims) < 0.0);
}

#[test]
fn wall_touching_box_does_not_publish_container_wall_as_free_surface() {
    let mut scene = clean_scene();
    scene.fluid.initial_liquid_volumes = vec![InitialLiquidVolume::Box {
        min_m: Vec3 { x: -0.6, y: 0.0, z: -0.4 },
        max_m: Vec3 { x: -0.2, y: 0.4, z: 0.4 },
    }];
    let surface = levelset_surface::initialize_from_document(&scene, 0.0).unwrap();
    assert!(value_at(&surface, 0, 4) < 0.0);
    assert!(value_at(&surface, 4, 0) < 0.0);
    assert!(surface.segments_fine.chunks_exact(4).all(|segment| {
        !(segment[0] == 0.0 && segment[2] == 0.0)
            && !(segment[1] == 0.0 && segment[3] == 0.0)
    }));
}

#[test]
fn adjacent_painted_bricks_have_no_zero_contour_on_their_internal_face() {
    let mut scene = clean_scene();
    scene.fluid.initial_brick_seeds_m = Some(vec![
        Vec3 { x: -0.5, y: 0.1, z: 0.1 },
        Vec3 { x: -0.1, y: 0.1, z: 0.1 },
    ]);
    let dims = lattice_dimensions(&scene);
    let internal = Vec3 { x: -0.2, y: 0.2, z: 0.025 };
    let scalar = initial_liquid_surface_scalar(&scene, internal, dims);
    assert!((scalar + 0.025).abs() < 1e-12,
        "the union interior must be strictly negative on a shared painted-brick face");
}
