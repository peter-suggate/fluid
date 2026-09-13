//! Method-control normalization at the Rust boundary. Mirrors the authored
//! activity controls; no host-side physics options are compiled in JavaScript.
use crate::{
    production_scene::{ProductionSceneOptions, ProductionTimeStep},
    resolution::ActivityPolicy,
    types::ValidationError,
    world::WorldOptions,
};
use serde_json::{Map, Value};

fn number(v: &Value) -> Option<f64> {
    v.as_f64().filter(|v| v.is_finite())
}
fn numeric_control(v: &Value) -> Option<f64> {
    number(v).or_else(|| v.as_str()?.parse::<f64>().ok().filter(|v| v.is_finite()))
}

pub fn activity_policy(base: &ActivityPolicy, values: &Map<String, Value>) -> ActivityPolicy {
    let mut data = serde_json::to_value(base).unwrap();
    let target = data.as_object_mut().unwrap();
    for &(key, low, high, integer) in &[
        ("energyThreshold", 0.01, 100., false),
        ("curvatureTolerance", 0.02, 2., false),
        ("anticipationSeconds", 0., 2., false),
        ("anticipationRadiusBricks", 1., 6., true),
        ("surfaceQuietEpochs", 1., 32., true),
        ("surfaceDisplacementToleranceCells", 0., 8., false),
        ("surfaceNormalToleranceDegrees", 0., 90., false),
        ("finestTravelCells", 0.05, 8., false),
        ("fourTravelCells", 0., 8., false),
        ("twoTravelCells", 0., 8., false),
        ("thinFeatureCells", 0.25, 8., false),
        ("thinFeatureDensity", 0., 0.5, false),
        ("residencyDensity", 0.00001, 0.5, false),
        ("residencyMassFineCells", 0., 8., false),
        ("surfaceDensityMinimum", 0., 0.49, false),
        ("surfaceDensityMaximum", 0.51, 1., false),
        ("detailTolerance", 0.005, 0.5, false),
        ("frontLookaheadSteps", 1., 32., true),
        ("topologyCadenceSteps", 1., 32., true),
        ("prepareBricksPerFrame", 1., 256., true),
        ("promoteEpochs", 1., 16., true),
        ("demoteEpochs", 1., 32., true),
        ("promoteScore", 0., 1., false),
        ("demoteScore", 0., 1., false),
        ("emergencyScore", 0., 1., false),
    ] {
        if let Some(value) = values.get(key).and_then(number) {
            let value = value.clamp(low, high);
            target.insert(
                key.into(),
                if integer {
                    Value::from(value.round() as u64)
                } else {
                    Value::from(value)
                },
            );
        }
    }
    for key in [
        "activitySignals",
        "coarseFirst",
        "surfaceCoarseningEnabled",
        "freezeTopology",
        "legacyFaceTransportForQa",
    ] {
        if let Some(value) = values.get(key).and_then(Value::as_bool) {
            target.insert(key.into(), Value::from(value));
        }
    }
    // Preserve the JavaScript acronym spelling at the external control boundary.
    if let Some(value) = values
        .get("legacyFaceTransportForQA")
        .and_then(Value::as_bool)
    {
        target.insert("legacyFaceTransportForQa".into(), Value::from(value));
    }
    if let Some(value) = values
        .get("forcedSurfaceResolutionForQA")
        .and_then(Value::as_u64)
    {
        if [1, 2, 4, 8].contains(&value) {
            target.insert("forcedSurfaceResolutionForQa".into(), Value::from(value));
        }
    }
    let mut policy: ActivityPolicy = serde_json::from_value(data).unwrap();
    if let Some(mode) = values.get("selectorMode") {
        let mode = mode.as_str().unwrap_or("coarse-first");
        policy.activity_signals = mode != "surface";
        policy.coarse_first = mode != "surface" && mode != "activity";
    }
    policy.coarse_first &= policy.activity_signals;
    policy.four_travel_cells = policy.four_travel_cells.min(policy.finest_travel_cells);
    policy.two_travel_cells = policy.two_travel_cells.min(policy.four_travel_cells);
    policy.demote_score = policy.demote_score.min(policy.promote_score);
    policy.emergency_score = policy.emergency_score.max(policy.promote_score);
    policy
}

pub fn apply_initial_values(
    production: &mut ProductionSceneOptions,
    world: &mut WorldOptions,
    values: &Value,
) -> Result<(), ValidationError> {
    let values = values
        .as_object()
        .ok_or_else(|| ValidationError("method values must be an object".into()))?;
    if values
        .get("brickFineResolution")
        .and_then(numeric_control)
        .is_some_and(|v| v != 8.)
    {
        return Err(ValidationError(
            "CPU compiled topology requires brickFineResolution=8".into(),
        ));
    }
    if let Some(v) = values.get("maximumMacroSpanBricks") {
        production.atlas.maximum_macro_span_bricks = if v.as_str() == Some("auto") {
            None
        } else {
            let n = numeric_control(v)
                .ok_or_else(|| ValidationError("invalid maximumMacroSpanBricks".into()))?;
            if n < 1. || n > u32::MAX as f64 || n.fract() != 0. || !(n as u32).is_power_of_two() {
                return Err(ValidationError(
                    "maximumMacroSpanBricks must be a positive power of two".into(),
                ));
            }
            Some(n as u32)
        };
    }
    if let Some(v) = values.get("surfaceFineRings").and_then(number) {
        production.atlas.surface_fine_rings = v.clamp(1., 8.).round() as u32;
    }
    if let Some(v) = values.get("timeStep") {
        production.time_step = if v.as_str() == Some("scene") {
            ProductionTimeStep::Scene
        } else {
            ProductionTimeStep::Paper
        };
        production.dt_s = None;
    }
    if let Some(v) = values.get("pressureIterations").and_then(number) {
        if v < 1. || v > 4096. || v.fract() != 0. {
            return Err(ValidationError("invalid pressureIterations".into()));
        }
        world.pressure_iterations = v as u32;
    }
    if let Some(v) = values.get("pressureRelativeTolerance").and_then(number) {
        if v < 0. || !(v as f32).is_finite() {
            return Err(ValidationError("invalid pressureRelativeTolerance".into()));
        }
        world.pressure_relative_tolerance = v as f32;
    }
    production.resolution.policy = activity_policy(&production.resolution.policy, values);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn controls_apply_before_generation_zero() {
        let mut p = ProductionSceneOptions::default();
        let mut w = WorldOptions::default();
        apply_initial_values(&mut p,&mut w,&serde_json::json!({"selectorMode":"surface","surfaceFineRings":3,"maximumMacroSpanBricks":"4","timeStep":"scene","pressureRelativeTolerance":0,"pressureIterations":32})).unwrap();
        assert!(!p.resolution.policy.coarse_first);
        assert_eq!(p.atlas.maximum_macro_span_bricks, Some(4));
        assert_eq!(p.atlas.surface_fine_rings, 3);
        assert_eq!(p.time_step, ProductionTimeStep::Scene);
        assert_eq!(w.pressure_relative_tolerance, 0.);
    }
    #[test]
    fn activity_controls_preserve_ordered_thresholds() {
        let p=activity_policy(&ActivityPolicy::default(),serde_json::json!({"finestTravelCells":0.1,"fourTravelCells":7,"twoTravelCells":5,"demoteScore":1,"promoteScore":0.4,"emergencyScore":0.1,"prepareBricksPerFrame":0}).as_object().unwrap());
        assert_eq!(p.two_travel_cells, 0.1);
        assert_eq!(p.demote_score, 0.4);
        assert_eq!(p.emergency_score, 0.4);
        assert_eq!(p.prepare_bricks_per_frame, 1);
    }
}
