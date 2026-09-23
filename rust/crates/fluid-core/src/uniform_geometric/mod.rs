//! Uniform Geometric 2D backend for advance-lab and differential scene probes.
//! Configuration is generated from the same method schema as the GPU backend.
pub mod diagnostics;
pub mod extension;
pub mod grid;
pub mod options;
pub mod pages;
pub mod pressure;
pub mod scene_runner;
pub mod session;
pub mod surface;
pub mod surface_volume;
pub mod transport;
pub mod velocity;
pub mod world;
pub use options::UniformGeometricOptions;

/// The generated contract admits every 3D choice; 2D implements the 3D
/// default algorithm only, so any other choice is refused before a step runs.
pub fn validate_supported(
    options: &UniformGeometricOptions,
) -> Result<(), crate::types::ValidationError> {
    let unsupported = |key: &str| {
        Err(crate::types::ValidationError(format!(
            "Uniform 2D implements only the default {key}"
        )))
    };
    if options.velocity_transport != "semi-lagrangian" {
        return unsupported("velocityTransport");
    }
    if options.liquid_only_velocity_advection != "off" {
        return unsupported("liquidOnlyVelocityAdvection");
    }
    // Rust-only phi/V coupling experiments with no 3D counterpart; retired.
    if options.volume_compaction != "off" {
        return unsupported("volumeCompaction");
    }
    if options.phi_seed_from_volume != "off" {
        return unsupported("phiSeedFromVolume");
    }
    if options.phi_agreement != "off" {
        return unsupported("phiAgreement");
    }
    Ok(())
}

pub mod physical;

mod inflow;
