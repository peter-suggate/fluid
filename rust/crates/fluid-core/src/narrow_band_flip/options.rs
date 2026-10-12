//! The native subset of narrowBandFlipValues (uniform-narrow-band-method.ts).
use crate::{types::ValidationError, uniform_geometric::UniformGeometricOptions};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NarrowBandFlipOptions {
    pub rigid_coupling: String,
    pub pressure_residual_tolerance: f32,
    pub extension_front_sweeps: f32,
    pub pressure_full_cycles: f32,
    pub pressure_v_cycles: f32,
    pub pressure_sweeps: f32,
    pub phi_fine_advection: String,
    /// h cells a sample's support grows by before it joins a tile to the band.
    pub fine_grid_padding: f32,
    /// Time over which the pressure solve closes the gap to the volume
    /// budget; zero turns volume control off.
    pub volume_control_seconds: f32,
}
impl Default for NarrowBandFlipOptions {
    fn default() -> Self {
        Self {
            rigid_coupling: "on".into(),
            pressure_residual_tolerance: 5.0,
            extension_front_sweeps: 1.0,
            pressure_full_cycles: 3.0,
            pressure_v_cycles: 4.0,
            pressure_sweeps: 6.0,
            phi_fine_advection: "cubic".into(),
            fine_grid_padding: 1.0,
            volume_control_seconds: 0.1,
        }
    }
}
impl NarrowBandFlipOptions {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if !["cubic", "trilinear"].contains(&self.phi_fine_advection.as_str()) {
            return Err(ValidationError(
                "Invalid narrow-band FLIP parameter phiFineAdvection".into(),
            ));
        }
        if !self.fine_grid_padding.is_finite() || !(0.0..=4.0).contains(&self.fine_grid_padding) {
            return Err(ValidationError(
                "Invalid narrow-band FLIP parameter fineGridPadding".into(),
            ));
        }
        if !self.volume_control_seconds.is_finite()
            || !(0.0..=2.0).contains(&self.volume_control_seconds)
        {
            return Err(ValidationError(
                "Invalid narrow-band FLIP parameter volumeControlSeconds".into(),
            ));
        }
        self.solver().validate()
    }
    /// The grid stages' options: narrowBandFlipValues' fixed arm. Nothing
    /// transports, sharpens or corrects a cell volume.
    pub fn solver(&self) -> UniformGeometricOptions {
        UniformGeometricOptions {
            density_sharpening: "off".into(),
            rigid_coupling: self.rigid_coupling.clone(),
            time_step: "sixtieth".into(),
            pressure_residual_tolerance: self.pressure_residual_tolerance,
            extension_front_sweeps: self.extension_front_sweeps,
            pressure_full_cycles: self.pressure_full_cycles,
            pressure_v_cycles: self.pressure_v_cycles,
            pressure_sweeps: self.pressure_sweeps,
            volume_dust_threshold: 0.0,
            phi_cubic_advection: if self.phi_fine_advection == "cubic" {
                "on"
            } else {
                "off"
            }
            .into(),
            phi_drain: "off".into(),
            total_surface_volume: "off".into(),
            surface_deficit_balancing: "off".into(),
        }
    }
}
