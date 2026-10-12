//! Uniform Geometric 2D backend for differential scene probes. Its grid,
//! pressure, extension, inflow and rigid stages also serve narrow_band_flip.
//! Configuration is generated from the same method schema as the GPU backend.
pub mod diagnostics;
pub mod extension;
pub mod grid;
pub mod options;
pub mod pages;
pub mod pressure;
pub mod scene_runner;
pub mod surface;
pub mod surface_volume;
pub mod transport;
pub mod velocity;
pub mod world;
pub use options::UniformGeometricOptions;

pub mod physical;

pub(crate) mod inflow;
