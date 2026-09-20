//! Uniform Geometric 2D backend for advance-lab and differential scene probes.
//! Configuration is generated from the same method schema as the GPU backend.
pub mod extension;
pub mod diagnostics;
pub mod grid;
pub mod options;
pub mod pressure;
pub mod scene_runner;
pub mod session;
pub mod surface;
pub mod swept_extension;
pub mod transport;
pub mod velocity;
pub mod world;
pub use options::UniformGeometricOptions;

pub mod physical;
