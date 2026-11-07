#![deny(unsafe_op_in_unsafe_fn)]

mod abi;
mod cache;
mod error;
mod model;
mod ops;
mod sampler;
mod session;
#[cfg(test)]
mod test_utils;

pub use error::Error;
pub use model::{Config, Model, RopeMode};
pub use sampler::SamplingConfig;
pub use session::Session;

pub type Result<T> = std::result::Result<T, Error>;

pub const ABI_VERSION: u32 = 1;
pub const MODEL_FORMAT_VERSION: u32 = 1;
