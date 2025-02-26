mod approx_algo;
#[cfg(feature = "approx_unique")]
mod approx_unique;
mod arg_min_max;
mod clip;
#[cfg(feature = "cutqcut")]
mod cut;
#[cfg(feature = "round_series")]
mod floor_divide;
#[cfg(feature = "fused")]
mod fused;
#[cfg(feature = "convert_index")]
mod index;
#[cfg(feature = "is_first_distinct")]
mod is_first_distinct;
#[cfg(feature = "is_in")]
mod is_in;
#[cfg(feature = "is_last_distinct")]
mod is_last_distinct;
#[cfg(feature = "is_unique")]
mod is_unique;
#[cfg(feature = "log")]
mod log;
#[cfg(feature = "rle")]
mod rle;
#[cfg(feature = "rolling_window")]
mod rolling;
#[cfg(feature = "search_sorted")]
mod search_sorted;
#[cfg(feature = "to_dummies")]
mod to_dummies;
mod various;

pub use approx_algo::*;
#[cfg(feature = "approx_unique")]
pub use approx_unique::*;
pub use arg_min_max::ArgAgg;
pub use clip::*;
#[cfg(feature = "cutqcut")]
pub use cut::*;
#[cfg(feature = "round_series")]
pub use floor_divide::*;
#[cfg(feature = "fused")]
pub use fused::*;
#[cfg(feature = "convert_index")]
pub use index::*;
#[cfg(feature = "is_first_distinct")]
pub use is_first_distinct::*;
#[cfg(feature = "is_in")]
pub use is_in::*;
#[cfg(feature = "is_last_distinct")]
pub use is_last_distinct::*;
#[cfg(feature = "is_unique")]
pub use is_unique::*;
#[cfg(feature = "log")]
pub use log::*;
use polars_core::prelude::*;
#[cfg(feature = "rle")]
pub use rle::*;
#[cfg(feature = "rolling_window")]
pub use rolling::*;
#[cfg(feature = "search_sorted")]
pub use search_sorted::*;
#[cfg(feature = "to_dummies")]
pub use to_dummies::*;
pub use various::*;

pub trait SeriesSealed {
    fn as_series(&self) -> &Series;
}

impl SeriesSealed for Series {
    fn as_series(&self) -> &Series {
        self
    }
}
