//! A thin wrapper around [sys] providing [Result]s with [CurandError].
//!
//! Two flavors of generation:
//! 1. Not generic: See [generate] for non-generic generation functions.
//! 2. Generic: See [UniformFill], [NormalFill], and [LogNormalFill] for generic generation functions.

use super::sys;
use std::mem::MaybeUninit;

/// Wrapper around [sys::curandStatus_t].
/// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gb94a31d5c165858c96b6c18b70644437)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CurandError(pub sys::hiprandStatus_t);

impl sys::hiprandStatus_t {
    /// Transforms into a [Result] of [CurandError]
    pub fn result(self) -> Result<(), CurandError> {
        match self {
            sys::hiprandStatus_t::HIPRAND_STATUS_SUCCESS => Ok(()),
            _ => Err(CurandError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CurandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CurandError {}

/// Create new random number generator with the default pseudo rng type.
///
/// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g56ff2b3cf7e28849f73a1e22022bcbfd).
pub fn create_generator() -> Result<sys::hiprandGenerator_t, CurandError> {
    create_generator_kind(sys::hiprandRngType_t::HIPRAND_RNG_PSEUDO_DEFAULT)
}

/// Create new random number generator.
///
/// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g56ff2b3cf7e28849f73a1e22022bcbfd).
pub fn create_generator_kind(
    kind: sys::hiprandRngType_t,
) -> Result<sys::hiprandGenerator_t, CurandError> {
    let mut generator = MaybeUninit::uninit();
    unsafe {
        sys::hiprandCreateGenerator(generator.as_mut_ptr(), kind).result()?;
        Ok(generator.assume_init())
    }
}

/// Set the seed value of the pseudo-random number generator.
///
/// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gbcd2982aa3d53571b8ad12d8188b139b)
///
/// # Safety
/// The generator must be allocated and not already freed.
pub unsafe fn set_seed(generator: sys::hiprandGenerator_t, seed: u64) -> Result<(), CurandError> {
    sys::hiprandSetPseudoRandomGeneratorSeed(generator, seed).result()
}

/// Set the offset value of the pseudo-random number generator.
///
/// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gb21ba987f85486e552797206451b0939)
///
/// # Safety
/// The generator must be allocated and not already freed.
pub unsafe fn set_offset(
    generator: sys::hiprandGenerator_t,
    offset: u64,
) -> Result<(), CurandError> {
    sys::hiprandSetGeneratorOffset(generator, offset).result()
}

/// Set the current stream for CURAND kernel launches.
///
/// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gc78c8d07c7acea4242e2a62bc41ff1f5)
///
/// # Safety
/// 1. The generator must be allocated and not already freed.
/// 2. The stream must be allocated and not already freed.
pub unsafe fn set_stream(
    generator: sys::hiprandGenerator_t,
    stream: sys::hipStream_t,
) -> Result<(), CurandError> {
    sys::hiprandSetStream(generator, stream).result()
}

/// Destroy an existing generator.
///
/// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g8d82c56e2b869fef4f9929a775ee18d0).
///
/// # Safety
/// The generator must not have already been freed.
pub unsafe fn destroy_generator(generator: sys::hiprandGenerator_t) -> Result<(), CurandError> {
    sys::hiprandDestroyGenerator(generator).result()
}

pub mod generate {
    //! Functions to generate different distributions.

    use super::{sys, CurandError};

    /// Fills `out` with `num` f32 values in the range (0.0, 1.0].
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g5df92a7293dc6b2e61ea481a2069ebc2)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn uniform_f32(
        gen: sys::hiprandGenerator_t,
        out: *mut f32,
        num: usize,
    ) -> Result<(), CurandError> {
        sys::hiprandGenerateUniform(gen, out, num).result()
    }

    /// Fills `out` with `num` f64 values in the range (0.0, 1.0].
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gbb08f0268f05c9d87eac2b4a2cf7fc24)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn uniform_f64(
        gen: sys::hiprandGenerator_t,
        out: *mut f64,
        num: usize,
    ) -> Result<(), CurandError> {
        sys::hiprandGenerateUniformDouble(gen, out, num).result()
    }

    /// Fills `out` with `num` u32 values with all bits random.
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gf18b3cbdf0b7d9e2335bada92610adac)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn uniform_u32(
        gen: sys::hiprandGenerator_t,
        out: *mut u32,
        num: usize,
    ) -> Result<(), CurandError> {
        sys::hiprandGenerate(gen, out, num).result()
    }

    /// Fills `out` with `num` f32 values from a normal distribution
    /// parameterized by `mean` and `std`.
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gb9280e447ef04e1dec4611720bd0eb69)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn normal_f32(
        gen: sys::hiprandGenerator_t,
        out: *mut f32,
        num: usize,
        mean: f32,
        std: f32,
    ) -> Result<(), CurandError> {
        sys::hiprandGenerateNormal(gen, out, num, mean, std).result()
    }

    /// Fills `out` with `num` f64 values from a normal distribution
    /// parameterized by `mean` and `std`.
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g046759ff9b6bf8dafc9eaae04917dc8e)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn normal_f64(
        gen: sys::hiprandGenerator_t,
        out: *mut f64,
        num: usize,
        mean: f64,
        std: f64,
    ) -> Result<(), CurandError> {
        sys::hiprandGenerateNormalDouble(gen, out, num, mean, std).result()
    }

    /// Fills `out` with `num` f32 values from a log normal distribution
    /// parameterized by `mean` and `std`.
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g3569cc960eb1a31357752fc813e21f49)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn log_normal_f32(
        gen: sys::hiprandGenerator_t,
        out: *mut f32,
        num: usize,
        mean: f32,
        std: f32,
    ) -> Result<(), CurandError> {
        sys::hiprandGenerateLogNormal(gen, out, num, mean, std).result()
    }

    /// Fills `out` with `num` f64 values from a normal distribution
    /// parameterized by `mean` and `std`.
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g300c31530c8b461ca89f1e0232a6f05f)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn log_normal_f64(
        gen: sys::hiprandGenerator_t,
        out: *mut f64,
        num: usize,
        mean: f64,
        std: f64,
    ) -> Result<(), CurandError> {
        sys::hiprandGenerateLogNormalDouble(gen, out, num, mean, std).result()
    }

    /// Fills `out` with `num` u32 values from a poisson distribution
    /// parameterized by `lambda`.
    ///
    /// See [cuRAND docs](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g425c7c13db4444e6150d159bb1417f05)
    ///
    /// # Safety
    /// 1. generator must have been allocated and not freed.
    /// 2. `out` point to `num` values
    pub unsafe fn poisson_u32(
        gen: sys::hiprandGenerator_t,
        out: *mut u32,
        num: usize,
        lambda: f64,
    ) -> Result<(), CurandError> {
        sys::hiprandGeneratePoisson(gen, out, num, lambda).result()
    }
}

/// Fill with uniform distributed numbers of type `T`.
pub trait UniformFill<T> {
    /// # Safety
    /// This inherits the unsafe from methods in [generate].
    unsafe fn fill(self, out: *mut T, num: usize) -> Result<(), CurandError>;
}

impl UniformFill<f32> for sys::hiprandGenerator_t {
    unsafe fn fill(self, out: *mut f32, num: usize) -> Result<(), CurandError> {
        generate::uniform_f32(self, out, num)
    }
}

impl UniformFill<f64> for sys::hiprandGenerator_t {
    unsafe fn fill(self, out: *mut f64, num: usize) -> Result<(), CurandError> {
        generate::uniform_f64(self, out, num)
    }
}

impl UniformFill<u32> for sys::hiprandGenerator_t {
    unsafe fn fill(self, out: *mut u32, num: usize) -> Result<(), CurandError> {
        generate::uniform_u32(self, out, num)
    }
}

/// Fill with normally distributed numbers of type `T`.
pub trait NormalFill<T> {
    /// # Safety
    /// This inherits the unsafe from methods in [generate].
    unsafe fn fill(self, o: *mut T, n: usize, m: T, s: T) -> Result<(), CurandError>;
}

impl NormalFill<f32> for sys::hiprandGenerator_t {
    unsafe fn fill(self, o: *mut f32, n: usize, m: f32, s: f32) -> Result<(), CurandError> {
        generate::normal_f32(self, o, n, m, s)
    }
}

impl NormalFill<f64> for sys::hiprandGenerator_t {
    unsafe fn fill(self, o: *mut f64, n: usize, m: f64, s: f64) -> Result<(), CurandError> {
        generate::normal_f64(self, o, n, m, s)
    }
}

/// Fill with log normally distributed numbers of type `T`.
pub trait LogNormalFill<T> {
    /// # Safety
    /// This inherits the unsafe from methods in [generate].
    unsafe fn fill(self, o: *mut T, n: usize, m: T, s: T) -> Result<(), CurandError>;
}

impl LogNormalFill<f32> for sys::hiprandGenerator_t {
    unsafe fn fill(self, o: *mut f32, n: usize, m: f32, s: f32) -> Result<(), CurandError> {
        generate::log_normal_f32(self, o, n, m, s)
    }
}

impl LogNormalFill<f64> for sys::hiprandGenerator_t {
    unsafe fn fill(self, o: *mut f64, n: usize, m: f64, s: f64) -> Result<(), CurandError> {
        generate::log_normal_f64(self, o, n, m, s)
    }
}
