/* automatically generated by rust-bindgen 0.69.1 */

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ihipStream_t {
    _unused: [u8; 0],
}
pub type hipStream_t = *mut ihipStream_t;
#[repr(C)]
#[derive(Debug, Default, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct __half {
    pub __x: ::core::ffi::c_ushort,
}
#[test]
fn bindgen_test_layout___half() {
    const UNINIT: ::core::mem::MaybeUninit<__half> = ::core::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::core::mem::size_of::<__half>(),
        2usize,
        concat!("Size of: ", stringify!(__half))
    );
    assert_eq!(
        ::core::mem::align_of::<__half>(),
        2usize,
        concat!("Alignment of ", stringify!(__half))
    );
    assert_eq!(
        unsafe { ::core::ptr::addr_of!((*ptr).__x) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(__half),
            "::",
            stringify!(__x)
        )
    );
}
pub type half = __half;
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct rocrand_discrete_distribution_st {
    pub size: ::core::ffi::c_uint,
    pub offset: ::core::ffi::c_uint,
    pub alias: *mut ::core::ffi::c_uint,
    pub probability: *mut f64,
    pub cdf: *mut f64,
}
#[test]
fn bindgen_test_layout_rocrand_discrete_distribution_st() {
    const UNINIT: ::core::mem::MaybeUninit<rocrand_discrete_distribution_st> =
        ::core::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::core::mem::size_of::<rocrand_discrete_distribution_st>(),
        32usize,
        concat!("Size of: ", stringify!(rocrand_discrete_distribution_st))
    );
    assert_eq!(
        ::core::mem::align_of::<rocrand_discrete_distribution_st>(),
        8usize,
        concat!(
            "Alignment of ",
            stringify!(rocrand_discrete_distribution_st)
        )
    );
    assert_eq!(
        unsafe { ::core::ptr::addr_of!((*ptr).size) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(rocrand_discrete_distribution_st),
            "::",
            stringify!(size)
        )
    );
    assert_eq!(
        unsafe { ::core::ptr::addr_of!((*ptr).offset) as usize - ptr as usize },
        4usize,
        concat!(
            "Offset of field: ",
            stringify!(rocrand_discrete_distribution_st),
            "::",
            stringify!(offset)
        )
    );
    assert_eq!(
        unsafe { ::core::ptr::addr_of!((*ptr).alias) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(rocrand_discrete_distribution_st),
            "::",
            stringify!(alias)
        )
    );
    assert_eq!(
        unsafe { ::core::ptr::addr_of!((*ptr).probability) as usize - ptr as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(rocrand_discrete_distribution_st),
            "::",
            stringify!(probability)
        )
    );
    assert_eq!(
        unsafe { ::core::ptr::addr_of!((*ptr).cdf) as usize - ptr as usize },
        24usize,
        concat!(
            "Offset of field: ",
            stringify!(rocrand_discrete_distribution_st),
            "::",
            stringify!(cdf)
        )
    );
}
impl Default for rocrand_discrete_distribution_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct rocrand_generator_base_type {
    _unused: [u8; 0],
}
pub type hiprandGenerator_st = rocrand_generator_base_type;
pub type hiprandDiscreteDistribution_st = rocrand_discrete_distribution_st;
pub type hiprandDirectionVectors32_t = [::core::ffi::c_uint; 32usize];
pub type hiprandDirectionVectors64_t = [::core::ffi::c_ulonglong; 64usize];
pub type hiprandGenerator_t = *mut hiprandGenerator_st;
pub type hiprandDiscreteDistribution_t = *mut hiprandDiscreteDistribution_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum hiprandStatus {
    HIPRAND_STATUS_SUCCESS = 0,
    HIPRAND_STATUS_VERSION_MISMATCH = 100,
    HIPRAND_STATUS_NOT_INITIALIZED = 101,
    HIPRAND_STATUS_ALLOCATION_FAILED = 102,
    HIPRAND_STATUS_TYPE_ERROR = 103,
    HIPRAND_STATUS_OUT_OF_RANGE = 104,
    HIPRAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
    HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
    HIPRAND_STATUS_LAUNCH_FAILURE = 201,
    HIPRAND_STATUS_PREEXISTING_FAILURE = 202,
    HIPRAND_STATUS_INITIALIZATION_FAILED = 203,
    HIPRAND_STATUS_ARCH_MISMATCH = 204,
    HIPRAND_STATUS_INTERNAL_ERROR = 999,
    HIPRAND_STATUS_NOT_IMPLEMENTED = 1000,
}
pub use self::hiprandStatus as hiprandStatus_t;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum hiprandRngType {
    HIPRAND_RNG_TEST = 0,
    HIPRAND_RNG_PSEUDO_DEFAULT = 400,
    HIPRAND_RNG_PSEUDO_XORWOW = 401,
    HIPRAND_RNG_PSEUDO_MRG32K3A = 402,
    HIPRAND_RNG_PSEUDO_MTGP32 = 403,
    HIPRAND_RNG_PSEUDO_MT19937 = 404,
    HIPRAND_RNG_PSEUDO_PHILOX4_32_10 = 405,
    HIPRAND_RNG_QUASI_DEFAULT = 500,
    HIPRAND_RNG_QUASI_SOBOL32 = 501,
    HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 502,
    HIPRAND_RNG_QUASI_SOBOL64 = 503,
    HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 504,
}
pub use self::hiprandRngType as hiprandRngType_t;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum hiprandDirectionVectorSet {
    HIPRAND_DIRECTION_VECTORS_32_JOEKUO6 = 101,
    HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102,
    HIPRAND_DIRECTION_VECTORS_64_JOEKUO6 = 103,
    HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104,
}
pub use self::hiprandDirectionVectorSet as hiprandDirectionVectorSet_t;
extern "C" {
    pub fn hiprandCreateGenerator(
        generator: *mut hiprandGenerator_t,
        rng_type: hiprandRngType_t,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandCreateGeneratorHost(
        generator: *mut hiprandGenerator_t,
        rng_type: hiprandRngType_t,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandDestroyGenerator(generator: hiprandGenerator_t) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerate(
        generator: hiprandGenerator_t,
        output_data: *mut ::core::ffi::c_uint,
        n: usize,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerateChar(
        generator: hiprandGenerator_t,
        output_data: *mut ::core::ffi::c_uchar,
        n: usize,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerateShort(
        generator: hiprandGenerator_t,
        output_data: *mut ::core::ffi::c_ushort,
        n: usize,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerateLongLong(
        generator: hiprandGenerator_t,
        output_data: *mut ::core::ffi::c_ulonglong,
        n: usize,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerateUniform(
        generator: hiprandGenerator_t,
        output_data: *mut f32,
        n: usize,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerateUniformDouble(
        generator: hiprandGenerator_t,
        output_data: *mut f64,
        n: usize,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerateUniformHalf(
        generator: hiprandGenerator_t,
        output_data: *mut half,
        n: usize,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerateNormal(
        generator: hiprandGenerator_t,
        output_data: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerateNormalDouble(
        generator: hiprandGenerator_t,
        output_data: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerateNormalHalf(
        generator: hiprandGenerator_t,
        output_data: *mut half,
        n: usize,
        mean: half,
        stddev: half,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerateLogNormal(
        generator: hiprandGenerator_t,
        output_data: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerateLogNormalDouble(
        generator: hiprandGenerator_t,
        output_data: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerateLogNormalHalf(
        generator: hiprandGenerator_t,
        output_data: *mut half,
        n: usize,
        mean: half,
        stddev: half,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGeneratePoisson(
        generator: hiprandGenerator_t,
        output_data: *mut ::core::ffi::c_uint,
        n: usize,
        lambda: f64,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGenerateSeeds(generator: hiprandGenerator_t) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandSetStream(generator: hiprandGenerator_t, stream: hipStream_t) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandSetPseudoRandomGeneratorSeed(
        generator: hiprandGenerator_t,
        seed: ::core::ffi::c_ulonglong,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandSetGeneratorOffset(
        generator: hiprandGenerator_t,
        offset: ::core::ffi::c_ulonglong,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandSetQuasiRandomGeneratorDimensions(
        generator: hiprandGenerator_t,
        dimensions: ::core::ffi::c_uint,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGetVersion(version: *mut ::core::ffi::c_int) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandCreatePoissonDistribution(
        lambda: f64,
        discrete_distribution: *mut hiprandDiscreteDistribution_t,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandDestroyDistribution(
        discrete_distribution: hiprandDiscreteDistribution_t,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGetDirectionVectors32(
        vectors: *mut *mut hiprandDirectionVectors32_t,
        set: hiprandDirectionVectorSet_t,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGetDirectionVectors64(
        vectors: *mut *mut hiprandDirectionVectors64_t,
        set: hiprandDirectionVectorSet_t,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGetScrambleConstants32(
        constants: *mut *const ::core::ffi::c_uint,
    ) -> hiprandStatus_t;
}
extern "C" {
    pub fn hiprandGetScrambleConstants64(
        constants: *mut *const ::core::ffi::c_ulonglong,
    ) -> hiprandStatus_t;
}
