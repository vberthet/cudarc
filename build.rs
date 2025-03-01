use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(not(feature = "ci-check"))]
    link_hip();
}

#[allow(unused)]
fn link_hip() {
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");

    let candidates: Vec<PathBuf> = root_candidates().collect();

    let toolkit_root = root_candidates()
        .find(|path| path.join("include").join("hip").is_dir())
        .unwrap_or_else(|| {
            panic!(
                "Unable to find `include/hip` under any of: {:?}. Set the `CUDA_ROOT` environment variable to `$CUDA_ROOT/include/cuda.h` to override path.",
                candidates
            )
        });

    for path in lib_candidates(&toolkit_root) {
        println!("cargo:rustc-link-search={}", path.display());
    }


    println!("cargo:rustc-link-lib=dylib=amdhip64");
    #[cfg(feature = "cublas")]
    println!("cargo:rustc-link-lib=dylib=hipblas");

    #[cfg(feature = "cublaslt")]
    println!("cargo:rustc-link-lib=dylib=hipblaslt");

    #[cfg(feature = "curand")]
    println!("cargo:rustc-link-lib=dylib=hiprand");

    #[cfg(feature = "nccl")]
    println!("cargo:rustc-link-lib=dylib=rccl");
}

fn root_candidates() -> impl Iterator<Item = PathBuf> {
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok);

    let roots = [
        "/usr",
        "/usr/local/rocm",
        "/opt/rocm",
        "/usr/lib/rocm",
        // "C:/Program Files/NVIDIA GPU Computing Toolkit", #TODO Fix windows
        // "C:/CUDA", #TODO Fix windows
    ];
    let roots = roots.into_iter().map(Into::into);
    env_vars.chain(roots).map(Into::<PathBuf>::into)
}

fn lib_candidates(root: &Path) -> Vec<PathBuf> {
    [
        "lib",
        "lib/x64",
        "lib/Win32",
        "lib/x86_64",
        "lib/x86_64-linux-gnu",
        "lib64",
        "lib64/stubs",
        "targets/x86_64-linux",
        "targets/x86_64-linux/lib",
        "targets/x86_64-linux/lib/stubs",
    ]
    .iter()
    .map(|&p| root.join(p))
    .filter(|p| p.is_dir())
    .collect()
}
