fn main() {
    println!("cargo:rerun-if-env-changed=FA4_NATIVE_LIB_DIR");
    let dir = std::env::var_os("FA4_NATIVE_LIB_DIR")
        .expect("build the AOT bundle with scripts/build_aot.py and set FA4_NATIVE_LIB_DIR");
    let dir = std::path::PathBuf::from(dir)
        .canonicalize()
        .expect("invalid FA4_NATIVE_LIB_DIR");
    for name in ["libfa4bridge.so", "libcute_dsl_runtime.so", "libtvm_ffi.so"] {
        assert!(dir.join(name).is_file(), "missing {name} in native bundle");
        println!("cargo:rerun-if-changed={}", dir.join(name).display());
    }
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=dylib=fa4bridge");
}
