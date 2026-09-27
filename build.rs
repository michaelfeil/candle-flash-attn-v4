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
    if std::env::var_os("CARGO_FEATURE_DEBERTA").is_some() {
        let manifest = dir.join("manifest.json");
        println!("cargo:rerun-if-changed={}", manifest.display());
        let contents = std::fs::read_to_string(manifest).expect("DeBERTa requires an AOT manifest");
        assert!(
            contents.contains("fa4_deberta_fp16") && contents.contains("fa4_deberta_bf16"),
            "rebuild the FA4 bundle with scripts/build_aot.py --deberta"
        );
    }
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=dylib=fa4bridge");
}
