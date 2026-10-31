fn main() {
    println!("cargo:rerun-if-env-changed=LITELLM_PYTHON_BRIDGE_SOURCE_HASH");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}
