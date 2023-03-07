extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    let bindings = bindgen::Builder::default()
        .clang_arg("-Iinclude")
        .header("wrapper.h")
        .whitelist_type("ncclNet_v6_t")
        .whitelist_type("ncclDebugLogSubSys")
        .bitfield_enum("ncclDebugLogSubSys")
        .whitelist_var("NCCL_PTR_.*")
        .whitelist_var("NCCL_NET_HANDLE_MAXSIZE")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("unable to write bindings!");
}
