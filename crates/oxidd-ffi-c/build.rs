fn main() {
    #[cfg(feature = "cpp")]
    cc::Build::new()
        .cpp(true)
        .file("interop.cpp")
        .compile("cpp_interop");
}
