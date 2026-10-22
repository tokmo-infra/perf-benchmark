pub fn initialize_python() {
    static INITIALIZE: std::sync::Once = std::sync::Once::new();
    INITIALIZE.call_once(|| {
        // python-build-standalone uses a relocatable stdlib. Embedded test binaries
        // must locate that stdlib from the interpreter used to link PyO3.
        let executable = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_owned());
        let output = std::process::Command::new(executable)
            .args(["-c", "import sys; print(sys.base_prefix)"])
            .output()
            .expect("test Python interpreter");
        assert!(output.status.success());
        if std::env::var_os("PYTHONHOME").is_none() {
            std::env::set_var(
                "PYTHONHOME",
                String::from_utf8(output.stdout).unwrap().trim(),
            );
        }
        pyo3::Python::initialize();
    });
}
