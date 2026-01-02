#[cfg(feature = "silero-onnx")]
#[test]
fn silero_backend_errors_on_missing_model() {
    use scribble::backends::silero::SileroBackend;

    let msg = match SileroBackend::new("tests/fixtures/does-not-exist.onnx") {
        Ok(_) => panic!("expected error for missing model"),
        Err(err) => format!("{err:#}"),
    };
    assert!(
        msg.contains("failed to load Silero ONNX model"),
        "unexpected error message:\n{msg}"
    );
}
