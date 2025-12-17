use scribble::{opts::Opts, scribble::Scribble};

#[test]
fn transcribes_wav_to_json() -> anyhow::Result<()> {
    let wav = std::fs::File::open("tests/fixtures/treat_yo_self.wav")?;
    let mut out = Vec::new();

    let scribble = Scribble::new(
        "./models/ggml-large-v3-turbo.bin",
        "./models/ggml-silero-v6.2.0.bin",
    )?;

    let opts = Opts {
        enable_translate_to_english: false,
        enable_voice_activity_detection: false,
        language: None,
        output_type: scribble::output_type::OutputType::Json,
    };

    // Pass &mut out, not out
    scribble.transcribe(wav, &mut out, &opts)?;

    let json = String::from_utf8(out)?;
    assert!(json.contains("Treat. Yo. Self."));

    Ok(())
}

#[test]
fn transcribes_wav_to_json_with_vad() -> anyhow::Result<()> {
    let wav = std::fs::File::open("tests/fixtures/treat_yo_self.wav")?;
    let mut out = Vec::new();

    let scribble = Scribble::new(
        "./models/ggml-large-v3-turbo.bin",
        "./models/ggml-silero-v6.2.0.bin",
    )?;

    let opts = Opts {
        enable_translate_to_english: false,
        enable_voice_activity_detection: true,
        language: None,
        output_type: scribble::output_type::OutputType::Json,
    };

    // Pass &mut out, not out
    scribble.transcribe(wav, &mut out, &opts)?;

    let json = String::from_utf8(out)?;
    assert!(json.contains("Treat. Yo. Self."));

    Ok(())
}

#[test]
fn transcribes_wav_to_json_with_language() -> anyhow::Result<()> {
    let wav = std::fs::File::open("tests/fixtures/treat_yo_self.wav")?;
    let mut out = Vec::new();

    let scribble = Scribble::new(
        "./models/ggml-large-v3-turbo.bin",
        "./models/ggml-silero-v6.2.0.bin",
    )?;

    let opts = Opts {
        enable_translate_to_english: false,
        enable_voice_activity_detection: false,
        language: Some("en".to_string()),
        output_type: scribble::output_type::OutputType::Json,
    };

    // Pass &mut out, not out
    scribble.transcribe(wav, &mut out, &opts)?;

    let json = String::from_utf8(out)?;
    assert!(json.contains("Treat. Yo. Self."));

    Ok(())
}

#[test]
fn transcribes_wav_to_json_with_vad_and_language() -> anyhow::Result<()> {
    let wav = std::fs::File::open("tests/fixtures/treat_yo_self.wav")?;
    let mut out = Vec::new();

    let scribble = Scribble::new(
        "./models/ggml-large-v3-turbo.bin",
        "./models/ggml-silero-v6.2.0.bin",
    )?;

    let opts = Opts {
        enable_translate_to_english: false,
        enable_voice_activity_detection: true,
        language: Some("en".to_string()),
        output_type: scribble::output_type::OutputType::Json,
    };

    // Pass &mut out, not out
    scribble.transcribe(wav, &mut out, &opts)?;

    let json = String::from_utf8(out)?;
    assert!(json.contains("Treat. Yo. Self."));

    Ok(())
}
