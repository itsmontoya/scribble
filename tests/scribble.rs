use std::path::Path;

use scribble::{Opts, OutputType, Scribble, WhisperBackend};

const FIXTURE_WAV: &str = "tests/fixtures/jfk.wav";
const WHISPER_MODEL: &str = "./models/ggml-tiny.bin";
const VAD_MODEL: &str = "./models/ggml-silero-v6.2.0.bin";

fn require_file(path: &str) -> anyhow::Result<()> {
    if !Path::new(path).exists() {
        // We skip (not fail) so contributors can run `cargo test` without large model downloads.
        eprintln!("skipping: missing required test file: {path}");
        return Ok(());
    }
    Ok(())
}

fn new_scribble_or_skip() -> anyhow::Result<Option<Scribble<WhisperBackend>>> {
    require_file(WHISPER_MODEL)?;
    require_file(VAD_MODEL)?;
    if !Path::new(WHISPER_MODEL).exists() || !Path::new(VAD_MODEL).exists() {
        return Ok(None);
    }

    Ok(Some(Scribble::new([WHISPER_MODEL], VAD_MODEL)?))
}

fn run_to_json(scribble: &mut Scribble<WhisperBackend>, opts: &Opts) -> anyhow::Result<String> {
    let wav = std::fs::File::open(FIXTURE_WAV)?;
    let mut out = Vec::new();

    scribble.transcribe(wav, &mut out, opts)?;

    Ok(String::from_utf8(out)?)
}

#[test]
fn transcribes_wav_to_json_variants() -> anyhow::Result<()> {
    // Fixture must exist; if it doesn’t, that’s a real test setup bug.
    assert!(
        Path::new(FIXTURE_WAV).exists(),
        "missing fixture WAV: {FIXTURE_WAV}"
    );

    let Some(mut scribble) = new_scribble_or_skip()? else {
        return Ok(()); // skipped
    };

    let cases = [
        (
            "default",
            Opts {
                model_key: None,
                enable_translate_to_english: false,
                enable_voice_activity_detection: false,
                language: None,
                output_type: OutputType::Json,
                incremental_min_window_seconds: 1,
            },
        ),
        (
            "with_vad",
            Opts {
                model_key: None,
                enable_translate_to_english: false,
                enable_voice_activity_detection: true,
                language: None,
                output_type: OutputType::Json,
                incremental_min_window_seconds: 1,
            },
        ),
        (
            "with_language",
            Opts {
                model_key: None,
                enable_translate_to_english: false,
                enable_voice_activity_detection: false,
                language: Some("en".to_string()),
                output_type: OutputType::Json,
                incremental_min_window_seconds: 1,
            },
        ),
        (
            "with_vad_and_language",
            Opts {
                model_key: None,
                enable_translate_to_english: false,
                enable_voice_activity_detection: true,
                language: Some("en".to_string()),
                output_type: OutputType::Json,
                incremental_min_window_seconds: 1,
            },
        ),
    ];

    for (name, opts) in cases {
        let json = run_to_json(&mut scribble, &opts)?;
        assert!(
            json.contains(" We choose to go to the moon and this decay and do the other things, not because they are easy, but because they are hard."),
            "case `{name}` did not contain expected phrase. Output:\n{json}"
        );
    }

    Ok(())
}
