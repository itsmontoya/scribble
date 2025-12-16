use clap::ValueEnum;

#[derive(Debug, Clone, ValueEnum)]
pub enum OutputType {
    Json,
    Vtt,
}
