//! `dream-tts-serve`: the service in `lib.rs`, run from the command line.

fn main() -> anyhow::Result<()> {
    tts_serve::run(std::env::args_os())
}
