//! What a saved web page reads as: `cargo run -p tts-import --example article -- page.html`.
fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("a saved page");
    let raw = std::env::args().any(|a| a == "--as-is");
    let (page, title) = if raw {
        (std::fs::read_to_string(&path)?, String::new())
    } else {
        tts_import::web::page(&path, &std::fs::read(&path)?)?
    };
    let out = std::env::temp_dir().join("article.html");
    std::fs::write(&out, page)?;
    let document = tts_import::import(&out)?;
    println!("TITLE {title}");
    for chapter in document.chapters {
        println!("\n## {}\n{}", chapter.title.unwrap_or_default(), chapter.body);
    }
    Ok(())
}
