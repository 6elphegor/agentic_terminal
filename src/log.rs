use std::path::PathBuf;
use std::io::{BufWriter, Write};
use std::fs::File;
use dirs::cache_dir;
use uuid::Uuid;
use super::llm::{LLM, LLMApi};

pub fn save_session_log(llm: &LLM<impl LLMApi>) -> std::io::Result<()> {
    let log_dir = get_log_dir();
    let uuid = Uuid::now_v7();
    let log_path = log_dir.join(format!("{}.json", uuid));
    
    let file = File::create(log_path)?;
    let mut writer = BufWriter::new(file);
    let serialized = serde_json::to_string_pretty(llm)
        .expect("Failed to serialize LLM state");
    
    writer.write_all(serialized.as_bytes())?;
    writer.flush()?;
    Ok(())
}

fn get_log_dir() -> PathBuf {
    let mut path = cache_dir().unwrap_or_else(|| PathBuf::from("~/.cache"));
    path.push("agentic_terminal");
    path.push("logs");
    std::fs::create_dir_all(&path).expect("Failed to create log directory");
    path
}