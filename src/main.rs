use std::env;
use std::{thread, time};
use std::mem;

mod llm;
mod terminal;

use crate::llm::*;
use crate::terminal::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.contains(&"--help".to_string()) {
        eprintln!("Usage: {} <TASK>", args[0]);
        eprintln!("Accomplishes the given TASK using Claude to issue shell commands.");
        return Ok(());
    }

    // The user’s task is in the first argument
    let task = &args[1];

    // Retrieve your Anthropic API key from the environment
    let anthropic_api_key = match env::var("ANTHROPIC_API_KEY") {
        Ok(k) => k,
        Err(_) => {
            eprintln!("Error: Please set the environment variable ANTHROPIC_API_KEY.");
            return Ok(());
        }
    };

    let system_prompt = generate_system_prompt(task);

    let secret_info = SecretInfo::new(anthropic_api_key);
    let llm = LLM::new(secret_info, system_prompt);

    let terminal = Terminal::new();

    if let Err(e) = run_session_loop(llm, terminal) {
        eprintln!("Session loop terminated with error: {}", e);
        return Err(Box::new(e));
    }

    Ok(())
}

fn run_session_loop(mut llm: LLM, mut terminal: Terminal) -> Result<(), LLMError> {
    let ps1 = ">>";
    let mut last_terminal_output = String::new();
    loop {
        // Send the last terminal output to the LLM and handle potential errors
        let llm_resp = match prompt_llm(&mut llm, format!("{ps1}{last_terminal_output}")) {
            Ok(resp) => resp,
            Err(e) => {
                eprintln!("Error communicating with LLM: {}", e);
                // Depending on the error type, we might want to retry or exit
                match e {
                    LLMError::NetworkError(_) => {
                        // For network errors, we might want to retry after a delay
                        thread::sleep(time::Duration::from_secs(1));
                        continue;
                    }
                    _ => return Err(e), // For other errors, propagate up
                }
            }
        };

        // Show the LLM's command
        println!("LLM: {}", llm_resp.trim());

        // Execute in the hidden terminal
        match prompt_terminal(terminal, &llm_resp) {
            Some((new_terminal, output)) => {
                terminal = new_terminal;
                last_terminal_output = output;
                // Show the terminal output
                println!("Terminal: {}", last_terminal_output.trim_end());
            }
            None => {
                // If None, LLM typed "exit"
                println!("LLM terminated terminal session.");
                return Ok(());
            }
        }

        // Add a small delay between iterations
        thread::sleep(time::Duration::from_millis(200));
    }
}
