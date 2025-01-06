use std::env;
use std::{thread, time};
use std::mem;

mod llm;
mod anthropic;
mod terminal;
mod log;

use crate::llm::*;
use crate::anthropic::{AnthropicApi, Model};

use crate::terminal::*;

use crate::log::save_session_log;

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

    let anthropic_api = AnthropicApi::new(anthropic_api_key, Model::Sonnet3_5);
    let system_prompt = generate_system_prompt(task);
    let mut llm = LLM::new(anthropic_api, system_prompt);

    let terminal = Terminal::new();

    if let Err(e) = run_session_loop(&mut llm, terminal) {
        eprintln!("Session loop terminated with error: {}", e);
        save_session_log(&llm)?;
        return Err(Box::new(e));
    } else {
        save_session_log(&llm)?;
    }

    Ok(())
}

fn run_session_loop<Api: LLMApi>(llm: &mut LLM<Api>, mut terminal: Terminal) -> Result<(), LLMApiError> {
    let ps1 = ">>";
    let mut last_terminal_output = String::new();
    loop {
        // Send the last terminal output to the LLM and handle potential errors
        let llm_resp = match llm.prompt(format!("{ps1}{last_terminal_output}")) {
            Ok(resp) => resp,
            Err(e) => {
                eprintln!("Error communicating with LLM: {}", e);
                // Depending on the error type, we might want to retry or exit
                match e {
                    LLMApiError::NetworkError(_) => {
                        // For network errors, we might want to retry after a delay
                        thread::sleep(time::Duration::from_secs(1));
                        continue;
                    }, 
                    LLMApiError::RateLimitExceeded => {
                        thread::sleep(time::Duration::from_secs(1));
                        continue;
                    }, 
                    LLMApiError::OverloadedError => {
                        thread::sleep(time::Duration::from_secs(1));
                        continue;
                    }, 
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
