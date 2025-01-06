use clap::{Parser, ValueEnum};
use std::env;
use std::{thread, time};

mod llm;
mod anthropic;
mod openai;
mod terminal;
mod log;

use crate::llm::*;
use crate::anthropic::AnthropicApi;
use crate::openai::OAIApi;
use crate::terminal::*;
use crate::log::save_session_log;

#[derive(Clone, Debug, ValueEnum)]
enum ApiChoice {
    Anthropic,
    #[value(name = "openai")]
    OpenAI,
}

#[derive(Clone, Debug, ValueEnum)]
enum ModelChoice {
    #[value(name = "claude-3-5-haiku-latest")]
    Haiku3_5,
    #[value(name = "claude-3-5-sonnet-latest")]
    Sonnet3_5,
    #[value(name = "claude-3-opus-latest")]
    Opus3,
    #[value(name = "gpt-4o")]
    GPT4O,
    #[value(name = "gpt-4o-mini")]
    GPT4OMini,
    #[value(name = "o1")]
    O1,
    #[value(name = "o1-mini")]
    O1Mini,
    #[value(name = "o1-preview")]
    O1Preview,
}

/// CLI tool for interacting with LLM APIs
#[derive(Parser, Debug)]
#[command(name = "agentic-terminal")]
#[command(about = "Manifest thy will by granting an LLM agentic access to a bash session.", long_about = None)]
struct Cli {
    /// The task to perform
    task: String,

    /// Which API to use
    #[arg(long, value_enum, default_value_t = ApiChoice::Anthropic)]
    api: ApiChoice,

    /// Which model to use (defaults to claude-3-5-sonnet-latest for Anthropic or gpt-4o for OpenAI)
    #[arg(long, value_enum)]
    model: Option<ModelChoice>,
}

impl ModelChoice {
    fn to_anthropic_model(&self) -> Option<anthropic::Model> {
        match self {
            ModelChoice::Haiku3_5 => Some(anthropic::Model::Haiku3_5),
            ModelChoice::Sonnet3_5 => Some(anthropic::Model::Sonnet3_5),
            ModelChoice::Opus3 => Some(anthropic::Model::Opus3),
            _ => None,
        }
    }

    fn to_openai_model(&self) -> Option<openai::Model> {
        match self {
            ModelChoice::GPT4O => Some(openai::Model::GPT4O),
            ModelChoice::GPT4OMini => Some(openai::Model::GPT4OMini),
            ModelChoice::O1 => Some(openai::Model::O1),
            ModelChoice::O1Mini => Some(openai::Model::O1Mini),
            ModelChoice::O1Preview => Some(openai::Model::O1Preview),
            _ => None,
        }
    }
}

enum LLMKind {
    AnthropicLLM(LLM<AnthropicApi>),
    OpenAILLM(LLM<OAIApi>),
}

impl LLMKind {
    fn apply<FA, FO, R>(
        &mut self,
        f_anthropic: FA,
        f_openai: FO
    ) -> R
    where
        FA: FnOnce(&mut LLM<AnthropicApi>) -> R,
        FO: FnOnce(&mut LLM<OAIApi>) -> R,
    {
        match self {
            LLMKind::AnthropicLLM(llm) => f_anthropic(llm),
            LLMKind::OpenAILLM(llm) => f_openai(llm),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Get the appropriate model based on API choice
    let model_choice = cli.model.unwrap_or(match cli.api {
        ApiChoice::Anthropic => ModelChoice::Sonnet3_5,
        ApiChoice::OpenAI => ModelChoice::GPT4O,
    });

    // Prepare the system prompt
    let system_prompt = generate_system_prompt(&cli.task);

    // Build the appropriate LLMKind variant
    let mut llm_kind = match cli.api {
        ApiChoice::Anthropic => {
            // Retrieve Anthropic API key
            let anthropic_api_key = env::var("API_KEY")
                .map_err(|_| "Please set the environment variable API_KEY")?;

            // Convert model choice to Anthropic model
            let chosen_model = model_choice.to_anthropic_model()
                .ok_or_else(|| format!("Invalid Anthropic model: {:?}", model_choice))?;

            let anthropic_api = AnthropicApi::new(anthropic_api_key, chosen_model);
            LLMKind::AnthropicLLM(LLM::new(anthropic_api, system_prompt))
        }
        ApiChoice::OpenAI => {
            // Retrieve OpenAI API key
            let oai_api_key = env::var("API_KEY")
                .map_err(|_| "Please set the environment variable API_KEY")?;

            // Convert model choice to OpenAI model
            let chosen_model = model_choice.to_openai_model()
                .ok_or_else(|| format!("Invalid OpenAI model: {:?}", model_choice))?;

            let oai_api = OAIApi::new(oai_api_key, chosen_model);
            LLMKind::OpenAILLM(LLM::new(oai_api, system_prompt))
        }
    };

    // Set up the local pseudo-terminal
    let terminal = Terminal::new();

    // Run the conversation loop
    if let Err(e) = run_session_loop(&mut llm_kind, terminal) {
        eprintln!("Session loop terminated with error: {}", e);
        // If an error occurs, still save the session log
        if let Err(e2) = llm_kind.apply(
            |anthropic_llm| save_session_log(anthropic_llm),
            |openai_llm| save_session_log(openai_llm),
        ) {
            eprintln!("Failed to save session log: {}", e2);
        }
        return Err(Box::new(e));
    }

    // On successful exit, also save session log
    if let Err(e2) = llm_kind.apply(
        |anthropic_llm| save_session_log(anthropic_llm),
        |openai_llm| save_session_log(openai_llm),
    ) {
        eprintln!("Failed to save session log: {}", e2);
    }

    Ok(())
}

/// Runs a loop that prompts the LLM and feeds it terminal output
/// We'll just call a single generic function using `apply`, 
/// so we don't need a `match` here.
fn run_session_loop(llm_kind: &mut LLMKind, terminal: Terminal) -> Result<(), LLMApiError> {
    // Pass a closure for each variant:
    match llm_kind {
        LLMKind::AnthropicLLM(llm) => run_session_loop_generic(llm, terminal), 
        LLMKind::OpenAILLM(llm) => run_session_loop_generic(llm, terminal), 
    }
}

/// The “real” session loop, made generic over any `LLMApi`.
fn run_session_loop_generic<Api: LLMApi>(
    llm: &mut LLM<Api>,
    mut terminal: Terminal
) -> Result<(), LLMApiError> {
    let ps1 = ">>";
    let mut last_terminal_output = String::new();

    loop {
        // Send the last terminal output to the LLM
        let llm_resp = match llm.prompt(format!("{ps1}{last_terminal_output}")) {
            Ok(resp) => resp,
            Err(e) => {
                eprintln!("Error communicating with LLM: {}", e);
                match e {
                    LLMApiError::NetworkError(_) 
                    | LLMApiError::RateLimitExceeded
                    | LLMApiError::OverloadedError => {
                        // Sleep briefly and retry certain errors
                        thread::sleep(time::Duration::from_secs(1));
                        continue;
                    }
                    _ => return Err(e), 
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


