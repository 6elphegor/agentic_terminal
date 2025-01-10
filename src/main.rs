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
    /*let api_key = env::var("API_KEY")
        .map_err(|_| "Please set the environment variable API_KEY")?;
    let anthropic_api = OAIApi::new(api_key, openai::Model::GPT4O);
    let mut llm = LLM::new(anthropic_api, "describe the image using echo".to_string());

    let resp = llm.prompt(Content::Image(Image::from_file("img.png").unwrap()), time::Duration::from_secs(10)).unwrap();
    println!("Response: {:?}", resp);
    return Ok(());*/











    let cli = Cli::parse();

    // Get the appropriate model based on API choice
    let model_choice = cli.model.unwrap_or(match cli.api {
        ApiChoice::Anthropic => ModelChoice::Sonnet3_5,
        ApiChoice::OpenAI => ModelChoice::GPT4O,
    });

    // Prepare the system prompt
    let system_prompt = generate_system_prompt(&cli.task);

    let api_key = env::var("API_KEY")
        .map_err(|_| "Please set the environment variable API_KEY")?;

    // Build the appropriate LLMKind variant
    let mut llm_kind = match cli.api {
        ApiChoice::Anthropic => {
            // Convert model choice to Anthropic model
            let chosen_model = model_choice.to_anthropic_model()
                .ok_or_else(|| format!("Invalid Anthropic model: {:?}", model_choice))?;

            let anthropic_api = AnthropicApi::new(api_key, chosen_model);
            LLMKind::AnthropicLLM(LLM::new(anthropic_api, system_prompt))
        }
        ApiChoice::OpenAI => {
            // Convert model choice to OpenAI model
            let chosen_model = model_choice.to_openai_model()
                .ok_or_else(|| format!("Invalid OpenAI model: {:?}", model_choice))?;

            let oai_api = OAIApi::new(api_key, chosen_model);
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
fn run_session_loop(llm_kind: &mut LLMKind, terminal: Terminal) -> Result<(), LLMApiError> {
    match llm_kind {
        LLMKind::AnthropicLLM(llm) => run_session_loop_generic(llm, terminal), 
        LLMKind::OpenAILLM(llm) => run_session_loop_generic(llm, terminal), 
    }
}

fn run_session_loop_generic<Api: LLMApi>(
    llm: &mut LLM<Api>,
    mut terminal: Terminal
) -> Result<(), LLMApiError> {
    let mut last_content_output: Content = "".into();

    let timeout = time::Duration::from_secs(10);

    let mut n_msgs_printed = 0;

    loop {
        // Send the last content output to the LLM
        let (llm_resp, usage) = match llm.prompt(last_content_output, timeout) {
            Ok(resp) => resp,
            Err(e) => {
                eprintln!("Error communicating with LLM: {}", e);
                return Err(e);
            }
        };

        let n_msgs = llm.num_msgs();
        for id in n_msgs_printed..n_msgs {
            let msg = llm
                        .get_msg(id)
                        .unwrap()
                        .to_message_with_id_no_mask();

            match msg.role {
                Role::Assistant => {
                    println!("LLM: {}", msg.content.to_string());
                }, 
                Role::User => {
                    println!("Terminal: {}", msg.content.to_string());
                }, 
            }
        }
        n_msgs_printed = n_msgs;

        /*let response_id = llm.last_msg_id().unwrap();
        let prompt_id = response_id - 1;
        let last_response_msg = llm.get_msg(response_id).unwrap();
        let last_prompt_msg = llm.get_msg(prompt_id).unwrap();

        println!("Terminal: {}", last_prompt_msg.to_message_with_id_no_mask().content.to_string());
        println!("LLM: {}", last_response_msg.to_message_with_id_no_mask().content.to_string());*/

        match llm_resp {
            LLMResponse::Command(command) => {
                // Execute in the hidden terminal
                match prompt_terminal(terminal, &command) {
                    Some((new_terminal, output)) => {
                        terminal = new_terminal;
                        last_content_output = output.into();
                    }
                    None => {
                        // If None, LLM typed "exit"
                        println!("LLM terminated terminal session.");
                        return Ok(());
                    }
                }
            }, 
            LLMResponse::LLMSee(img_path) => {
                let content = match Image::from_file(&img_path) {
                    Ok(img) => img.into(), 
                    Err(e) => e.to_string().into(), 
                };

                last_content_output = content;
            }, 
            LLMResponse::MaskContent(id) => {
                llm.mask_message(id);
                last_content_output = format!("message {id} is masked").into();
            }, 
            LLMResponse::Exit => {
                println!("Terminal session terminated.");
                return Ok(());
            }, 
        }

        if usage.n_input_tokens + usage.n_output_tokens > llm.max_context_tokens() * 9 / 10 {
            llm.add_msg(
                Message {
                    role: Role::User, 
                    content: "Warning: over 90% of token context is used.".into(), 
                }
            );
        }

        // Add a small delay between iterations
        thread::sleep(time::Duration::from_millis(200));
    }
}


