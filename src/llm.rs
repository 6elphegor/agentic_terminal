use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct SecretInfo {
    anthropic_api_key: String,
}

impl SecretInfo {
    pub fn new(key: String) -> Self {
        Self {
            anthropic_api_key: key, 
        }
    }
}

/// A single message in the conversation. Valid roles are typically "user", "assistant", etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

/// The top-level request to Anthropic’s /v1/messages endpoint.
#[derive(Debug, Clone, Serialize)]
struct AnthropicRequest {
    model: String,
    system: String,
    max_tokens: usize,
    messages: Vec<AnthropicMessage>,
}

/// The response shape returned by Anthropic’s /v1/messages,
#[derive(Debug, Clone, Deserialize)]
struct AnthropicResponse {
    content: Option<Vec<ContentItem>>, 
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
}

/// Each item in the `content` array typically has { "type": "text", "text": "some text" }.
#[derive(Debug, Clone, Deserialize)]
struct ContentItem {
    #[serde(rename = "type")]
    item_type: String,
    text: String,
}

#[derive(Debug)]
pub struct LLM {
    secret_info: SecretInfo,
    system_prompt: String,
    messages: Vec<AnthropicMessage>,
    model: String,
}

impl LLM {
    pub fn new(secret_info: SecretInfo, system_prompt: String) -> Self {
        LLM {
            secret_info,
            system_prompt,
            messages: vec![],
            model: "claude-3-5-sonnet-latest".to_string(),
        }
    }
}

pub fn generate_system_prompt(task: &str) -> String {
    format!(
        "You will interact directly with a terminal to complete the task: {task}
You are not permitted to modify any files or folders you did not create, but you may read any file \
that you do not expect to contain sensitive user information if it helps complete the task.
You may use any terminal command as you see fit as long as you do not expect the command will violate user privacy.
But do not use terminal programs that require interactive input such as nano.
Only use terminal programs that just return an output.
Also, cd command does not work, don't use it, paths must be relative to current directory.
This is due to limitations of the terminal you will be interfacing with.
When the task is completed or if you do not expect to be able to complete it, exit the terminal.

Respond ONLY with the exact command to run. Do not include any explanation or commentary.
When you want to exit, respond with exactly 'exit'."
    )
}





/// Call Anthropic's /v1/messages endpoint.
/// Returns the updated LLM state and the response content on success,
/// or an error describing what went wrong.
pub fn prompt_llm(llm: &mut LLM, user_message: String) -> Result<String, LLMError> {
    let mut updated_messages = llm.messages.clone();
    updated_messages.push(AnthropicMessage {
        role: "user".to_string(),
        content: user_message,
    });

    // Build the request body
    let request_body = AnthropicRequest {
        model: llm.model.clone(),
        system: llm.system_prompt.clone(),
        max_tokens: 1024,
        messages: updated_messages.clone(),
    };

    // Make the API request
    let client = reqwest::blocking::Client::new();
    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &llm.secret_info.anthropic_api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&request_body)
        .send()?;

    // Check if the status code indicates success
    if !response.status().is_success() {
        return Err(LLMError::InvalidResponse(format!(
            "HTTP error {}: {}", 
            response.status(),
            response.text().unwrap_or_else(|_| "Could not read error response".to_string())
        )));
    }

    // Parse the response body
    let body = response.text()?;
    let anthropic_resp: AnthropicResponse = serde_json::from_str(&body)?;

    // Extract the response content
    let llm_resp = anthropic_resp.content
        .ok_or(LLMError::MissingContent)?
        .first()
        .ok_or(LLMError::EmptyContent)?
        .text
        .clone();

    updated_messages.push(AnthropicMessage {
        role: "assistant".to_string(),
        content: llm_resp.clone(),
    });

    llm.messages = updated_messages;

    Ok(llm_resp)
}

use std::fmt;

#[derive(Debug)]
pub enum LLMError {
    NetworkError(reqwest::Error),
    ParseError(serde_json::Error),
    EmptyContent,
    MissingContent,
    InvalidResponse(String),
}

// Implement std::error::Error for our custom error type
impl std::error::Error for LLMError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LLMError::NetworkError(e) => Some(e),
            LLMError::ParseError(e) => Some(e),
            _ => None,
        }
    }
}

// Implement Display for pretty printing
impl fmt::Display for LLMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LLMError::NetworkError(e) => write!(f, "Network error: {}", e),
            LLMError::ParseError(e) => write!(f, "Failed to parse LLM response: {}", e),
            LLMError::EmptyContent => write!(f, "Empty content array in LLM response"),
            LLMError::MissingContent => write!(f, "No content field in LLM response"),
            LLMError::InvalidResponse(msg) => write!(f, "Invalid API response: {}", msg),
        }
    }
}

// Implement From traits for automatic conversion
impl From<reqwest::Error> for LLMError {
    fn from(err: reqwest::Error) -> LLMError {
        LLMError::NetworkError(err)
    }
}

impl From<serde_json::Error> for LLMError {
    fn from(err: serde_json::Error) -> LLMError {
        LLMError::ParseError(err)
    }
}
