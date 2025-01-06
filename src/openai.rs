use serde::{Serialize, Deserialize};
use crate::llm::{LLMApi, LLMApiError, Message, Role};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAIApi {
    #[serde(skip_serializing)]
    secret_key: Option<String>,
    model: Model,
}

impl OAIApi {
    pub fn new(key: String, model: Model) -> Self {
        Self {
            secret_key: Some(key),
            model,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Model {
    #[serde(rename = "gpt-4o")]
    GPT4O,
    #[serde(rename = "gpt-4o-mini")]
    GPT4OMini,
    #[serde(rename = "o1")]
    O1,
    #[serde(rename = "o1-mini")]
    O1Mini,
    #[serde(rename = "o1-preview")]
    O1Preview,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Sampling {
    Temperature { temperature: f32 },
    TopP { top_p: f32 },
}

#[derive(Debug, Clone, Serialize)]
struct OAIRequest {
    model: Model,
    messages: Vec<OAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    sampling: Option<Sampling>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<ReasoningEffort>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OAIMessage {
    role: OAIRole,
    content: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OAIRole {
    Assistant,
    User,
    Developer,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl From<Role> for OAIRole {
    fn from(role: Role) -> Self {
        match role {
            Role::Assistant => OAIRole::Assistant,
            Role::User => OAIRole::User,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct OAIResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<Choice>,
    pub usage: UsageInfo,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: OAIMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub completion_tokens_details: TokenDetails,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenDetails {
    pub reasoning_tokens: u32,
    pub accepted_prediction_tokens: u32,
    pub rejected_prediction_tokens: u32,
}

impl From<reqwest::StatusCode> for LLMApiError {
    fn from(status: reqwest::StatusCode) -> Self {
        match status.as_u16() {
            401 => LLMApiError::AuthenticationError,
            403 => LLMApiError::PermissionError,
            429 => LLMApiError::RateLimitExceeded,
            500 => LLMApiError::ApiError,
            503 => LLMApiError::OverloadedError,
            _ => LLMApiError::Other,
        }
    }
}



impl LLMApi for OAIApi {
    fn prompt(&self, system_msg: &str, msgs: &[Message]) -> Result<String, LLMApiError> {
        let secret_key = self.secret_key.as_ref().ok_or(LLMApiError::AuthenticationError)?;

        // As of January 2025 
        // O1 models do not support developer messages
        // change when support is added
        let system_msg = match self.model {
            Model::O1 | Model::O1Mini | Model::O1Preview => OAIMessage {
                role: OAIRole::User,
                content: system_msg.to_string(),
            }, 
            _ => OAIMessage {
                role: OAIRole::Developer,
                content: system_msg.to_string(),
            }, 
        };

        let mut messages = vec![system_msg];

        messages.extend(msgs.iter().map(|msg| OAIMessage {
            role: msg.role.into(),
            content: msg.content.clone(),
        }));

        let request_body = OAIRequest {
            model: self.model,
            messages,
            max_completion_tokens: None,
            sampling: None,
            reasoning_effort: None,
        };

        let client = reqwest::blocking::Client::new();
        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", secret_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()?;

        if !response.status().is_success() {
            return Err(response.status().into());
        }

        let body = response.text()?;
        let result: OAIResponse = serde_json::from_str(&body)?;
        
        result.choices
            .first()
            .map(|choice| choice.message.content.clone())
            .ok_or(LLMApiError::Other)
    }
}