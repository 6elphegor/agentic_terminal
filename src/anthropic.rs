use serde::{Serialize, Deserialize};
use crate::llm::{LLMApi, LLMApiError, Message, Role};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicApi {
    #[serde(skip_serializing)]
    secret_key: Option<String>,
    model: Model,
}

impl AnthropicApi {
    pub fn new(key: String, model: Model) -> Self {
        Self {
            secret_key: Some(key), 
            model: model, 
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Model {
    #[serde(rename = "claude-3-5-haiku-latest")]
    Haiku3_5,
    #[serde(rename = "claude-3-5-sonnet-latest")]
    Sonnet3_5,
    #[serde(rename = "claude-3-opus-latest")]
    Opus3,
}

#[derive(Debug, Clone, Serialize)]
struct AnthropicRequest {
    model: Model,
    system: String,
    max_tokens: usize,
    messages: Vec<AnthropicMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicMessage {
    role: AnthropicRole,
    content: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    Assistant, 
    User, 
}

impl From<Role> for AnthropicRole {
    fn from(role: Role) -> Self {
        match role {
            Role::Assistant => AnthropicRole::Assistant,
            Role::User => AnthropicRole::User,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicResult {
    Success(AnthropicResponse),
    Error(ErrorResponse),
}

impl AnthropicResult {
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }

    pub fn into_response(self) -> Result<AnthropicResponse, AnthropicError> {
        match self {
            Self::Success(response) => Ok(response),
            Self::Error(err) => Err(err.error),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicResponse {
    pub content: Vec<ContentItem>,
    pub id: String,
    pub model: String,
    pub role: AnthropicRole,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    #[serde(rename = "type")]
    pub message_type: String,
    pub usage: UsageInfo,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentItem {
    #[serde(rename = "type")]
    pub item_type: String,
    pub text: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UsageInfo {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorResponse {
    #[serde(rename = "type")]
    pub response_type: String,
    pub error: AnthropicError,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub error_type: ErrorType,
    pub message: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorType {
    InvalidRequestError,
    AuthenticationError,
    PermissionError,
    NotFoundError,
    RequestTooLarge,
    RateLimitError,
    ApiError,
    OverloadedError,
}

impl Into<LLMApiError> for ErrorType {
    fn into(self) -> LLMApiError {
        match self {
            Self::InvalidRequestError => LLMApiError::InvalidRequestError,
            Self::AuthenticationError => LLMApiError::AuthenticationError,
            Self::PermissionError => LLMApiError::PermissionError,
            Self::NotFoundError => LLMApiError::NotFoundError,
            Self::RequestTooLarge => LLMApiError::RequestTooLarge,
            Self::RateLimitError => LLMApiError::RateLimitExceeded,
            Self::ApiError => LLMApiError::ApiError,
            Self::OverloadedError => LLMApiError::OverloadedError,
        }
    }
}

impl LLMApi for AnthropicApi {
    fn prompt(&self, system_msg: &str, msgs: &[Message]) -> Result<String, LLMApiError> {
        let secret_key = self.secret_key.as_ref().ok_or(LLMApiError::AuthenticationError)?;

        let msgs: Vec<AnthropicMessage> = msgs.iter().map(|msg| {
            AnthropicMessage {
                role: msg.role.into(),
                content: msg.content.clone(),
            }
        }).collect();

        let request_body = AnthropicRequest {
            model: self.model,
            system: system_msg.to_string(),
            max_tokens: 1024,
            messages: msgs,
        };

        let client = reqwest::blocking::Client::new();
        let response = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", secret_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request_body)
            .send()?;

        let body = response.text()?;
        let result: AnthropicResult = serde_json::from_str(&body)?;

        match result {
            AnthropicResult::Success(response) => {
                response.content
                    .first()
                    .map(|item| item.text.clone())
                    .ok_or(LLMApiError::Other)
            },
            AnthropicResult::Error(err) => Err(err.error.error_type.into())
        }
    }
}