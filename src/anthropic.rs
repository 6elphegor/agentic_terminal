use serde::{Serialize, Serializer, ser::SerializeMap, Deserialize};
use crate::llm::{self, LLMApi, ApiResponse, LLMApiError, Message, Role};

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

impl Model {
    pub fn max_context_tokens(self) -> usize {
        match self {
            Model::Haiku3_5 => 200_000, 
            Model::Sonnet3_5 => 200_000, 
            Model::Opus3 => 200_000, 
        }
    }

    pub fn max_output_tokens(self) -> usize {
        match self {
            Model::Haiku3_5 => 8192, 
            Model::Sonnet3_5 => 8192, 
            Model::Opus3 => 4096, 
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct AnthropicRequest {
    model: Model,
    system: String,
    max_tokens: usize,
    messages: Vec<AnthropicMessage>,
}

#[derive(Debug, Clone, Serialize)]
struct AnthropicMessage {
    role: AnthropicRole,
    content: Content,
}

impl From<llm::Message> for AnthropicMessage {
    fn from(msg: llm::Message) -> Self {
        Self {
            role: msg.role.into(),
            content: msg.content.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Content {
    PureText(String), 
    Mixed(Vec<ContentElem>), 
}

impl From<llm::Content> for Content {
    fn from(content: llm::Content) -> Self {
        match content {
            llm::Content::Single(c) => {
                match c {
                    llm::ContentItem::Text(txt) => Content::PureText(txt), 
                    llm::ContentItem::Image(img) => Content::Mixed(vec![ContentElem::Image(img.into())]), 
                }
            }, 
            llm::Content::Multiple(cs) => Content::Mixed(
                cs.into_iter()
                    .map(|c| c.into())
                    .collect()
            )
        }
    }
}

#[derive(Debug, Clone)]
pub enum ContentElem {
    Text(String), 
    Image(Image), 
}

impl Serialize for ContentElem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        match self {
            ContentElem::Text(txt) => {
                map.serialize_entry("type", "text")?;
                map.serialize_entry("text", txt)?;
            }
            ContentElem::Image(img) => {
                map.serialize_entry("type", "image")?;
                map.serialize_entry("source", img)?;
            }
        }
        map.end()
    }
}

impl From<llm::ContentItem> for ContentElem {
    fn from(content: llm::ContentItem) -> Self {
        match content {
            llm::ContentItem::Text(text) => ContentElem::Text(text),
            llm::ContentItem::Image(image) => ContentElem::Image(image.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Image {
    media_type: MediaType, 
    data: String, 
}

impl From<llm::Image> for Image {
    fn from(image: llm::Image) -> Self {
        Self {
            media_type: image.image_type.into(),
            data: image.data,
        }
    }
}

impl Serialize for Image {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(3))?;
        map.serialize_entry("type", "base64")?;
        map.serialize_entry("media_type", &self.media_type)?;
        map.serialize_entry("data", &self.data)?;
        map.end()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum MediaType {
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[serde(rename = "image/png")] 
    Png,
    #[serde(rename = "image/gif")]
    Gif,
    #[serde(rename = "image/webp")]
    Webp,
}

impl From<llm::ImageType> for MediaType {
    fn from(image_type: llm::ImageType) -> Self {
        match image_type {
            llm::ImageType::Jpeg => MediaType::Jpeg,
            llm::ImageType::Png => MediaType::Png,
            llm::ImageType::Gif => MediaType::Gif,
            llm::ImageType::Webp => MediaType::Webp,
        }
    }
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
    pub stop_reason: StopReason,
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

#[derive(Debug, Copy, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
}

impl TryInto<llm::StopReason> for StopReason {
    type Error = &'static str;

    fn try_into(self) -> Result<llm::StopReason, Self::Error> {
        match self {
            StopReason::EndTurn => Ok(llm::StopReason::EndTurn),
            StopReason::MaxTokens => Ok(llm::StopReason::MaxTokens),
            StopReason::StopSequence => Err("StopSequence has no equivalent in StopReason"),
            StopReason::ToolUse => Err("ToolUse has no equivalent in StopReason"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct UsageInfo {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl Into<llm::Usage> for UsageInfo {
    fn into(self) -> llm::Usage {
        llm::Usage {
            n_input_tokens: self.input_tokens as usize, 
            n_output_tokens: self.output_tokens as usize, 
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorResponse {
    #[serde(rename = "type")]
    pub response_type: String,
    pub error: AnthropicError,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub error_type: ErrorType,
    pub message: String,
}

#[derive(Debug, Clone, Copy, Deserialize)]
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
    fn max_context_tokens(&self) -> usize {
        self.model.max_context_tokens()
    }
    
    fn prompt(&self, system_msg: &str, msgs: impl IntoIterator<Item = Message>) -> Result<ApiResponse, LLMApiError> {
        let secret_key = self.secret_key.as_ref().ok_or(LLMApiError::AuthenticationError)?;

        let msgs: Vec<AnthropicMessage> = msgs.into_iter().map(|msg| msg.into()).collect();

        let request_body = AnthropicRequest {
            model: self.model,
            system: system_msg.to_string(),
            max_tokens: self.model.max_output_tokens(), 
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
                let resp = response.content
                    .first()
                    .map(|item| item.text.clone())
                    .unwrap_or(String::new());

                let stop_reason = response.stop_reason
                    .try_into()
                    .map_err(|_| LLMApiError::Other)?;

                let usage = response.usage.into();

                Ok(
                    ApiResponse {
                        resp, 
                        stop_reason, 
                        usage, 
                    }
                )
            },
            AnthropicResult::Error(err) => Err(err.error.error_type.into())
        }
    }
}