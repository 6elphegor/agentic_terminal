use serde::{Serialize, Serializer, ser::SerializeMap, Deserialize};
use crate::llm::{self, LLMApi, ApiResponse, StopReason, LLMApiError, Message, Role};

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

#[derive(Debug, Clone, Serialize)]
struct OAIMessage {
    role: OAIRole,
    content: Content,
}

impl From<llm::Message> for OAIMessage {
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
            llm::Content::Text(text) => Content::PureText(text),
            llm::Content::Image(_) => Content::Mixed(vec![content.into()]),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ContentElem {
    Text(String), 
    Image(Image), 
}

impl From<llm::Content> for ContentElem {
    fn from(content: llm::Content) -> Self {
        match content {
            llm::Content::Text(text) => ContentElem::Text(text),
            llm::Content::Image(image) => ContentElem::Image(image.into()),
        }
    }
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
                map.serialize_entry("type", "image_url")?;
                map.serialize_entry("image_url", img)?;
            }
        }
        map.end()
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
        let data_url = format!("data:{};base64,{}", self.media_type, self.data);
        let mut map = serializer.serialize_map(Some(1))?;
        map.serialize_entry("url", &data_url)?;
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

impl std::fmt::Display for MediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MediaType::Jpeg => write!(f, "image/jpeg"),
            MediaType::Png => write!(f, "image/png"),
            MediaType::Gif => write!(f, "image/gif"),
            MediaType::Webp => write!(f, "image/webp"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OAIRole {
    Assistant,
    User,
    Developer,
}

impl From<llm::Role> for OAIRole {
    fn from(role: llm::Role) -> Self {
        match role {
            llm::Role::Assistant => OAIRole::Assistant,
            llm::Role::User => OAIRole::User,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
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
    pub message: OAIMessageResp,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Deserialize)]
struct OAIMessageResp {
    role: OAIRole,
    content: String,
}

#[derive(Debug, Copy, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
}

impl TryInto<StopReason> for FinishReason {
    type Error = &'static str;

    fn try_into(self) -> Result<StopReason, Self::Error> {
        match self {
            FinishReason::Stop => Ok(StopReason::EndTurn),
            FinishReason::Length => Ok(StopReason::MaxTokens),
            FinishReason::ContentFilter => Err("ContentFilter has no equivalent in StopReason"),
            FinishReason::ToolCalls => Err("ToolCalls has no equivalent in StopReason"),
        }
    }
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
    fn prompt(&self, system_msg: &str, msgs: &[Message]) -> Result<ApiResponse, LLMApiError> {
        let secret_key = self.secret_key.as_ref().ok_or(LLMApiError::AuthenticationError)?;

        // As of January 2025 
        // O1 models do not support developer messages
        // change when support is added
        let system_msg = match self.model {
            Model::O1 | Model::O1Mini | Model::O1Preview => OAIMessage {
                role: OAIRole::User,
                content: Content::PureText(system_msg.to_string()),
            }, 
            _ => OAIMessage {
                role: OAIRole::Developer,
                content: Content::PureText(system_msg.to_string()),
            }, 
        };

        let mut messages = vec![system_msg];

        messages.extend(msgs.iter().map(|msg| msg.clone().into()));

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

        let choice = result.choices
            .first()
            .ok_or(LLMApiError::Other)?;

        let resp = choice.message.content.clone();
        let stop_reason = choice.finish_reason
            .try_into()
            .map_err(|_| LLMApiError::Other)?;

        Ok(
            ApiResponse {
                resp, 
                stop_reason, 
            }
        )
    }
}