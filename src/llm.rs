use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::fmt;
use std::error::Error;


pub trait LLMApi: Serialize {
    fn prompt(&self, system_msg: &str, msgs: &[Message]) -> Result<String, LLMApiError>;
}

#[derive(Debug)]
pub enum LLMApiError {
    NetworkError(reqwest::Error),
    ParseError(serde_json::Error),
    InvalidRequestError,
    AuthenticationError,
    PermissionError,
    NotFoundError,
    RequestTooLarge,
    RateLimitExceeded,
    ApiError,
    OverloadedError,
    Other,
}

// Implement From for network errors
impl From<reqwest::Error> for LLMApiError {
    fn from(error: reqwest::Error) -> Self {
        Self::NetworkError(error)
    }
}

// Implement From for JSON parsing errors
impl From<serde_json::Error> for LLMApiError {
    fn from(error: serde_json::Error) -> Self {
        Self::ParseError(error)
    }
}


impl fmt::Display for LLMApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LLMApiError::NetworkError(err) => write!(f, "Network error: {}", err),
            LLMApiError::ParseError(err) => write!(f, "Parse error: {}", err),
            LLMApiError::InvalidRequestError => write!(f, "Invalid request error"),
            LLMApiError::AuthenticationError => write!(f, "Authentication error"),
            LLMApiError::PermissionError => write!(f, "Permission error"),
            LLMApiError::NotFoundError => write!(f, "Resource not found"),
            LLMApiError::RequestTooLarge => write!(f, "Request too large"),
            LLMApiError::RateLimitExceeded => write!(f, "Rate limit exceeded"),
            LLMApiError::ApiError => write!(f, "API error"),
            LLMApiError::OverloadedError => write!(f, "Service overloaded"),
            LLMApiError::Other => write!(f, "Unknown error"),
        }
    }
}

impl Error for LLMApiError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            LLMApiError::NetworkError(err) => Some(err),
            LLMApiError::ParseError(err) => Some(err),
            _ => None,
        }
    }
}



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLM<Api: LLMApi> {
    api: Api, 
    system_msg: String, 
    messages: Vec<Message>, 
}

impl<Api: LLMApi> LLM<Api> {
    pub fn new(api: Api, system_msg: String) -> Self {
        Self {
            api: api, 
            system_msg: system_msg, 
            messages: Vec::new(), 
        }
    }

    pub fn prompt(&mut self, user_msg: String) -> Result<String, LLMApiError> {
        self.messages.push(
            Message {
                role: Role::User, 
                content: user_msg, 
            }
        );

        match self.api.prompt(&self.system_msg, &self.messages) {
            Ok(resp) => {
                self.messages.push(
                    Message {
                        role: Role::Assistant, 
                        content: resp.clone(), 
                    }
                );
                Ok(resp)
            }, 
            Err(err) => {
                self.messages.pop();
                Err(err)
            }, 
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Role {
    Assistant, 
    User, 
}


pub fn generate_system_prompt(task: &str) -> String {
    format!(
        "You will interact directly with a terminal to complete the task: {task}
You are not permitted to modify any files or folders you did not create, but you may read any file \
that you do not expect to contain sensitive user information if it helps complete the task.
You may use any terminal command as you see fit as long as you do not expect the command will violate user privacy.
But do not use terminal programs that require interactive input such as nano.
Only use terminal programs that just return an output.
Also, cd command does not work, don't use it, paths must be relative to current directory or absolute.
This is due to limitations of the terminal you will be interfacing with.
When the task is completed or if you do not expect to be able to complete it, exit the terminal.

Respond ONLY with the exact command to run. Do not include any explanation or commentary.
When you want to exit, respond with exactly 'exit'."
    )
}
