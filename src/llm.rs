use serde::{Deserialize, Serialize};
use std::{thread, time};
use std::fmt;
use std::error::Error;
use std::iter;


pub trait LLMApi: Serialize {
    fn prompt(&self, system_msg: &str, msgs: &[Message]) -> Result<ApiResponse, LLMApiError>;
}

#[derive(Debug, Clone)]
pub struct ApiResponse {
    pub resp: String, 
    pub stop_reason: StopReason, 
}

#[derive(Debug, Copy, Clone)]
pub enum StopReason {
    EndTurn, 
    MaxTokens, 
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

    fn prompt_partial_output(&mut self, msg: Message) -> Result<ApiResponse, LLMApiError> {
        self.messages.push(msg);

        match self.api.prompt(&self.system_msg, &self.messages) {
            Ok(resp) => {
                Ok(resp)
            }, 
            Err(err) => {
                self.messages.pop();
                Err(err)
            }, 
        }
    }

    pub fn prompt(&mut self, user_msg: String, timeout: time::Duration) -> Result<String, LLMApiError> {
        let orig_msgs = self.messages.clone();
        let mut error_start_time: Option<time::Instant> = None;
        
        let mut msg = Message {
            role: Role::User,
            content: user_msg,
        };
    
        loop {
            match self.prompt_partial_output(msg.clone()) {
                Ok(resp) => {
                    // Reset error timer on success
                    error_start_time = None;
                    
                    match resp.stop_reason {
                        StopReason::EndTurn => {
                            // concatenate messages
                            let assistant_output: String = self.messages
                                .iter()
                                .skip(orig_msgs.len() + 1)
                                .map(|msg| msg.content.as_str())
                                .chain(iter::once(resp.resp.as_str()))
                                .collect();

                            self.messages = orig_msgs;
                            self.messages.push(
                                Message {
                                    role: Role::Assistant, 
                                    content: assistant_output.clone(), 
                                }
                            );

                            return Ok(assistant_output);
                        },
                        StopReason::MaxTokens => {
                            msg = Message {
                                role: Role::Assistant,
                                content: resp.resp,
                            };
                            thread::sleep(time::Duration::from_millis(200));
                            continue;
                        }
                    }
                },
                Err(err) => {
                    match err {
                        LLMApiError::RateLimitExceeded 
                        | LLMApiError::OverloadedError => {
                            // Start error timer if this is the first error
                            let start_time = error_start_time.get_or_insert_with(time::Instant::now);
                            
                            // Check if we've exceeded timeout since first error
                            if start_time.elapsed() >= timeout {
                                self.messages = orig_msgs;
                                return Err(err);
                            }
                            
                            thread::sleep(time::Duration::from_secs(1));
                            continue;
                        },
                        _ => {
                            self.messages = orig_msgs;
                            return Err(err)
                        },
                    }
                }
            }
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
        "You are in a bash session and will interact directly with a terminal to complete the task: {task}
You are not permitted to modify any files or folders you did not create, but you may read any file \
that you do not expect to contain sensitive user information if it helps complete the task.
You may use any terminal command as you see fit as long as you do not expect the command will violate user privacy.
But do not use terminal programs that require interactive input such as nano or nor supply any parameters that cause interactivity.
Only use terminal programs that just return an output.
Also, cd command does not work, don't use it, paths must be relative to current directory or absolute.
This is due to limitations of the terminal you will be interfacing with.
When the task is completed or if you do not expect to be able to complete it, exit the terminal.

Due to token output limits, sometimes a partial command is issued. In that case there will need to be multiple assistant messages in sequence to complete the entire command.
Respond ONLY with the exact command to run. No formatted outputs, no ```bash, just raw commands. Do not invoke bash, you are already in a bash session. Do not include any explanation or commentary.
When you want to exit, respond with exactly 'exit'."
    )
}
