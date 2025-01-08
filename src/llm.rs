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

#[derive(Debug, Clone)]
pub enum LLMResponse {
    Command(String),
    LLMSee(String),
    Exit,
}

impl LLMResponse {
    pub fn from_str(s: &str) -> Self {
        if s.trim() == "exit" {
            Self::Exit
        } else if let Some(path) = s.strip_prefix("llmsee ") {
            Self::LLMSee(path.to_string())
        } else {
            Self::Command(s.to_string())
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            LLMResponse::Command(cmd) => cmd.clone(),
            LLMResponse::LLMSee(path) => format!("llmsee {}", path),
            LLMResponse::Exit => "exit".to_string(),
        }
    }
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

    pub fn prompt(&mut self, user_content: Content, timeout: time::Duration) -> Result<LLMResponse, LLMApiError> {
        let orig_msgs = self.messages.clone();
        let mut error_start_time: Option<time::Instant> = None;
        
        let mut msg = Message {
            role: Role::User,
            content: user_content,
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
                                .map(|msg| <&str>::try_from(&msg.content).unwrap())
                                .chain(iter::once(resp.resp.as_str()))
                                .collect();

                            self.messages = orig_msgs;
                            self.messages.push(
                                Message {
                                    role: Role::Assistant, 
                                    content: assistant_output.clone().into(), 
                                }
                            );

                            return Ok(LLMResponse::from_str(assistant_output.as_str()));
                        },
                        StopReason::MaxTokens => {
                            msg = Message {
                                role: Role::Assistant,
                                content: resp.resp.into(),
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
    pub content: Content,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Role {
    Assistant, 
    User, 
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Content {
    Text(String), 
    Image(Image), 
}

impl From<String> for Content {
    fn from(text: String) -> Self {
        Content::Text(text)
    }
}

impl From<&str> for Content {
    fn from(text: &str) -> Self {
        Content::Text(text.to_string())
    }
}

impl From<Image> for Content {
    fn from(image: Image) -> Self {
        Content::Image(image)
    }
}

impl<'a> TryFrom<&'a Content> for &'a str {
    type Error = &'static str;

    fn try_from(content: &'a Content) -> Result<Self, Self::Error> {
        match content {
            Content::Text(text) => Ok(text.as_str()),
            Content::Image(_) => Err("Cannot convert Image content to &str"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    pub image_type: ImageType, 
    #[serde(skip)]
    pub data: String, 
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum ImageType {
    Jpeg,
    Png,
    Gif,
    Webp,
}












use std::path::Path;
use std::fs::File;
use std::io::{self, Read, BufReader};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

#[derive(Debug)]
pub enum ImageLoadError {
    FileError(std::io::Error),
    UnsupportedExtension,
    NoExtension,
}

impl fmt::Display for ImageLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileError(e) => write!(f, "Failed to read file: {}", e),
            Self::UnsupportedExtension => write!(f, "Unsupported image extension"),
            Self::NoExtension => write!(f, "File has no extension"),
        }
    }
}

impl Error for ImageLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::FileError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ImageLoadError {
    fn from(err: std::io::Error) -> Self {
        Self::FileError(err)
    }
}

impl ImageType {
    pub fn extension(&self) -> &'static str {
        match self {
            ImageType::Jpeg => "jpg",
            ImageType::Png => "png",
            ImageType::Gif => "gif",
            ImageType::Webp => "webp",
        }
    }
    
    fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "png" => Some(Self::Png),
            "gif" => Some(Self::Gif),
            "webp" => Some(Self::Webp),
            _ => None,
        }
    }
}

impl Image {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ImageLoadError> {
        let path = path.as_ref();
        
        // Get image type from extension
        let image_type = path.extension()
            .ok_or(ImageLoadError::NoExtension)?
            .to_str()
            .and_then(ImageType::from_extension)
            .ok_or(ImageLoadError::UnsupportedExtension)?;
        
        // Read file with buffered reader
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;
        
        // Convert to base64
        let data = BASE64.encode(&buffer);
        
        Ok(Image {
            image_type,
            data,
        })
    }
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

There is a special command, llmsee img_path, that lets you see an image, no other command work for viewing images.
Everything you output must be a single line terminal command. If you need to think or just say something, use the colon command, example : \"my thoughts must be in quotes\".

Due to token output limits, sometimes a partial command is issued. In that case there will need to be multiple assistant messages in sequence to complete the entire command.
Respond ONLY with the exact command to run. No formatted outputs, no ```bash, just raw commands. Do not invoke bash, you are already in a bash session. Do not include any explanation or commentary.
When you want to exit, respond with exactly 'exit'."
    )
}
