use serde::{Deserialize, Serialize};
use std::{thread, time};
use std::fmt;
use std::error::Error;
use std::iter;


pub trait LLMApi: Serialize {
    fn prompt(&self, system_msg: &str, msgs: impl IntoIterator<Item = Message>) -> Result<ApiResponse, LLMApiError>;
    fn max_context_tokens(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct ApiResponse {
    pub resp: String, 
    pub stop_reason: StopReason, 
    pub usage: Usage, 
}

#[derive(Debug, Copy, Clone)]
pub enum StopReason {
    EndTurn, 
    MaxTokens, 
}

#[derive(Debug, Clone)]
pub struct Usage {
    pub n_input_tokens: usize, 
    pub n_output_tokens: usize, 
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
    messages: Vec<MaskableMessage>, 
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LLMResponse {
    Command(String),
    LLMSee(String),
    MaskContent(usize),
    UserControl, 
    AgentControl, 
    Exit,
}

/*impl LLMResponse {
    pub fn from_str(s: &str) -> Self {
        if s.trim() == "exit" {
            Self::Exit
        } else if let Some(path) = s.strip_prefix("llmsee ") {
            Self::LLMSee(path.to_string())
        } else if let Some(num_str) = s.strip_prefix("maskcontent ") {
            if let Ok(num) = num_str.trim().parse::<usize>() {
                Self::MaskContent(num)
            } else {
                Self::Command(s.to_string())
            }
        } else {
            Self::Command(s.to_string())
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            LLMResponse::Command(cmd) => cmd.clone(),
            LLMResponse::LLMSee(path) => format!("llmsee {}", path),
            LLMResponse::MaskContent(num) => format!("maskcontent {}", num),
            LLMResponse::Exit => "exit".to_string(),
        }
    }
}*/

impl<Api: LLMApi> LLM<Api> {
    pub fn new(api: Api, system_msg: String) -> Self {
        Self {
            api: api, 
            system_msg: system_msg, 
            messages: Vec::new(), 
        }
    }

    pub fn max_context_tokens(&self) -> usize {
        self.api.max_context_tokens()
    }

    pub fn add_msg(&mut self, msg: Message) {
        let id = self.next_msg_id();
        self.messages.push(
            MaskableMessage {
                id: id, 
                is_masked: false, 
                msg: msg, 
            }
        )
    }

    pub fn num_msgs(&self) -> usize {
        self.messages.len()
    }

    pub fn last_msg_id(&self) -> Option<usize> {
        if self.messages.is_empty() {
            None
        } else {
            Some(self.messages.len() - 1)
        }
    }

    pub fn get_msg(&self, id: usize) -> Option<&MaskableMessage> {
        self.messages.get(id)
    }

    pub fn next_msg_id(&self) -> usize {
        self.messages.len()
    }

    pub fn mask_message(&mut self, id: usize) {
        self.messages[id].is_masked = true;
    }

    fn prompt_partial_output(&mut self) -> Result<ApiResponse, LLMApiError> {
        match self.api.prompt(&self.system_msg, self.messages.iter().filter_map(|msg| msg.to_message_with_id())) {
            Ok(resp) => {
                Ok(resp)
            }, 
            Err(err) => {
                self.messages.pop();
                Err(err)
            }, 
        }
    }

    pub fn prompt(&mut self, timeout: time::Duration) -> Result<(Result<LLMResponse, serde_json::Error>, Usage), LLMApiError> {
        let mut error_start_time: Option<time::Instant> = None;

        if self.messages.is_empty() {
            self.add_msg(
                Message {
                    role: Role::User,
                    content: "".into(),
                }
            );
        }

        let num_orig_msgs = self.messages.len();
        
        loop {
            match self.prompt_partial_output() {
                Ok(resp) => {
                    // Reset error timer on success
                    error_start_time = None;
                    
                    match resp.stop_reason {
                        StopReason::EndTurn => {
                            if self.messages.len() == num_orig_msgs {
                                self.add_msg(
                                    Message {
                                        role: Role::Assistant,
                                        content: resp.resp.into(),
                                    }
                                );
                            } else {
                                self.messages
                                    .last_mut()
                                    .unwrap()
                                    .extend_with_content(resp.resp.into());
                            }

                            let msg = self.messages.pop().unwrap();
                            let output_string = <String>::try_from(&msg.msg.content).unwrap();
                            let trimmed = trim_id_prefix(&output_string);
                            self.add_msg(
                                Message {
                                    role: Role::Assistant, 
                                    content: trimmed.into(), 
                                }
                            );

                            return Ok( (serde_json::from_str(trimmed), resp.usage) );
                        },
                        StopReason::MaxTokens => {
                            if self.messages.len() == num_orig_msgs {
                                self.add_msg(
                                    Message {
                                        role: Role::Assistant,
                                        content: resp.resp.into(),
                                    }
                                );
                            } else {
                                self.messages
                                    .last_mut()
                                    .unwrap()
                                    .extend_with_content(resp.resp.into());
                            }
                            
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
                                self.messages.truncate(num_orig_msgs);
                                return Err(err);
                            }
                            
                            thread::sleep(time::Duration::from_secs(1));
                            continue;
                        },
                        _ => {
                            self.messages.truncate(num_orig_msgs);
                            return Err(err)
                        },
                    }
                }
            }
        }
    }
}

fn trim_id_prefix(input: &str) -> &str {
    // Find the position after ">>" if it exists
    if let Some(pos) = input.find(">>") {
        let prefix = &input[..pos];
        // Parse prefix to check if it's a valid integer
        if prefix.parse::<i64>().is_ok() {
            return &input[(pos + 2)..];
        }
    }
    input
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaskableMessage {
    id: usize, 
    is_masked: bool,
    msg: Message,
}

impl MaskableMessage {
    pub fn extend_with_content(&mut self, content: Content) {
        self.msg.extend_with_content(content);
    }

    pub fn get_message(&self) -> &Message {
        &self.msg
    }

    pub fn to_message_with_id(&self) -> Option<Message> {
        match self.is_masked {
            true => None, 
            false => Some(self.to_message_with_id_no_mask()), 
        }
    }

    pub fn to_message_with_id_no_mask(&self) -> Message {
        let id = self.id;
        let id_msg = format!("{id}>>");
        let content_with_id = match &self.msg.content {
            Content::Single(c) => {
                match c {
                    ContentItem::Text(txt) => Content::Single( (id_msg + txt).into() ), 
                    ContentItem::Image(img) => Content::Multiple(vec![id_msg.into(), img.clone().into()])
                }
            }, 
            Content::Multiple(cs) => {
                let cs_with_id = iter::once(id_msg.into())
                    .chain(cs.iter().cloned())
                    .collect();
                Content::Multiple(cs_with_id)
            }, 
        };

        Message {
            role: self.msg.role, 
            content: content_with_id, 
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Content,
}

impl Message {
    pub fn extend_with_content(&mut self, content: Content) {
        self.content.extend(content);
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Role {
    Assistant, 
    User, 
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Content {
    Single(ContentItem), 
    Multiple(Vec<ContentItem>), 
}

impl Content {
    pub fn to_string(&self) -> String {
        match self {
            Content::Single(c) => c.to_string(), 
            Content::Multiple(cs) => cs
                                    .iter()
                                    .map(|c| c.to_string())
                                    .collect(), 
        }
    }

    pub fn extend(&mut self, other: Content) {
        match (&*self, other) {
            (Content::Single(c), Content::Single(other_c)) => {
                *self = Content::Multiple(vec![c.clone(), other_c]);
            }
            (Content::Single(c), Content::Multiple(mut other_cs)) => {
                other_cs.insert(0, c.clone());
                *self = Content::Multiple(other_cs);
            }
            (Content::Multiple(_), Content::Single(other_c)) => {
                if let Content::Multiple(cs) = self {
                    cs.push(other_c);
                }
            }
            (Content::Multiple(_), Content::Multiple(other_cs)) => {
                if let Content::Multiple(cs) = self {
                    cs.extend(other_cs);
                }
            }
        }
    }
}

impl From<String> for Content {
    fn from(s: String) -> Self {
        Content::Single(s.into())
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Content::Single(s.into())
    }
}

impl From<Image> for Content {
    fn from(img: Image) -> Self {
        Content::Single(img.into())
    }
}

impl From<Vec<ContentItem>> for Content {
    fn from(cts: Vec<ContentItem>) -> Self {
        Content::Multiple(cts)
    }
}

impl TryFrom<&Content> for String {
    type Error = &'static str;
    
    fn try_from(content: &Content) -> Result<Self, Self::Error> {
        match content {
            Content::Single(c) => match c {
                ContentItem::Text(s) => Ok(s.clone()),
                ContentItem::Image(_) => Err("Cannot convert Image content to String"),
            },
            Content::Multiple(items) => {
                items.iter()
                    .map(|item| match item {
                        ContentItem::Text(s) => Ok(s.as_str()),
                        ContentItem::Image(_) => Err("Cannot convert Image content to String"),
                    })
                    .collect::<Result<Vec<&str>, _>>()
                    .map(|strs| strs.into_iter().collect())
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentItem {
    Text(String), 
    Image(Image), 
}

impl ContentItem {
    pub fn to_string(&self) -> String {
        match self {
            ContentItem::Text(txt) => txt.to_string(), 
            ContentItem::Image(img) => img.to_string(), 
        }
    }
}

impl From<String> for ContentItem {
    fn from(text: String) -> Self {
        ContentItem::Text(text)
    }
}

impl From<&str> for ContentItem {
    fn from(text: &str) -> Self {
        ContentItem::Text(text.to_string())
    }
}

impl From<Image> for ContentItem {
    fn from(image: Image) -> Self {
        ContentItem::Image(image)
    }
}

impl<'a> TryFrom<&'a ContentItem> for &'a str {
    type Error = &'static str;

    fn try_from(content: &'a ContentItem) -> Result<Self, Self::Error> {
        match content {
            ContentItem::Text(text) => Ok(text.as_str()),
            ContentItem::Image(_) => Err("Cannot convert Image content to &str"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    pub image_type: ImageType, 
    #[serde(skip)]
    pub data: String, 
}

impl Image {
    pub fn to_string(&self) -> String {
        format!("<{} image>", self.image_type.extension())
    }
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








fn output_examples() -> [LLMResponse; 6] {
    [
        LLMResponse::Command("echo \"hello\"".to_string()),
        LLMResponse::LLMSee("img.png".to_string()),
        LLMResponse::MaskContent(42),
        LLMResponse::UserControl, 
        LLMResponse::AgentControl, 
        LLMResponse::Exit,
    ]
}

pub fn generate_system_prompt(task: &str) -> String {
    let output_exps = serde_json::to_string(&output_examples()).unwrap();

    format!(
        "You are in a bash session and will interact directly with a terminal to complete the task: {task}.
Command limits:
You are not permitted to modify any file or folder that is not clearly necessary for completing the task.
You may use any terminal command as you see fit as long as you do not expect the command will violate user privacy.
But only use terminal programs with arguments that are noninteractive. For example, nano and sudo without -S are forbidden because they require interactive input.
Only use terminal programs that just return an output.
cd command does not work, don't use it, paths must be relative to current directory or absolute.
When the task is completed or if it cannot be completed, exit the terminal.
Format:
Each message will have a prefix, id>>, where id is the integer identifier.
The output format is json. Only one json can be outputted. Output EXACTLY the json format, nothing else. Here is an array of examples:
{output_exps}
The format must be precisely one of these types.
Example session:
Terminal: 0>>
LLM: 1>>{{\"Command\":\"echo -e '1\\\\n1\\\\n2\\\\n3\\\\n5' > fibonacci.txt\"}}
Terminal: 2>>
LLM: 3>>{{\"Command\":\"cat fibonacci.txt\"}}
Terminal: 4>>1
1
2
3
5
\"Exit\"
The output is \"Exit\", not {{\"Exit\"}}, and all else is in the context.
Note due to json quirks you must use two slashes for newlines \\\\n within strings, because json parsing treats \\n as a real newline.
This is neccessary.
Everything you output must be a single line terminal command. If you need to think or just say something, use the colon command, example : \"my thoughts must be in quotes\".
Special Commands:
llmsee img_path, lets you see an image, no other command works for viewing images.
maskcontent id, masks the content with the specified id which frees space in the context window, use for content that takes up significant space (like documents/codefiles/etc) and is no longer expected to be needed.
Be especially aggressive with this for images as they take up significant context, often only a single image is needed in the entire context at a time.
usercontrol, hands control to the user, use this if you cannot do something yourself, for example don't know passcode.
agentcontrol, hands back control to you, you never call this.
Context:
Due to token output limits, sometimes a partial command is issued. In that case there will need to be multiple assistant messages in sequence to complete the entire command.
When the token context is nearly full, the terminal will give you a warning. At that point it may be wise to masking content.
Sometimes it is more appropriate to compress content than to erase it entirely. You can output summarized content, then mask the original in such cases.
Programming:
Implementations are rarely needed in the context once complete. When appropriate, mask implementations and keep what is really needed to avoid ambiguity such as types and perhaps a concise description.",
        task = task,
        output_exps = output_exps
    )
}


