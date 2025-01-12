use rexpect::{spawn, session::PtySession, error::Error};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum CommandOutput {
    Complete(String), 
    Partial(String), 
}

pub struct Terminal {
    session: PtySession,
}

impl Terminal {
    pub fn new() -> Result<Self, Error> {
        let mut session = spawn("/bin/bash", None/*Some(30000)*/)?; // 30 second timeout?
        
        // Wait for initial prompt and clear it
        session.exp_regex(r"[\$\#] $")?;
        
        // Set up clean environment
        session.send_line("export PS1='CMD_END> '")?;
        session.exp_string("CMD_END> ")?;
        
        Ok(Terminal { session })
    }

    fn clean_output(&self, raw_output: &str, suffix: &str) -> String {
        let trimmed = raw_output.trim();
        trimmed
            .strip_suffix(suffix)
            .unwrap_or(trimmed)
            .to_string()
    }

    pub fn run_line(&mut self, line: &str, timeout: Duration) -> Result<CommandOutput, Error> {
        let line = line.trim();
        if line.trim() == "exit" {
            //return Err("Exit requested".into());
            panic!();
        }

        self.session.send_line(line)?;

        let mut last_char_time = Instant::now();
        let mut output = String::new();
        let mut prompt_buffer = String::new();
        const PROMPT: &str = "CMD_END>";
        
        loop {
            if last_char_time.elapsed() >= timeout {
                // Timeout occurred
                self.session.send_control('c')?;
                // Wait for prompt to return after Ctrl+C
                self.session.exp_string(PROMPT)?;

                return Ok(
                    CommandOutput::Partial(
                        self.clean_output(&output, PROMPT)
                    )
                );
            }

            if let Some(c) = self.session.try_read() {
                last_char_time = Instant::now(); // Reset timer on character receipt
                output.push(c);
                prompt_buffer.push(c);
                
                // Keep prompt buffer at most as long as our prompt
                if prompt_buffer.len() > PROMPT.len() {
                    prompt_buffer.remove(0);
                }
                
                // Check if we've reached the prompt
                if prompt_buffer == PROMPT {
                    return Ok(
                        CommandOutput::Complete(
                            self.clean_output(&output, PROMPT)
                        )
                    );
                }
            }
            
            // Small sleep to prevent busy waiting
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn run_command(&mut self, command: &str, timeout: Duration) -> Result<CommandOutput, Box<dyn std::error::Error>> {
        let mut output = String::new();

        let command = command.trim();

        // Replace newlines with actual newlines and send command
        for line in command.lines() {
            match self.run_line(line, timeout)? {
                CommandOutput::Complete(out) => {
                    output += &out;
                }, 
                CommandOutput::Partial(pout) => {
                    return Ok(
                        CommandOutput::Partial(output + &pout)
                    );
                }, 
            }
        }

        Ok(
            CommandOutput::Complete(output)
        )
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        let _ = self.session.send_line("exit"); 
        // Give it a moment to clean up
        let _ = std::thread::sleep(Duration::from_millis(100));
    }
}