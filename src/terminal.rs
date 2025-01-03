use std::process::{Command, Stdio, Child};
use std::io::{Write, BufReader, BufRead};
use std::thread;
use std::sync::mpsc;

pub struct Terminal {
    process: Child,
    stdout_thread: thread::JoinHandle<()>, 
    stderr_thread: thread::JoinHandle<()>, 
    rx: mpsc::Receiver<TerminalOutput>, 
}

impl Terminal {
    pub fn new() -> Self {
        let mut process = Command::new("bash")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start shell process");

        let (tx, rx) = mpsc::channel();

        let stdout = process.stdout.take().unwrap();
        let tx_stdout = tx.clone();
        let stdout_thread = thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            while let Ok(n) = reader.read_line(&mut line) {
                let tout = decode_terminal_output(&line);
                tx_stdout.send(tout).ok();
                line.clear();
            }
        });

        let stderr = process.stderr.take().unwrap();
        let tx_stderr = tx;
        let stderr_thread = thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut line = String::new();
            while let Ok(n) = reader.read_line(&mut line) {
                let tout = decode_terminal_output(&line);
                tx_stderr.send(tout).ok();
                line.clear();
            }
        });

        Terminal { 
            process, 
            stdout_thread, 
            stderr_thread, 
            rx, 
        }
    }
}

pub fn prompt_terminal(mut terminal: Terminal, command: &str) -> Option<(Terminal, String)> {
    let command = command.trim();
    if command == "exit" {
        return None;
    }

    // maps terminal output in an invertible way to ensure that the end is unambiguous
    let command = embellish_command(command);
    
    // Write command
    if let Some(mut stdin) = terminal.process.stdin.take() {
        if stdin.write_all(format!("{}\n", command).as_bytes()).is_err() {
            return None;
        }
        if stdin.flush().is_err() {
            return None;
        }
        terminal.process.stdin = Some(stdin);
    } else {
        return None;
    }

    
    let mut output = String::new();
    while let Ok(data) = terminal.rx.recv() {
        match data {
            TerminalOutput::Line(line) => {
                output.push_str(&line);
            }, 
            TerminalOutput::End => {
                break;
            }
        }
    }

    Some((terminal, output))
}

enum TerminalOutput {
    Line(String), 
    End, 
}

fn embellish_command(cmd: &str) -> String {
    format!("{cmd} 2>&1 | awk '{{print \"1\" $0}} END {{print \"0\"}}'")
}

fn decode_terminal_output(output: &str) -> TerminalOutput {
    if let Some(ch) = output.chars().nth(0) {
        match ch {
            '1' => TerminalOutput::Line(output[1..].into()), 
            _ => TerminalOutput::End, 
        }
    } else {
        TerminalOutput::End
    }
}