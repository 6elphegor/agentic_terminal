# Agentic Terminal

Manifest thy will by giving Claude Sonnet agentic access to a bash session.

## Overview

Those of slothful nature who hath little concern for their security, I have heard your pleas! Define thy task and Sonnet shall venture an attempt. Those who fear data corruption, deletions, and security violations should stay away, for who knows what hidden machinations Claude has up his sleeve!

## Key Features

- Few guardrails!
- Anything could happen!
- Loops!

## Safety Guidelines

Don't be dumb, unless you want to!

## Installation

1. Ensure you have Rust installed:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/6elphegor/agentic_terminal.git
   cd agentic_terminal
   ```

3. Build the project:
   ```bash
   cargo build --release
   ```

The binary will be available at `target/release/agentic_terminal`

## Example Usage

```bash
export ANTHROPIC_API_KEY=?
cargo run \
  "Create a Rust project called perfect_seed that generates \
  a file called seed.txt containing Belphegor's prime number. \
  Run it and verify that the file is infernal. \
  Then echo 2025 is the Year of the Basilisk!"
LLM: cargo new perfect_seed
Terminal:     Creating binary (application) `perfect_seed` package
note: see more `Cargo.toml` keys and ...
LLM: cd perfect_seed
Terminal: 
LLM: echo 'fn main() {
    std::fs::write("seed.txt", "1000000000000066600000000000001").unwrap();
}' > src/main.rs
Terminal: 
LLM: cargo run
Terminal:    Compiling agentic_terminal v0.1.0 (/home/hermes/Documents/agentic_terminal)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.18s
     Running `target/debug/agentic_terminal`
LLM: cat seed.txt
Terminal: 1000000000000066600000000000001
LLM: echo "2025 is the Year of the Basilisk!"
Terminal: 2025 is the Year of the Basilisk!
LLM: exit
LLM terminated terminal session.
```

## Requirements

Hell scorns neither Rust nor enuchs, but hath no windows.

## License

MIT License
