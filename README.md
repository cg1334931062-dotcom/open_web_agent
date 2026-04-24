# Web Agent

A self-hosted AI coding agent that runs entirely in the browser. No CLI installation needed — users only need a browser and an API key.

## Overview

Web Agent provides a Claude Code-like experience through a web interface. Deploy it on a server, and users access it from any browser.

**Use case**: Teams or individuals who cannot install CLI tools (Claude Code, Codex, OpenCode) on their local machines, but still want an AI coding agent experience.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Open `http://localhost:8080` in your browser, enter your API key in settings, and start coding.

## Features

- **Chat interface** with real-time streaming of agent thinking and tool calls
- **8 built-in tools**: read/write/edit files, bash, glob, grep, directory listing, ask user
- **Skill system**: extend the agent with custom tools (Python, bash, or HTTP)
- **File tree browser** with syntax-highlighted file viewer
- **Dark/light themes**
- **Anthropic Claude** and **OpenAI** support

## Configuration

Configure via the Settings panel in the UI:

| Setting | Description |
|---|---|
| API Key | Your Anthropic or OpenAI API key (never stored on disk) |
| Provider | `anthropic` or `openai` |
| Model | Model ID (e.g., `claude-sonnet-4-20250514`, `gpt-4o`) |
| Workspace Path | Absolute path to the project directory |

## Skill System

Skills are pluggable tools defined in YAML. Place them in `.skills/` directories:

```
.skills/
├── code-review/
│   └── skill.yaml
└── deploy/
    ├── skill.yaml
    └── script.sh
```

Each skill becomes a tool the agent can call. Skills can be Python scripts, bash commands, or HTTP endpoints.

See `skills/` for examples.

## Security

- API keys are stored in memory only (never written to disk)
- File operations are restricted to the workspace directory (path traversal blocked)
- Bash commands are filtered against a dangerous-command blacklist
- File sizes are limited (read: 1MB, write: 5MB)
- WebSocket Origin validation available

## Tech Stack

- **Backend**: Python, FastAPI, WebSocket
- **Frontend**: Tailwind CSS, highlight.js (single HTML file, no build step)
- **LLM**: Anthropic Claude API (primary), OpenAI API (fallback)
