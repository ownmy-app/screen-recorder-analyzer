# screen-recorder-analyzer

<p align="center">
  <b>Built by <a href="https://nometria.com">Nometria</a></b> — We take AI-built apps to production.
</p>

> Analyze any screen recording: Whisper audio transcription + Tesseract OCR on keyframes + GPT action extraction. CLI and REST API.

Extracts a chronological list of what the user was doing:

```json
[
  {"id": "1", "tools": ["excel"], "action": ["viewing spreadsheet data"]},
  {"id": "2", "tools": ["hubspot"], "action": ["navigating CRM", "viewing contacts"]},
  {"id": "3", "tools": ["gmail"], "action": ["composing email", "sending email"]}
]
```

---

## Quick start

```bash
# System requirements
brew install ffmpeg tesseract          # macOS
sudo apt install ffmpeg tesseract-ocr  # Ubuntu

# Install (base -- just needs OPENAI_API_KEY)
pip install screen-recorder-analyzer

# Install with all OCR/audio engines
pip install screen-recorder-analyzer[full]

# Set API key
export OPENAI_API_KEY=sk-proj-...

# Analyze a recording (one command does it all: transcribe + OCR + actions)
screen-analyze demo.mp4

# Use OpenAI Whisper API instead of local model (no torch download needed)
screen-analyze demo.mp4 --whisper-backend api

# JSON output (suitable for piping)
screen-analyze demo.mp4 --format json

# Run tests (no GPU/OCR/Whisper required)
git clone https://github.com/nometria/screen-recorder-analyzer
cd screen-recorder-analyzer
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Install

```bash
pip install screen-recorder-analyzer[full]
```

System requirements: `ffmpeg`, `tesseract-ocr` on PATH.

```bash
# macOS
brew install ffmpeg tesseract

# Ubuntu/Debian
sudo apt install ffmpeg tesseract-ocr
```

---

## CLI

```bash
# Analyze a recording (text output)
screen-analyze demo.mp4

# JSON output
screen-analyze demo.mp4 --format json

# Use a larger Whisper model for better accuracy
screen-analyze demo.mp4 --whisper small

# Skip GPT step (transcription + OCR only)
screen-analyze demo.mp4 --no-actions

# Analyze more frames
screen-analyze demo.mp4 --max-frames 200 --frame-skip 14
```

---

## REST API

```bash
# Start server
pip install screen-recorder-analyzer[full,api]
uvicorn screen_recorder_analyzer.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/process-video/ \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/recording.mp4"}'
```

---

## Python library

```python
from screen_recorder_analyzer import VideoProcessor, extract_actions

processor = VideoProcessor(whisper_model_size="small", frame_skip=14)
results = processor.process("demo.mp4")

actions = extract_actions(results)
for action in actions:
    print(f"[{action['id']}] {action['tools']}: {action['action']}")
```

---

## Multi-LLM support

Action extraction supports multiple LLM backends. Set the provider via environment variables:

```bash
# Use Anthropic Claude
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Use any LiteLLM-supported model
export LLM_PROVIDER=litellm
export LLM_MODEL=together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
```

Install the optional backend:
```bash
pip install screen-recorder-analyzer[anthropic]  # Anthropic Claude
pip install screen-recorder-analyzer[litellm]    # LiteLLM (any provider)
```

---

## Whisper backend

Choose between a local Whisper model (default) and the OpenAI Whisper API:

```bash
# Local model (default) -- requires openai-whisper + torch
screen-analyze demo.mp4 --whisper-backend local

# OpenAI API -- no local model download, just needs OPENAI_API_KEY
screen-analyze demo.mp4 --whisper-backend api

# Or set via env var
export WHISPER_BACKEND=api
screen-analyze demo.mp4
```

---

## Configuration (env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required for openai/whisper-api | OpenAI API key |
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, or `litellm` |
| `LLM_MODEL` | per-provider | Model override (e.g. `gpt-4o`, `claude-sonnet-4-20250514`) |
| `ANTHROPIC_API_KEY` | required for anthropic | Anthropic API key |
| `WHISPER_MODEL` | `base` | Whisper model size (local backend only) |
| `WHISPER_BACKEND` | `local` | `local` (openai-whisper) or `api` (OpenAI Whisper API) |
| `FRAME_SKIP` | `29` | Analyze every N+1 frames |
| `MAX_FRAMES` | `100` | Max frames to OCR |
| `OCR_LANG` | `eng` | Tesseract language |

---

## Use cases

- **Productivity analysis** — understand how employees use tools
- **UX research** — extract task flows from usability test recordings
- **Process mining** — map manual workflows before automating them
- **Support** — auto-summarize customer screen shares

---

## Commercial viability

- CLI: open source
- API: self-hostable, or offer as a cloud service (pay per minute of video processed)
- Paid: team dashboards, tool usage analytics, process bottleneck detection

---

---

## Built by Nometria

<a href="https://nometria.com">
  <img src="https://img.shields.io/badge/nometria.com-Take%20AI%20apps%20to%20production-111827?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMTIgMkw0IDdWMTdMMTIgMjJMMjAgMTdWN0wxMiAyWiIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIi8+PC9zdmc+" alt="Nometria" />
</a>

**screen-recorder-analyzer** is open source and free to use. It's one of the developer tools we built while helping teams ship AI-generated apps to production.

Understanding how users interact with deployed apps is critical. We built this to extract actionable insights from screen recordings -- transcription, OCR, and AI-powered action detection.

**What Nometria does:**
- :rocket: **Deploy AI apps to AWS** -- one click, production-ready
- :lock: **Security & compliance** -- SOC 2, HIPAA-ready infrastructure
- :chart_with_upwards_trend: **Scale reliably** -- handles real user traffic from day one
- :wrench: **Full source code ownership** -- you own everything, no lock-in

If you're building with AI tools (Base44, Lovable, Bolt, Replit, Cursor) and need to go to production -- **[nometria.com](https://nometria.com)**

---

## Example output

Running `pytest tests/ -v`:

```
============================= test session starts ==============================
platform darwin -- Python 3.13.9, pytest-9.0.2, pluggy-1.5.0
cachedir: .pytest_cache
rootdir: /tmp/ownmy-releases/screen-recorder-analyzer
configfile: pyproject.toml
plugins: anyio-4.12.1, cov-7.1.0
collecting ... collected 6 items

tests/test_processor.py::test_processor_imports_cleanly PASSED           [ 16%]
tests/test_processor.py::test_video_processor_init PASSED                [ 33%]
tests/test_processor.py::test_video_processor_missing_file PASSED        [ 50%]
tests/test_processor.py::test_extract_actions_raises_without_api_key PASSED [ 66%]
tests/test_processor.py::test_action_prompt_structure PASSED             [ 83%]
tests/test_processor.py::test_api_app_creates SKIPPED (fastapi not i...) [100%]

========================= 5 passed, 1 skipped in 0.65s =========================
```

See `examples/sample-output.json` for what a full analysis of a user session looks like.
