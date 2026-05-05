# screen-recorder-analyzer

Built by the [Nometria](https://nometria.com) team. We help developers take apps built with AI tools (Lovable, Bolt, Base44, Replit) to production — handling deployment to AWS, security, scaling, and giving you full code ownership. [Learn more →](https://nometria.com)

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

## Industry Benchmark Context

This project combines three ML pipelines (speech recognition, OCR, action extraction). Below are industry-standard reference scores for interpreting our benchmark results.

### Speech Recognition (Whisper)

| Model | LibriSpeech Clean WER | LibriSpeech Other WER | Notes |
|-------|----------------------|----------------------|-------|
| Whisper Large-v3 | 2.0-2.7% | 5.2% | Near-human accuracy |
| Whisper Small | 3.4% | 7.6% | Good balance of speed/accuracy |
| Whisper Base (our default) | 5.6% | 13.7% | Fast, suitable for real-time |
| Human baseline | 4.0-6.8% | 6.8% | Professional transcriptionists |

WER = Word Error Rate (lower is better). LibriSpeech is the standard benchmark for English speech recognition.

### Screen OCR (Tesseract)

| Scenario | Character Accuracy | Word Accuracy | Notes |
|----------|-------------------|---------------|-------|
| Clean printed text (300 dpi) | 95-99% | 95%+ | Ideal conditions |
| Screen captures (mixed fonts) | 80-90% | 82-90% | Our target scenario |
| ICDAR 2015 scene text (top systems) | -- | 85-95% | Text-in-the-wild, different from screen OCR |
| Our pipeline (with preprocessing) | Measured in benchmarks | Measured in benchmarks | Dark theme inversion + adaptive threshold |

ICDAR (International Conference on Document Analysis and Recognition) provides standard benchmarks for scene text recognition, but screen OCR (anti-aliased fonts, UI chrome, dark themes) is a distinct challenge with no single standard benchmark.

### Action Extraction

Screen-to-action extraction is a **novel task** with no industry-standard benchmark. Our pipeline combines Whisper transcription + Tesseract OCR + LLM inference to produce structured action logs from arbitrary desktop recordings. The closest analogues are:

- **Process mining**: operates on structured event logs, not raw video
- **Activity recognition**: classifies video into predefined categories, not arbitrary desktop workflows
- **UI understanding**: emerging research area, no established benchmark datasets for desktop recordings

---

## Benchmark Results

Run benchmarks with:

```bash
pip install -e ".[dev]"
pytest benchmarks/bench_actions.py benchmarks/bench_pipeline.py benchmarks/bench_gaps.py benchmarks/bench_ocr.py -v -s
```

### Summary (37 tests: 33 passed, 4 skipped)

| Category | Tests | Passed | Skipped | Notes |
|----------|-------|--------|---------|-------|
| OCR accuracy | 5 | 1 | 4 | Tesseract OCR tests skipped without `tesseract` binary |
| Action categorization | 9 | 9 | 0 | Mock LLM responses, JSON parsing, edge cases |
| Pipeline speed | 10 | 10 | 0 | Init ~3ms, action extraction <1ms (excl. LLM) |
| Gap analysis | 13 | 13 | 0 | 24 common apps verified, edge cases covered |

### Key metrics

- **Processor init**: ~3ms avg (Whisper model is lazy-loaded)
- **Action extraction** (excl. LLM latency): <1ms avg over 20 iterations
- **Frame skip configs**: frame_skip=29 (default) analyzes ~10 frames per 10s of 30fps video
- **Common apps tested**: 24 applications (Google Sheets, Notion, Jira, Figma, Slack, VS Code, etc.)

### Improvements implemented

Based on benchmark gap analysis, the following improvements were added to `processor.py`:

1. **OCR preprocessing** -- Adaptive thresholding and bilateral filtering improve text extraction on noisy or low-contrast frames
2. **Dark theme detection** -- Automatically inverts dark-themed UI screenshots (light text on dark background) before OCR
3. **Frame deduplication** -- Skips near-identical consecutive frames to avoid redundant OCR processing
4. **Robust LLM JSON parsing** -- Handles JSON wrapped in objects (`{"actions": [...]}`) in addition to bare arrays, and extracts from common wrapper keys

### Identified gaps (documented in benchmarks)

- Scene-change-based keyframe selection would be more efficient than fixed frame_skip
- Multi-language OCR could be auto-detected from frame content
- No confidence scoring on individual OCR results

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

