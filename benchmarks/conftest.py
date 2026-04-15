"""Shared fixtures for benchmark tests."""
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _make_text_image(
    text,
    width=1920,
    height=1080,
    font_size=32,
    bg_color="white",
    text_color="black",
):
    """Create an RGB image with the given text rendered on it."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except (OSError, IOError):
            font = ImageFont.load_default()
    y = 40
    for line in text.split("\n"):
        draw.text((40, y), line, fill=text_color, font=font)
        y += font_size + 8
    return img


def pil_to_cv2(img):
    """Convert PIL Image to OpenCV BGR ndarray."""
    arr = np.array(img)
    return arr[:, :, ::-1].copy()


@pytest.fixture
def sample_frames():
    """Return a list of (expected_text, cv2_frame) tuples with known text."""
    texts = [
        "File  Edit  View  Insert  Format  Tools  Extensions  Help",
        "Google Chrome - New Tab\nhttps://www.google.com",
        "Microsoft Excel - Budget_2026.xlsx\nA1: Revenue  B1: $1,250,000  C1: Growth 12%",
        "Terminal -- bash\n$ git status\nOn branch main\nnothing to commit, working tree clean",
        "Slack - #engineering channel\nJohn: Can we deploy v2.1 today?\nSarah: Tests are passing, LGTM",
        "HubSpot CRM - Contacts\nMike Johnson  |  VP of Sales  |  mike@acme.com",
    ]
    return [(t, pil_to_cv2(_make_text_image(t))) for t in texts]


@pytest.fixture
def simple_ocr_frame():
    """A single frame with large, clear text for OCR accuracy testing."""
    text = "Hello World 12345"
    return text, pil_to_cv2(_make_text_image(text, font_size=64))


@pytest.fixture
def mock_llm_responses():
    """Pre-canned LLM responses for action extraction benchmarks."""
    return {
        "excel_browsing": json.dumps([
            {"id": "1", "tools": ["excel"], "action": ["viewing spreadsheet data"]},
            {"id": "2", "tools": ["excel"], "action": ["sorting column A"]},
        ]),
        "multi_tool": json.dumps([
            {"id": "1", "tools": ["chrome"], "action": ["navigating to dashboard"]},
            {"id": "2", "tools": ["hubspot"], "action": ["searching contacts"]},
            {"id": "3", "tools": ["gmail"], "action": ["composing email"]},
            {"id": "4", "tools": ["gmail"], "action": ["sending email"]},
        ]),
        "coding_session": json.dumps([
            {"id": "1", "tools": ["vscode"], "action": ["editing Python file"]},
            {"id": "2", "tools": ["terminal"], "action": ["running pytest"]},
            {"id": "3", "tools": ["github"], "action": ["opening pull request"]},
        ]),
        "malformed_markdown": '```json\n[{"id":"1","tools":["excel"],"action":["viewing"]}]\n```',
        "empty_actions": json.dumps([]),
    }
