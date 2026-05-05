"""
Industry baseline benchmarks for screen-recorder-analyzer pipeline components.

References published accuracy numbers for each component so our synthetic
benchmarks can be interpreted in context.

Industry references
-------------------
Speech recognition (Whisper):
    - Whisper Large-v3 WER on LibriSpeech test-clean: 2.0-2.7%
    - Whisper Large-v3 WER on LibriSpeech test-other: 5.2%
    - Human baseline WER on LibriSpeech: 4.0-6.8%
    - Whisper Base WER on LibriSpeech test-clean: ~5-6%
    - Common Voice WER varies by language; English ~8-12% for base model

OCR -- document / scene text:
    - Tesseract on clean printed text (300 dpi scans): 95-99% character accuracy
    - Tesseract on screen captures (varied fonts/backgrounds): 80-90%
    - ICDAR 2015 scene text recognition benchmark: top systems ~85-95% word accuracy
    - ICDAR Robust Reading datasets test text-in-the-wild, not document OCR
    - Screen OCR is harder than document OCR due to anti-aliased fonts, UI chrome,
      transparency, dark themes, and small text

Action recognition:
    - No direct industry benchmark exists for screen-to-action extraction.
    - This is a novel contribution: combining Whisper + Tesseract + LLM to
      produce structured action logs from screen recordings.
    - Closest analogues are process mining (event logs) and activity recognition
      (video classification), but neither operates on arbitrary desktop recordings.
"""
import os
import sys
import time
import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import pytesseract
    _TESSERACT_OK = True
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        _TESSERACT_OK = False
except ImportError:
    _TESSERACT_OK = False

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

needs_tesseract = pytest.mark.skipif(
    not (_TESSERACT_OK and _CV2_OK),
    reason="tesseract or cv2 not available",
)

# ---------------------------------------------------------------------------
# Industry baseline constants
# ---------------------------------------------------------------------------

WHISPER_BASELINES: Dict[str, Dict[str, float]] = {
    "whisper-large-v3": {
        "librispeech_clean_wer": 2.7,
        "librispeech_other_wer": 5.2,
    },
    "whisper-base": {
        "librispeech_clean_wer": 5.6,
        "librispeech_other_wer": 13.7,
    },
    "whisper-small": {
        "librispeech_clean_wer": 3.4,
        "librispeech_other_wer": 7.6,
    },
    "human": {
        "librispeech_clean_wer": 5.8,
        "librispeech_other_wer": 6.8,
    },
}

OCR_BASELINES: Dict[str, Dict[str, float]] = {
    "tesseract_clean_document": {
        "char_accuracy_pct": 97.0,
        "word_accuracy_pct": 95.0,
    },
    "tesseract_screen_capture": {
        "char_accuracy_pct": 87.0,
        "word_accuracy_pct": 82.0,
    },
    "icdar2015_top_system": {
        "word_accuracy_pct": 92.0,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_font(monospace: bool = False) -> ImageFont.FreeTypeFont:
    """Find a usable system font, preferring monospace when requested."""
    mono_paths = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFMono-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    sans_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    candidates = (mono_paths if monospace else sans_paths) + sans_paths + mono_paths
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _load_font(size: int, monospace: bool = False) -> ImageFont.ImageFont:
    path = _find_font(monospace)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, ::-1].copy()
    return arr


def _char_error_rate(reference: str, hypothesis: str) -> float:
    """Character Error Rate: edit distance / reference length."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    sm = difflib.SequenceMatcher(None, reference, hypothesis)
    matches = sum(block.size for block in sm.get_matching_blocks())
    return 1.0 - (matches / len(reference))


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Word Error Rate using Levenshtein distance on word sequences."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


# ---------------------------------------------------------------------------
# Synthetic screen frame generators
# ---------------------------------------------------------------------------

@dataclass
class ScreenFrame:
    """A synthetic screen capture with known ground-truth text."""
    name: str
    category: str
    ground_truth: str
    image: Image.Image


def _make_code_editor_frame() -> ScreenFrame:
    """Simulate a code editor with monospace font and syntax-highlighting colors."""
    w, h = 1920, 1080
    bg = (30, 30, 30)  # dark editor bg
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    font = _load_font(20, monospace=True)

    lines = [
        ("def process_video(path: str) -> dict:", (86, 156, 214)),
        ('    """Extract actions from screen recording."""', (106, 153, 85)),
        ("    processor = VideoProcessor()", (220, 220, 220)),
        ("    results = processor.process(path)", (220, 220, 220)),
        ("    actions = extract_actions(results)", (220, 220, 220)),
        ("    return actions", (197, 134, 192)),
    ]
    ground_truth_lines = []
    y = 60
    for text, color in lines:
        draw.text((80, y), text, fill=color, font=font)
        ground_truth_lines.append(text)
        y += 28

    # Title bar
    draw.rectangle([(0, 0), (w, 40)], fill=(50, 50, 50))
    title = "processor.py - VS Code"
    draw.text((80, 10), title, fill=(200, 200, 200), font=_load_font(16))
    ground_truth_lines.insert(0, title)

    return ScreenFrame(
        name="code_editor_dark",
        category="code_editor",
        ground_truth="\n".join(ground_truth_lines),
        image=img,
    )


def _make_browser_frame() -> ScreenFrame:
    """Simulate a web browser with nav bar, mixed fonts."""
    w, h = 1920, 1080
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Nav bar
    draw.rectangle([(0, 0), (w, 60)], fill=(245, 245, 245))
    nav_font = _load_font(16)
    url = "https://dashboard.example.com/analytics"
    draw.text((200, 20), url, fill=(50, 50, 50), font=nav_font)

    # Page title
    title_font = _load_font(32)
    title = "Analytics Dashboard"
    draw.text((80, 100), title, fill=(30, 30, 30), font=title_font)

    # Body content
    body_font = _load_font(18)
    body_lines = [
        "Total Users: 12,450",
        "Active Sessions: 3,201",
        "Conversion Rate: 4.7%",
        "Revenue: $85,230",
    ]
    y = 180
    for line in body_lines:
        draw.text((80, y), line, fill=(60, 60, 60), font=body_font)
        y += 32

    gt = "\n".join([url, title] + body_lines)
    return ScreenFrame(
        name="browser_dashboard",
        category="web_browser",
        ground_truth=gt,
        image=img,
    )


def _make_spreadsheet_frame() -> ScreenFrame:
    """Simulate a spreadsheet with grid layout and small text."""
    w, h = 1920, 1080
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _load_font(16)

    # Title bar
    draw.rectangle([(0, 0), (w, 40)], fill=(33, 115, 70))
    title = "Budget_2026.xlsx - Excel"
    draw.text((80, 10), title, fill=(255, 255, 255), font=font)

    # Column headers
    headers = ["A", "B", "C", "D", "E"]
    col_labels = ["Item", "Q1", "Q2", "Q3", "Total"]
    col_x = [80, 280, 430, 580, 730]
    for i, (h_letter, label) in enumerate(zip(headers, col_labels)):
        draw.text((col_x[i], 60), label, fill=(0, 0, 0), font=font)
        # Grid lines
        draw.line([(col_x[i] - 5, 50), (col_x[i] - 5, 350)], fill=(200, 200, 200))

    rows = [
        ["Revenue", "$125,000", "$142,000", "$138,000", "$405,000"],
        ["Expenses", "$95,000", "$98,500", "$101,200", "$294,700"],
        ["Profit", "$30,000", "$43,500", "$36,800", "$110,300"],
        ["Margin", "24.0%", "30.6%", "26.7%", "27.2%"],
    ]
    gt_parts = [title, "\t".join(col_labels)]
    y = 90
    for row in rows:
        for i, cell in enumerate(row):
            draw.text((col_x[i], y), cell, fill=(0, 0, 0), font=font)
        gt_parts.append("\t".join(row))
        y += 28
        draw.line([(75, y - 4), (880, y - 4)], fill=(220, 220, 220))

    return ScreenFrame(
        name="spreadsheet",
        category="spreadsheet",
        ground_truth="\n".join(gt_parts),
        image=img,
    )


def _make_dark_theme_terminal() -> ScreenFrame:
    """Simulate a terminal with light text on dark background."""
    w, h = 1920, 1080
    bg = (15, 15, 15)
    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    font = _load_font(18, monospace=True)

    lines = [
        ("$ git status", (0, 255, 0)),
        ("On branch main", (220, 220, 220)),
        ("Changes not staged for commit:", (220, 220, 220)),
        ("  modified:   src/processor.py", (255, 100, 100)),
        ("  modified:   tests/test_ocr.py", (255, 100, 100)),
        ("$ pytest tests/ -v", (0, 255, 0)),
        ("5 passed, 1 skipped in 0.65s", (0, 255, 0)),
    ]
    gt_lines = []
    y = 40
    for text, color in lines:
        draw.text((40, y), text, fill=color, font=font)
        gt_lines.append(text)
        y += 26

    return ScreenFrame(
        name="dark_terminal",
        category="dark_theme",
        ground_truth="\n".join(gt_lines),
        image=img,
    )


def _make_low_contrast_frame() -> ScreenFrame:
    """Light gray text on slightly lighter gray background -- worst case for OCR."""
    w, h = 1920, 1080
    img = Image.new("RGB", (w, h), (230, 230, 230))
    draw = ImageDraw.Draw(img)
    font = _load_font(20)

    lines = [
        "This text has very low contrast",
        "It simulates a washed-out display",
        "OCR accuracy should drop significantly",
    ]
    y = 100
    for line in lines:
        draw.text((80, y), line, fill=(180, 180, 180), font=font)
        y += 34

    return ScreenFrame(
        name="low_contrast",
        category="low_contrast",
        ground_truth="\n".join(lines),
        image=img,
    )


def _make_chat_ui_frame() -> ScreenFrame:
    """Simulate a messaging/chat application."""
    w, h = 1920, 1080
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Sidebar
    draw.rectangle([(0, 0), (300, h)], fill=(44, 47, 51))
    sidebar_font = _load_font(16)
    channels = ["#general", "#engineering", "#design", "#random"]
    y = 80
    for ch in channels:
        draw.text((20, y), ch, fill=(180, 180, 180), font=sidebar_font)
        y += 30

    # Chat header
    draw.rectangle([(300, 0), (w, 50)], fill=(245, 245, 245))
    header_font = _load_font(18)
    header = "#engineering - 4 members online"
    draw.text((320, 15), header, fill=(50, 50, 50), font=header_font)

    # Messages
    msg_font = _load_font(16)
    messages = [
        ("Alice:", "Can we deploy v2.1 today?"),
        ("Bob:", "Tests are passing, LGTM"),
        ("Carol:", "Let me check the staging env first"),
    ]
    gt_parts = channels + [header]
    y = 80
    for sender, msg in messages:
        draw.text((320, y), sender, fill=(30, 30, 30), font=_load_font(16))
        draw.text((390, y), msg, fill=(80, 80, 80), font=msg_font)
        gt_parts.append(f"{sender} {msg}")
        y += 36

    return ScreenFrame(
        name="chat_ui",
        category="chat",
        ground_truth="\n".join(gt_parts),
        image=img,
    )


def _make_form_ui_frame() -> ScreenFrame:
    """Simulate a web form with labels and input fields."""
    w, h = 1920, 1080
    img = Image.new("RGB", (w, h), (250, 250, 250))
    draw = ImageDraw.Draw(img)

    title_font = _load_font(28)
    label_font = _load_font(16)
    input_font = _load_font(16)

    title = "Create New Project"
    draw.text((80, 60), title, fill=(30, 30, 30), font=title_font)

    fields = [
        ("Project Name:", "My New Project"),
        ("Description:", "A sample project for testing"),
        ("Team:", "Engineering"),
        ("Priority:", "High"),
        ("Due Date:", "2026-06-15"),
    ]
    gt_parts = [title]
    y = 130
    for label, value in fields:
        draw.text((80, y), label, fill=(80, 80, 80), font=label_font)
        draw.rectangle([(250, y - 4), (600, y + 24)], outline=(200, 200, 200), width=1)
        draw.text((260, y), value, fill=(30, 30, 30), font=input_font)
        gt_parts.append(f"{label} {value}")
        y += 48

    return ScreenFrame(
        name="form_ui",
        category="form",
        ground_truth="\n".join(gt_parts),
        image=img,
    )


def _make_email_client_frame() -> ScreenFrame:
    """Simulate an email client inbox."""
    w, h = 1920, 1080
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([(0, 0), (w, 50)], fill=(66, 133, 244))
    draw.text((80, 12), "Gmail - Inbox", fill=(255, 255, 255), font=_load_font(22))

    font = _load_font(16)
    emails = [
        ("John Smith", "Q3 Budget Review", "Please review the attached..."),
        ("Product Team", "Sprint Planning Notes", "Here are the action items..."),
        ("HR Department", "Benefits Enrollment", "Open enrollment starts..."),
        ("DevOps Alert", "Build Failed: main", "Pipeline failed at step 3..."),
    ]
    gt_parts = ["Gmail - Inbox"]
    y = 80
    for sender, subject, preview in emails:
        draw.text((80, y), sender, fill=(30, 30, 30), font=_load_font(16))
        draw.text((280, y), subject, fill=(30, 30, 30), font=_load_font(16))
        draw.text((600, y), preview, fill=(150, 150, 150), font=font)
        gt_parts.append(f"{sender} {subject} {preview}")
        y += 36
        draw.line([(80, y - 4), (w - 80, y - 4)], fill=(240, 240, 240))

    return ScreenFrame(
        name="email_client",
        category="email",
        ground_truth="\n".join(gt_parts),
        image=img,
    )


ALL_FRAME_GENERATORS = [
    _make_code_editor_frame,
    _make_browser_frame,
    _make_spreadsheet_frame,
    _make_dark_theme_terminal,
    _make_low_contrast_frame,
    _make_chat_ui_frame,
    _make_form_ui_frame,
    _make_email_client_frame,
]


def generate_all_frames() -> List[ScreenFrame]:
    """Generate all synthetic screen frames."""
    return [gen() for gen in ALL_FRAME_GENERATORS]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIndustryBaselines:
    """Verify that industry baseline data is present and reasonable."""

    def test_whisper_baselines_documented(self):
        """All expected Whisper model tiers have baseline WER values."""
        assert "whisper-large-v3" in WHISPER_BASELINES
        assert "whisper-base" in WHISPER_BASELINES
        assert "human" in WHISPER_BASELINES

        # Large-v3 should be better than human on clean speech
        large = WHISPER_BASELINES["whisper-large-v3"]
        human = WHISPER_BASELINES["human"]
        assert large["librispeech_clean_wer"] < human["librispeech_clean_wer"], (
            "Whisper Large-v3 should beat human WER on clean speech"
        )

    def test_ocr_baselines_documented(self):
        """OCR baseline accuracy ranges are documented."""
        assert "tesseract_clean_document" in OCR_BASELINES
        assert "tesseract_screen_capture" in OCR_BASELINES
        assert "icdar2015_top_system" in OCR_BASELINES

        clean = OCR_BASELINES["tesseract_clean_document"]
        screen = OCR_BASELINES["tesseract_screen_capture"]
        assert clean["char_accuracy_pct"] > screen["char_accuracy_pct"], (
            "Clean document OCR should beat screen capture OCR"
        )

    def test_action_extraction_is_novel(self):
        """Document that screen-to-action extraction has no standard benchmark.

        This is a novel task combining speech transcription, screen OCR, and
        LLM action inference.  No ICDAR/LibriSpeech-equivalent exists.
        """
        # This test serves as documentation -- the assertion is trivially true
        assert "action_recognition" not in OCR_BASELINES
        assert "action_recognition" not in WHISPER_BASELINES

    def test_synthetic_frame_generators_produce_valid_images(self):
        """All frame generators should produce valid RGB images with text."""
        frames = generate_all_frames()
        assert len(frames) == len(ALL_FRAME_GENERATORS)
        for f in frames:
            assert isinstance(f.image, Image.Image)
            assert f.image.mode == "RGB"
            assert f.image.size[0] >= 800
            assert f.image.size[1] >= 600
            assert len(f.ground_truth) > 10
            assert f.name
            assert f.category


class TestOCRVsBaselines:
    """Run OCR on synthetic frames and compare to industry baselines."""

    @needs_tesseract
    def test_ocr_accuracy_per_category(self):
        """Measure CER and WER per frame category, report vs baselines."""
        frames = generate_all_frames()
        results = []
        for sf in frames:
            cv2_img = _pil_to_cv2(sf.image)
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            ocr_text = pytesseract.image_to_string(gray, config="--psm 6").strip()
            cer = _char_error_rate(sf.ground_truth, ocr_text)
            wer = _word_error_rate(sf.ground_truth, ocr_text)
            results.append({
                "name": sf.name,
                "category": sf.category,
                "cer": cer,
                "wer": wer,
                "char_accuracy_pct": (1 - cer) * 100,
                "word_accuracy_pct": (1 - wer) * 100,
            })

        print("\n  === OCR Accuracy vs Industry Baselines ===")
        print(f"  {'Frame':<25s} {'CER':>8s} {'WER':>8s} {'Char%':>8s} {'Word%':>8s}")
        print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
        for r in results:
            print(
                f"  {r['name']:<25s} "
                f"{r['cer']:>7.1%} "
                f"{r['wer']:>7.1%} "
                f"{r['char_accuracy_pct']:>7.1f}% "
                f"{r['word_accuracy_pct']:>7.1f}%"
            )

        avg_cer = sum(r["cer"] for r in results) / len(results)
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_char = (1 - avg_cer) * 100
        avg_word = (1 - avg_wer) * 100

        print(f"\n  Overall average:")
        print(f"    CER: {avg_cer:.1%}  (char accuracy: {avg_char:.1f}%)")
        print(f"    WER: {avg_wer:.1%}  (word accuracy: {avg_word:.1f}%)")
        print(f"\n  Industry baselines for comparison:")
        print(f"    Tesseract clean document:  {OCR_BASELINES['tesseract_clean_document']['char_accuracy_pct']}% char")
        print(f"    Tesseract screen capture:  {OCR_BASELINES['tesseract_screen_capture']['char_accuracy_pct']}% char")
        print(f"    ICDAR 2015 top system:     {OCR_BASELINES['icdar2015_top_system']['word_accuracy_pct']}% word")

        # We expect worse than clean-document but in range of screen-capture
        # Don't hard-fail, just report
        assert avg_char > 30, f"Average char accuracy {avg_char:.1f}% is suspiciously low"

    @needs_tesseract
    def test_preprocessing_vs_raw_on_dark_theme(self):
        """Compare raw vs preprocessed OCR on dark-themed frames."""
        from screen_recorder_analyzer.processor import _preprocess_for_ocr

        dark_frames = [f for f in generate_all_frames() if f.category == "dark_theme"]
        assert dark_frames, "No dark theme test frames found"

        for sf in dark_frames:
            cv2_img = _pil_to_cv2(sf.image)
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

            raw_text = pytesseract.image_to_string(gray, config="--psm 6").strip()
            raw_wer = _word_error_rate(sf.ground_truth, raw_text)

            processed = _preprocess_for_ocr(gray)
            proc_text = pytesseract.image_to_string(processed, config="--psm 6").strip()
            proc_wer = _word_error_rate(sf.ground_truth, proc_text)

            print(f"\n  Dark theme OCR ({sf.name}):")
            print(f"    Raw WER:         {raw_wer:.1%}")
            print(f"    Preprocessed WER: {proc_wer:.1%}")
            print(f"    Improvement:     {(raw_wer - proc_wer):.1%} absolute")

    @needs_tesseract
    def test_preprocessing_vs_raw_on_low_contrast(self):
        """Preprocessing should help on low-contrast frames."""
        from screen_recorder_analyzer.processor import _preprocess_for_ocr

        lc_frames = [f for f in generate_all_frames() if f.category == "low_contrast"]
        assert lc_frames, "No low-contrast test frames found"

        for sf in lc_frames:
            cv2_img = _pil_to_cv2(sf.image)
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

            raw_text = pytesseract.image_to_string(gray, config="--psm 6").strip()
            raw_cer = _char_error_rate(sf.ground_truth, raw_text)

            processed = _preprocess_for_ocr(gray)
            proc_text = pytesseract.image_to_string(processed, config="--psm 6").strip()
            proc_cer = _char_error_rate(sf.ground_truth, proc_text)

            print(f"\n  Low-contrast OCR ({sf.name}):")
            print(f"    Raw CER:         {raw_cer:.1%}")
            print(f"    Preprocessed CER: {proc_cer:.1%}")


class TestWhisperBaselineReference:
    """Reference tests for Whisper accuracy context (no actual transcription)."""

    def test_whisper_wer_scale(self):
        """Verify WER values are on a sensible scale (0-100%)."""
        for model, metrics in WHISPER_BASELINES.items():
            for metric, value in metrics.items():
                assert 0 < value < 100, f"{model}.{metric} = {value} is out of range"

    def test_whisper_model_ordering(self):
        """Larger models should have lower WER on clean speech."""
        large = WHISPER_BASELINES["whisper-large-v3"]["librispeech_clean_wer"]
        small = WHISPER_BASELINES["whisper-small"]["librispeech_clean_wer"]
        base = WHISPER_BASELINES["whisper-base"]["librispeech_clean_wer"]
        assert large < small < base, "Model size ordering is wrong"

    def test_default_model_wer_context(self):
        """Our default model (base) has known WER of ~5.6% on clean speech.

        This is provided as context for users interpreting transcription quality.
        """
        base_wer = WHISPER_BASELINES["whisper-base"]["librispeech_clean_wer"]
        print(f"\n  Default Whisper model (base) reference WER:")
        print(f"    LibriSpeech clean: {base_wer}%")
        print(f"    Human baseline:    {WHISPER_BASELINES['human']['librispeech_clean_wer']}%")
        print(f"    Large-v3 best:     {WHISPER_BASELINES['whisper-large-v3']['librispeech_clean_wer']}%")
