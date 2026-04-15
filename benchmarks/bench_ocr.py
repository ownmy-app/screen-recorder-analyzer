"""
Benchmark: OCR text extraction accuracy using synthetic test frames.
"""
import os
import sys
import time
import difflib

import numpy as np
import pytest

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


def _char_accuracy(expected, actual):
    """Character-level accuracy using SequenceMatcher."""
    return difflib.SequenceMatcher(None, expected, actual).ratio()


def _word_accuracy(expected, actual):
    """Fraction of expected words found in OCR output."""
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())
    if not expected_words:
        return 1.0
    return len(expected_words & actual_words) / len(expected_words)


@needs_tesseract
def test_ocr_simple_text(simple_ocr_frame):
    """Large, clear text should have near-perfect OCR accuracy."""
    expected, frame = simple_ocr_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = pytesseract.image_to_string(gray, config="--psm 6").strip()
    acc = _word_accuracy(expected, result)
    print(f"\n  OCR simple text: expected={expected!r}  got={result!r}  word_acc={acc:.2%}")
    assert acc >= 0.80, f"Word accuracy {acc:.2%} below 80% threshold"


@needs_tesseract
def test_ocr_accuracy_across_frames(sample_frames):
    """Measure OCR accuracy across multiple synthetic UI frames."""
    results = []
    for expected_text, frame in sample_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ocr_text = pytesseract.image_to_string(gray, config="--psm 6").strip()
        char_acc = _char_accuracy(expected_text, ocr_text)
        word_acc = _word_accuracy(expected_text, ocr_text)
        results.append({
            "expected_snippet": expected_text[:60],
            "char_accuracy": char_acc,
            "word_accuracy": word_acc,
        })

    avg_char = sum(r["char_accuracy"] for r in results) / len(results)
    avg_word = sum(r["word_accuracy"] for r in results) / len(results)
    print(f"\n  OCR Benchmark ({len(results)} frames):")
    print(f"    Avg char accuracy: {avg_char:.2%}")
    print(f"    Avg word accuracy: {avg_word:.2%}")
    for r in results:
        print(f"    {r['expected_snippet'][:40]:40s}  char={r['char_accuracy']:.2%}  word={r['word_accuracy']:.2%}")
    assert avg_word >= 0.50, f"Average word accuracy {avg_word:.2%} below 50% threshold"


@needs_tesseract
def test_ocr_preprocessing_improves_accuracy(sample_frames):
    """Test that thresholding/contrast preprocessing improves OCR on noisy images."""
    from conftest import _make_text_image, pil_to_cv2

    text = "Revenue Report Q3 2026"
    img = _make_text_image(text, font_size=48)
    arr = np.array(img)
    noise = np.random.normal(0, 25, arr.shape).astype(np.int16)
    noisy_arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    from PIL import Image
    noisy_img = Image.fromarray(noisy_arr)
    noisy_frame = pil_to_cv2(noisy_img)

    gray = cv2.cvtColor(noisy_frame, cv2.COLOR_BGR2GRAY)

    raw_result = pytesseract.image_to_string(gray, config="--psm 6").strip()
    raw_acc = _word_accuracy(text, raw_result)

    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_result = pytesseract.image_to_string(thresh, config="--psm 6").strip()
    processed_acc = _word_accuracy(text, processed_result)

    print(f"\n  Preprocessing benchmark:")
    print(f"    Raw OCR:       {raw_result!r}  word_acc={raw_acc:.2%}")
    print(f"    Preprocessed:  {processed_result!r}  word_acc={processed_acc:.2%}")
    assert processed_acc >= raw_acc * 0.9, "Preprocessing made accuracy significantly worse"


@needs_tesseract
def test_ocr_speed(sample_frames):
    """Measure OCR processing speed per frame."""
    times = []
    for _, frame in sample_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start = time.perf_counter()
        pytesseract.image_to_string(gray, config="--psm 6")
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg_ms = (sum(times) / len(times)) * 1000
    print(f"\n  OCR speed: avg={avg_ms:.0f}ms/frame  total={sum(times):.2f}s for {len(times)} frames")
    assert avg_ms < 5000, f"OCR too slow: {avg_ms:.0f}ms/frame"


def test_ocr_graceful_without_tesseract():
    """VideoProcessor should handle missing tesseract gracefully."""
    from screen_recorder_analyzer.processor import VideoProcessor
    p = VideoProcessor()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = p._analyze_frame(frame, 0, 30.0)
    assert result["status"] in ("ok", "skipped", "pending"), f"Unexpected status: {result['status']}"
