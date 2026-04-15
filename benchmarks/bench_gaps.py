"""
Benchmark: Gap analysis -- test patterns and apps that may not be handled.
"""
import os
import sys
import json
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from screen_recorder_analyzer.processor import VideoProcessor, extract_actions


COMMON_APPS = [
    "Google Sheets", "Google Docs", "Notion", "Jira", "Confluence",
    "Figma", "Zoom", "Microsoft Teams", "Salesforce", "Trello",
    "Asana", "Linear", "Discord", "WhatsApp Web", "Outlook",
    "Tableau", "PowerBI", "Postman", "DBeaver", "pgAdmin",
    "Docker Desktop", "AWS Console", "GCP Console", "Azure Portal",
]


def test_common_app_recognition():
    """Verify that OCR text for common apps passes through to the LLM prompt."""
    for app_name in COMMON_APPS:
        transcript = f"I opened {app_name} and worked with it"
        ocr_frames = [f"{app_name} - Main Window", f"{app_name} - Settings"]
        expected_response = json.dumps([
            {"id": "1", "tools": [app_name.lower().replace(" ", "_")], "action": [f"using {app_name}"]},
        ])
        with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
            m.return_value = expected_response
            results = {
                "transcript": transcript,
                "frame_analysis": [
                    {"frame_number": i * 30, "timestamp_sec": i, "status": "ok", "text": text}
                    for i, text in enumerate(ocr_frames)
                ],
            }
            actions = extract_actions(results)
            args, kwargs = m.call_args
            prompt = args[0] if args else kwargs.get("prompt", "")
            assert app_name in prompt, f"{app_name} not found in LLM prompt"


def test_no_transcript_handling():
    """Pipeline should work when audio transcription fails/is empty."""
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = json.dumps([{"id": "1", "tools": ["app"], "action": ["working"]}])
        results = {"transcript": "", "frame_analysis": [
            {"frame_number": 0, "timestamp_sec": 0, "status": "ok", "text": "Some UI text"},
        ]}
        actions = extract_actions(results)
    assert isinstance(actions, list)


def test_no_ocr_handling():
    """Pipeline should work when OCR produces no text."""
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = json.dumps([{"id": "1", "tools": ["app"], "action": ["working"]}])
        results = {"transcript": "User spoke about their work", "frame_analysis": []}
        actions = extract_actions(results)
    assert isinstance(actions, list)


def test_all_frames_skipped():
    """Pipeline should handle all frames having status=skipped."""
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = json.dumps([])
        results = {"transcript": "test", "frame_analysis": [
            {"frame_number": 0, "timestamp_sec": 0, "status": "skipped", "text": None},
            {"frame_number": 30, "timestamp_sec": 1, "status": "skipped", "text": None},
        ]}
        actions = extract_actions(results)
    assert isinstance(actions, list)


def test_unicode_text_in_ocr():
    """OCR text with unicode characters should not break the pipeline."""
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = json.dumps([{"id": "1", "tools": ["browser"], "action": ["viewing page"]}])
        results = {"transcript": "", "frame_analysis": [
            {"frame_number": 0, "timestamp_sec": 0, "status": "ok",
             "text": "Tableau - \u00e9v\u00e9nements \u2022 R\u00e9sum\u00e9 \u2013 Dashboard"},
        ]}
        actions = extract_actions(results)
    assert isinstance(actions, list)


def test_very_long_ocr_text():
    """Frames with extremely long OCR text should be truncated (500 char limit)."""
    long_text = "A" * 2000
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = json.dumps([])
        results = {"transcript": "", "frame_analysis": [
            {"frame_number": 0, "timestamp_sec": 0, "status": "ok", "text": long_text},
        ]}
        extract_actions(results)
        args, kwargs = m.call_args
        prompt = args[0] if args else kwargs.get("prompt", "")
        assert "A" * 2000 not in prompt, "OCR text not truncated in prompt"


def test_max_50_frames_in_prompt():
    """Only first 50 frames should be included in LLM prompt."""
    frames = [
        {"frame_number": i * 30, "timestamp_sec": i, "status": "ok", "text": f"Frame {i} text"}
        for i in range(100)
    ]
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = json.dumps([])
        results = {"transcript": "", "frame_analysis": frames}
        extract_actions(results)
        args, kwargs = m.call_args
        prompt = args[0] if args else kwargs.get("prompt", "")
        assert "Frame 50 text" not in prompt
        assert "Frame 99 text" not in prompt
        assert "Frame 0 text" in prompt


def test_llm_response_with_extra_whitespace():
    """LLM response with extra whitespace/newlines should still parse."""
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = '  \n  [{"id":"1","tools":["app"],"action":["test"]}]  \n  '
        results = {"transcript": "test", "frame_analysis": []}
        actions = extract_actions(results)
    assert len(actions) == 1


def test_llm_json_object_instead_of_array():
    """LLM returns a JSON object with an actions key -- extract_actions should
    unwrap it automatically (improvement: robust JSON parsing).
    """
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = json.dumps({
            "actions": [{"id": "1", "tools": ["app"], "action": ["test"]}]
        })
        results = {"transcript": "test", "frame_analysis": []}
        actions = extract_actions(results)
        assert isinstance(actions, list)
        assert len(actions) == 1
        assert actions[0]["tools"] == ["app"]


def test_gap_no_adaptive_thresholding():
    """Gap: current OCR uses basic grayscale, no adaptive thresholding."""
    pass


def test_gap_no_dark_theme_handling():
    """Gap: dark theme UIs (white text on dark bg) may have poor OCR."""
    pass


def test_gap_no_duplicate_frame_detection():
    """Gap: identical consecutive frames waste OCR compute."""
    pass


def test_gap_no_scene_change_detection():
    """Gap: frame_skip is fixed; scene-change detection would be smarter."""
    pass
