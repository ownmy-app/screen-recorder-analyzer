#!/usr/bin/env python3
"""
screen-analyze CLI - one command does it all.

Usage:
  screen-analyze video.mp4                              # transcribe + OCR + actions
  screen-analyze video.mp4 --whisper-backend api        # use OpenAI Whisper API (no local model)
  screen-analyze video.mp4 --whisper large --format json
  screen-analyze video.mp4 --no-actions                 # skip LLM action extraction step
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="screen-analyze",
        description=(
            "Analyze a screen recording in one shot: "
            "transcribe audio, OCR keyframes, and extract user actions via LLM."
        ),
    )
    parser.add_argument("video", help="Path to video file (mp4, mov, mkv, ...)")
    parser.add_argument("--whisper", default="base", metavar="MODEL",
                        help="Whisper model size: tiny|base|small|medium|large (default: base)")
    parser.add_argument("--whisper-backend", default=None, choices=["local", "api"],
                        help="Whisper backend: 'local' (default, needs openai-whisper+torch) "
                             "or 'api' (uses OpenAI Whisper API, needs OPENAI_API_KEY)")
    parser.add_argument("--frame-skip", type=int, default=29, metavar="N",
                        help="Analyze every N+1 frames (default: 29 = every 30th)")
    parser.add_argument("--max-frames", type=int, default=100, metavar="N",
                        help="Max frames to OCR (default: 100)")
    parser.add_argument("--ocr-lang", default="eng", metavar="LANG",
                        help="Tesseract language code (default: eng)")
    parser.add_argument("--no-actions", action="store_true",
                        help="Skip LLM action extraction step")
    parser.add_argument("--format", choices=["json", "text"], default="text",
                        help="Output format (default: text)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    from .processor import VideoProcessor, extract_actions

    # Resolve whisper backend: CLI flag > env var > "local"
    whisper_backend = (
        args.whisper_backend
        or os.environ.get("WHISPER_BACKEND", "local").lower().strip()
    )

    processor = VideoProcessor(
        whisper_model_size=args.whisper,
        whisper_backend=whisper_backend,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        ocr_lang=args.ocr_lang,
    )

    print("Processing video...", file=sys.stderr)
    results = processor.process(args.video)

    if not args.no_actions:
        try:
            results["structured_actions"] = extract_actions(results)
        except Exception as e:
            results["structured_actions_error"] = str(e)

    if args.format == "json":
        # Remove heavy frame_analysis from output
        results.pop("frame_analysis", None)
        print(json.dumps(results, indent=2))
        return

    # Human-readable
    meta = results.get("metadata", {})
    if meta:
        print(f"\nVideo: {args.video}")
        print(f"Duration: {meta.get('duration_seconds', '?'):.1f}s  "
              f"Resolution: {meta.get('width')}x{meta.get('height')}")

    transcript = results.get("transcript", "")
    if transcript and not transcript.startswith("["):
        print(f"\nTranscript ({len(transcript)} chars):")
        print(f"  {transcript[:300]}{'...' if len(transcript) > 300 else ''}")

    actions = results.get("structured_actions", [])
    if actions and isinstance(actions, list):
        print(f"\nExtracted actions ({len(actions)}):")
        for a in actions:
            tools = ", ".join(a.get("tools", []))
            acts = "; ".join(a.get("action", []))
            print(f"  [{a.get('id')}] {tools}: {acts}")


if __name__ == "__main__":
    main()
