#!/usr/bin/env python3
"""
screen-analyze CLI

Usage:
  screen-analyze video.mp4
  screen-analyze video.mp4 --whisper large --max-frames 50 --format json
  screen-analyze video.mp4 --no-actions   # skip GPT step
"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="screen-analyze",
        description="Analyze a screen recording: transcribe audio, OCR frames, extract user actions.",
    )
    parser.add_argument("video", help="Path to video file (mp4, mov, mkv, ...)")
    parser.add_argument("--whisper", default="base", metavar="MODEL",
                        help="Whisper model size: tiny|base|small|medium|large (default: base)")
    parser.add_argument("--frame-skip", type=int, default=29, metavar="N",
                        help="Analyze every N+1 frames (default: 29 = every 30th)")
    parser.add_argument("--max-frames", type=int, default=100, metavar="N",
                        help="Max frames to OCR (default: 100)")
    parser.add_argument("--ocr-lang", default="eng", metavar="LANG",
                        help="Tesseract language code (default: eng)")
    parser.add_argument("--no-actions", action="store_true",
                        help="Skip GPT action extraction step")
    parser.add_argument("--format", choices=["json", "text"], default="text",
                        help="Output format (default: text)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    from .processor import VideoProcessor, extract_actions

    processor = VideoProcessor(
        whisper_model_size=args.whisper,
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
