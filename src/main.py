"""FourTakes CLI entrypoint.

Usage:
    python -m src.main <video-file-or-directory> [options]

Examples:
    python -m src.main sample.mp4
    python -m src.main ./videos/ --output-dir results
    python -m src.main sample.mp4 --mock          # no API calls
    python -m src.main sample.mp4 --model accounts/fireworks/models/some-model
"""

import argparse
import json
import sys

from .config import load_config, load_prompts
from .logging_config import setup_logging
from .pipeline import FourTakesPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fourtakes",
        description=(
            "Generate four styled captions (formal, sarcastic, humorous-tech, "
            "humorous-non-tech) for short video clips."
        ),
    )
    parser.add_argument(
        "input",
        help="Path to a video file or a directory of videos",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSON results (default: OUTPUT_DIR env var or 'output')",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the caption model (default: FIREWORKS_CAPTION_MODEL env var)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode: no real API calls, canned captions",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio extraction and transcription",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=None,
        help="Seconds between extracted frames (default: FRAME_INTERVAL_SECONDS)",
    )
    parser.add_argument(
        "--print-results",
        action="store_true",
        help="Also print the combined results JSON to stdout",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    setup_logging()
    config = load_config()

    # CLI flags override env config (still never hardcoded)
    if args.model:
        config["fireworks_caption_model"] = args.model
    if args.mock:
        config["mock_mode"] = True
    if args.no_audio:
        config["enable_audio_transcription"] = False
    if args.frame_interval:
        config["frame_interval_seconds"] = args.frame_interval
    output_dir = args.output_dir or config["output_dir"]

    prompts = load_prompts(config["prompts_path"])
    pipeline = FourTakesPipeline(config, prompts)

    try:
        results = pipeline.process_path(args.input)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    written = pipeline.write_results(results, output_dir)

    failures = [r for r in results if r["status"] != "ok"]
    print(
        f"\nProcessed {len(results)} video(s): "
        f"{len(results) - len(failures)} ok, {len(failures)} failed"
    )
    for path in written:
        print(f"  wrote {path}")
    if failures:
        for r in failures:
            print(f"  FAILED {r['video_id']}: {r.get('error')}")

    if args.print_results:
        print(json.dumps(results, indent=2, ensure_ascii=False))

    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
