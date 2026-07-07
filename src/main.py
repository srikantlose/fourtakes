"""FourTakes CLI entrypoint.

Two modes:

1. Submission (task) mode — the judged path. Runs when no input path is
   given and the tasks file exists (default /input/tasks.json), or when
   --tasks is passed explicitly:
       python -m src.main
       python -m src.main --tasks input/tasks.json --results out/results.json

2. Local dev mode — caption a local video file or directory:
       python -m src.main sample.mp4
       python -m src.main ./videos/ --output-dir results
       python -m src.main sample.mp4 --mock          # no API calls
"""

import argparse
import json
import sys
from pathlib import Path

from . import task_runner
from .config import load_config, load_prompts
from .logging_config import setup_logging
from .pipeline import FourTakesPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fourtakes",
        description=(
            "Generate styled captions (formal, sarcastic, humorous_tech, "
            "humorous_non_tech) for short video clips."
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=(
            "Path to a video file or a directory of videos (local dev mode). "
            "Omit to run submission mode against the tasks file."
        ),
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Tasks JSON file (default: TASKS_PATH env var or /input/tasks.json)",
    )
    parser.add_argument(
        "--results",
        default=None,
        help="Results JSON file (default: RESULTS_PATH env var or /output/results.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSON results in dev mode (default: OUTPUT_DIR env var)",
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
        help="Also print the combined results JSON to stdout (dev mode)",
    )
    return parser


def run_dev_mode(args, config, prompts) -> int:
    """Caption a local file or directory (original CLI behavior)."""
    output_dir = args.output_dir or config["output_dir"]
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


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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
    if args.tasks:
        config["tasks_path"] = args.tasks
    if args.results:
        config["results_path"] = args.results

    prompts = load_prompts(config["prompts_path"])

    if args.input:
        return run_dev_mode(args, config, prompts)

    # Submission mode: judged containers run with no arguments and the
    # harness mounts the tasks file at /input/tasks.json.
    if args.tasks or Path(config["tasks_path"]).exists():
        return task_runner.run(config, prompts)

    parser.print_help(sys.stderr)
    print(
        f"\nError: no input path given and tasks file not found at "
        f"{config['tasks_path']}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
