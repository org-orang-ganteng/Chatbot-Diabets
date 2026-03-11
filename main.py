from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from src.bio_rag.pipeline import BioRAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bio-RAG: diabetes-focused evidence-based QA with hallucination scoring"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Can vitamin D help reduce complications in diabetes?",
        help="Medical question to answer.",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    pipe = BioRAGPipeline()
    result = pipe.ask(args.question)

    print(json.dumps(result.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
