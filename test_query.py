#!/usr/bin/env python
"""CLI for querying the Financial Document Intelligence Agent.

Usage:
    python test_query.py "What was ING's net profit in 2024?"
    python test_query.py "Wat was de netto winst van ING in 2024?"
    python test_query.py "What are the main risk factors?" --doc 2024-ing-bank-nv-annual-report.pdf
"""
from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from src.generation.answer_generator import AnswerGenerator
from src.retrieval.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Financial Document Intelligence Agent")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument(
        "--doc",
        default=None,
        metavar="FILENAME",
        help="Restrict search to a specific document (e.g. 2024-ing-bank-nv-annual-report.pdf)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of chunks to pass to the generator (default: 5)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Question: {args.question}")
    if args.doc:
        print(f"Document filter: {args.doc}")
    print("="*60)

    # --- Retrieval ---
    print("\n[1/2] Retrieving relevant chunks …")
    t0 = time.time()
    retriever = Retriever()
    chunks = retriever.retrieve(
        args.question,
        top_k_initial=15,
        top_k_final=args.top_k,
        document_filter=args.doc,
    )
    retrieve_ms = (time.time() - t0) * 1000

    if not chunks:
        print("No relevant chunks found in the vector store.")
        print("Have you run the ingestion pipeline? See src/ingestion/pipeline.py")
        sys.exit(1)

    print(f"  Retrieved {len(chunks)} chunks in {retrieve_ms:.0f}ms")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] {c.document_name}, p.{c.page_number}  (score={c.score:.3f})")

    # --- Generation ---
    print("\n[2/2] Generating answer with Gemini 2.5 Pro …")
    t1 = time.time()
    generator = AnswerGenerator()
    response = generator.generate(args.question, chunks)
    generate_ms = (time.time() - t1) * 1000

    # --- Output ---
    print(f"\n{'='*60}")
    print("ANSWER")
    print("="*60)
    print(response.answer)

    if response.citations:
        print(f"\n{'─'*60}")
        print(f"CITATIONS  ({len(response.citations)})")
        print("─"*60)
        for c in response.citations:
            print(f"  • {c.document_name}  —  Page {c.page_number}")
            if c.source_text:
                preview = c.source_text[:120].replace("\n", " ")
                print(f"    \"{preview}…\"")

    print(f"\n{'─'*60}")
    print(f"Model: {response.model_used}  |  Chunks used: {response.chunks_used}")
    print(f"Retrieval: {retrieve_ms:.0f}ms  |  Generation: {generate_ms:.0f}ms")
    print("─"*60)


if __name__ == "__main__":
    main()
