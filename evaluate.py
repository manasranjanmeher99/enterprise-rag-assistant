"""Check retrieval (and optionally answer/citation quality) against eval_questions.json.

Usage:
    python evaluate.py              # retrieval only: fast, no API calls
    python evaluate.py --answers    # also generate answers with the LLM and check citations
"""
import argparse
import json
from pathlib import Path

from generator import NO_ANSWER, cited_numbers, stream_answer
from retriever import retrieve

DEFAULT_FILE = Path(__file__).resolve().parent / "eval_questions.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--answers", action="store_true", help="also test generated answers")
    parser.add_argument("--file", default=str(DEFAULT_FILE))
    args = parser.parse_args()

    items = json.loads(Path(args.file).read_text(encoding="utf-8"))
    retrieval_pass = answer_pass = 0

    for item in items:
        question = item["question"]
        expected = item.get("expected_source")  # None means "should NOT be answerable"
        chunks = retrieve(question)
        found = sorted({c.source for c in chunks})

        ok = (not chunks) if expected is None else (expected in found)
        retrieval_pass += ok
        print(f"[{'PASS' if ok else 'FAIL'}] retrieval | {question}")
        if not ok:
            print(f"         expected={expected}  got={found}")

        if args.answers:
            answer = "".join(stream_answer(question, [], chunks)) if chunks else NO_ANSWER
            if expected is None:
                good = "couldn't find" in answer.lower()
            else:
                keywords = [k.lower() for k in item.get("expected_keywords", [])]
                good = (
                    bool(cited_numbers(answer, len(chunks)))
                    and all(k in answer.lower() for k in keywords)
                )
            answer_pass += good
            print(f"[{'PASS' if good else 'FAIL'}] answer    | {answer[:120]!r}")

    print(f"\nRetrieval: {retrieval_pass}/{len(items)}")
    if args.answers:
        print(f"Answers:   {answer_pass}/{len(items)}")


if __name__ == "__main__":
    main()
