"""Word-based chunking of CUADv1.json, using only Python's standard library.

Run from the project root: python3 scripts/contract_chunking.py
Outputs JSONL in data/ (which should be gitignored).
"""
import json
import re
from pathlib import Path

CHUNK_WORDS = 200
OVERLAP_WORDS = 50

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "CUADv1.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cuad_chunks.jsonl"


def make_chunks(context, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
    """Return (start_char, end_char, text) for overlapping word windows."""
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be >= 0 and smaller than chunk_words")

    word_spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", context)]
    step = chunk_words - overlap_words

    for start_word in range(0, len(word_spans), step):
        end_word = min(start_word + chunk_words, len(word_spans))
        start_char = word_spans[start_word][0]
        end_char = word_spans[end_word - 1][1]
        yield start_char, end_char, context[start_char:end_char]
        if end_word == len(word_spans):
            break


def get_annotations(qas):
    """Collect CUAD answer spans and their corresponding question labels."""
    annotations = []
    for qa in qas:
        for answer in qa.get("answers", []):
            text = answer.get("text", "")
            start = answer.get("answer_start", -1)
            if not text or not isinstance(start, int) or start < 0:
                continue
            annotations.append({
                "question": qa["question"],
                "answer_text": text,
                "start_char": start,
                "end_char": start + len(text),
            })
    return annotations


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"CUAD dataset not found: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as source:
        dataset = json.load(source)

    total_chunks = 0
    total_contracts = len(dataset["data"])
    partial_chunks = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for contract_index, contract in enumerate(dataset["data"]):
            for paragraph_index, paragraph in enumerate(contract["paragraphs"]):
                context = paragraph["context"]
                annotations = get_annotations(paragraph["qas"])

                for chunk_index, (start, end, text) in enumerate(make_chunks(context)):
                    # Preserve both fully contained and partially overlapping answers.
                    # A partially overlapping answer is NOT a complete positive example.
                    matched = []
                    for annotation in annotations:
                        if annotation["start_char"] < end and annotation["end_char"] > start:
                            fully_contained = (annotation["start_char"] >= start and
                                               annotation["end_char"] <= end)
                            matched.append({
                                **annotation,
                                "fully_contained": fully_contained,
                                "relative_start": annotation["start_char"] - start,
                                "relative_end": annotation["end_char"] - start,
                            })

                    if any(not item["fully_contained"] for item in matched):
                        partial_chunks += 1

                    record = {
                        "contract_id": contract_index,
                        "contract_title": contract["title"],
                        "paragraph_id": paragraph_index,
                        "chunk_id": chunk_index,
                        "start_char": start,
                        "end_char": end,
                        "text": text,
                        "annotations": matched,
                        "fully_contained_labels": sorted({
                            item["question"] for item in matched if item["fully_contained"]
                        }),
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1

    print("Contracts:", total_contracts)
    print("Chunks created:", total_chunks)
    print("Chunks with partially overlapping answers:", partial_chunks)
    print("Saved to:", OUTPUT_PATH)
    print("Note: Review partial overlaps before using chunks as labeled training examples.")


if __name__ == "__main__":
    main()
