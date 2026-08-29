"""Prepare small, reproducible subsets of DocumentVQA validation data."""

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import typer
from datasets import load_dataset

from .jsonl import write_jsonl
from .schemas import QuestionRecord

DATASET_NAME = "HuggingFaceM4/DocumentVQA"
VALIDATION_FILES = (
    "hf://datasets/HuggingFaceM4/DocumentVQA/data/validation-*.parquet"
)
REQUIRED_FIELDS = {
    "docId",
    "questionId",
    "question",
    "answers",
    "question_types",
    "image",
}

app = typer.Typer()


def select_indices_by_document(
    doc_ids: Sequence[Any],
    size: int,
    seed: int,
    excluded_doc_ids: Optional[Set[Any]] = None,
) -> List[int]:
    """Select exactly ``size`` questions without splitting documents across sets."""
    if size <= 0:
        raise ValueError("Split size must be greater than zero")

    excluded = excluded_doc_ids or set()
    indices_by_document: Dict[Any, List[int]] = defaultdict(list)
    for index, doc_id in enumerate(doc_ids):
        if doc_id not in excluded:
            indices_by_document[doc_id].append(index)

    rng = random.Random(seed)
    document_ids = list(indices_by_document)
    rng.shuffle(document_ids)

    selected: List[int] = []
    for doc_id in document_ids:
        document_indices = indices_by_document[doc_id]
        rng.shuffle(document_indices)
        selected.extend(document_indices[: size - len(selected)])
        if len(selected) == size:
            return selected

    raise ValueError(f"Requested {size} questions, but only {len(selected)} are available")


def to_question_record(sample: Mapping[str, Any], images_dir: Path) -> QuestionRecord:
    """Convert one DocumentVQA row to the pipeline's question format."""
    missing = REQUIRED_FIELDS.difference(sample)
    if missing:
        raise ValueError(f"DocumentVQA row is missing fields: {sorted(missing)}")

    question = str(sample["question"]).strip()
    raw_answers = sample["answers"] or []
    answers = [str(answer).strip() for answer in raw_answers if str(answer).strip()]
    if not question:
        raise ValueError(f"Question {sample['questionId']} is empty")
    if not answers:
        raise ValueError(f"Question {sample['questionId']} has no reference answers")

    page_id = f"documentvqa_{sample['docId']}"
    image_path = images_dir / f"{page_id}.png"

    return {
        "page_id": page_id,
        "question_id": str(sample["questionId"]),
        "question": question,
        "answers": answers,
        "question_types": [str(value) for value in sample["question_types"]],
        "image_path": str(image_path),
    }


def prepare_split(
    dataset: Any,
    indices: Sequence[int],
    images_dir: Path,
) -> List[QuestionRecord]:
    """Convert selected rows and save each referenced document image once."""
    records: List[QuestionRecord] = []
    saved_pages: Set[str] = set()
    images_dir.mkdir(parents=True, exist_ok=True)

    for index in indices:
        sample = dataset[index]
        record = to_question_record(sample, images_dir)
        image_path = Path(record["image_path"])

        if record["page_id"] not in saved_pages:
            image = sample["image"]
            if not hasattr(image, "save"):
                raise ValueError(f"Document {sample['docId']} does not contain a valid image")
            image.save(image_path, format="PNG")
            saved_pages.add(record["page_id"])

        records.append(record)

    return records


def prepare_dataset(
    questions_dir: Path = Path("data/processed/questions"),
    images_dir: Path = Path("data/raw/documentvqa"),
    dev_size: int = 120,
    test_size: int = 380,
    seed: int = 42,
) -> None:
    """Create document-isolated development and held-out validation subsets."""
    typer.echo(f"Loading validation data from {DATASET_NAME}...")
    validation = load_dataset(
        "parquet",
        data_files={"validation": VALIDATION_FILES},
        split="validation",
    )

    validation_doc_ids = validation["docId"]
    dev_indices = select_indices_by_document(validation_doc_ids, dev_size, seed)
    dev_doc_ids = {validation_doc_ids[index] for index in dev_indices}
    test_indices = select_indices_by_document(
        validation_doc_ids,
        test_size,
        seed + 1,
        excluded_doc_ids=dev_doc_ids,
    )

    dev_records = prepare_split(validation, dev_indices, images_dir)
    test_records = prepare_split(validation, test_indices, images_dir)

    dev_pages = {record["page_id"] for record in dev_records}
    test_pages = {record["page_id"] for record in test_records}
    if overlap := dev_pages.intersection(test_pages):
        raise ValueError(f"Development and test sets share documents: {sorted(overlap)}")

    write_jsonl(questions_dir / "dev.jsonl", dev_records)
    write_jsonl(questions_dir / "test.jsonl", test_records)

    typer.echo(f"Prepared {len(dev_records)} development questions")
    typer.echo(f"Prepared {len(test_records)} test questions")
    typer.echo(f"Saved {len(dev_pages | test_pages)} unique document images to {images_dir}")


@app.command()
def main(
    questions_dir: Path = typer.Option(
        Path("data/processed/questions"), help="Directory for prepared question files"
    ),
    images_dir: Path = typer.Option(
        Path("data/raw/documentvqa"), help="Directory for selected document images"
    ),
    dev_size: int = typer.Option(120, help="Questions sampled from validation"),
    test_size: int = typer.Option(
        380, help="Questions held out from validation for final evaluation"
    ),
    seed: int = typer.Option(42, help="Random seed for reproducible sampling"),
) -> None:
    """Create development and held-out sets from DocumentVQA validation data."""
    prepare_dataset(questions_dir, images_dir, dev_size, test_size, seed)


if __name__ == "__main__":
    app()
