"""Offline tests for DocumentVQA dataset preparation."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class _DummyTyperApp:
    def command(self, *args, **kwargs):
        return lambda function: function


if importlib.util.find_spec("typer") is None:
    typer_stub = types.ModuleType("typer")
    typer_stub.Typer = lambda: _DummyTyperApp()
    typer_stub.Option = lambda default, *args, **kwargs: default
    typer_stub.echo = lambda *args, **kwargs: None
    sys.modules["typer"] = typer_stub

if importlib.util.find_spec("datasets") is None:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = lambda *args, **kwargs: None
    sys.modules["datasets"] = datasets_stub

from ocr_vlm.dataset import (
    VALIDATION_FILES,
    prepare_dataset,
    prepare_split,
    select_indices_by_document,
    to_question_record,
)
from ocr_vlm.jsonl import read_jsonl


QUESTION_FIELDS = {
    "page_id",
    "question_id",
    "question",
    "answers",
    "question_types",
    "image_path",
}


class FakeImage:
    def __init__(self):
        self.save_count = 0

    def save(self, path, format=None):
        self.save_count += 1
        Path(path).write_bytes(b"fake image")


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]


def make_row(doc_id, question_id, answers=None, image=None):
    return {
        "docId": doc_id,
        "questionId": question_id,
        "question": f" Question {question_id} ",
        "answers": answers if answers is not None else [f" Answer {question_id} "],
        "question_types": ["layout"],
        "image": image or FakeImage(),
    }


class SelectionTests(unittest.TestCase):
    def test_selection_is_exact_reproducible_and_respects_exclusions(self):
        doc_ids = [1, 1, 2, 3, 3, 4, 5]

        first = select_indices_by_document(doc_ids, size=4, seed=42, excluded_doc_ids={3})
        second = select_indices_by_document(doc_ids, size=4, seed=42, excluded_doc_ids={3})

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(doc_ids[index] != 3 for index in first))

    def test_selection_rejects_invalid_or_unavailable_sizes(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            select_indices_by_document([1, 2], size=0, seed=42)

        with self.assertRaisesRegex(ValueError, "only 2 are available"):
            select_indices_by_document([1, 2], size=3, seed=42)


class RecordTests(unittest.TestCase):
    def test_row_is_converted_to_question_record(self):
        record = to_question_record(
            make_row(14465, 49153, answers=[" 0.28 ", "0.280"]),
            Path("images"),
        )

        self.assertEqual(set(record), QUESTION_FIELDS)
        self.assertEqual(record["page_id"], "documentvqa_14465")
        self.assertEqual(record["question_id"], "49153")
        self.assertEqual(record["question"], "Question 49153")
        self.assertEqual(record["answers"], ["0.28", "0.280"])
        self.assertEqual(record["image_path"], "images/documentvqa_14465.png")

    def test_conversion_rejects_missing_fields_and_empty_answers(self):
        missing_image = make_row(1, 10)
        del missing_image["image"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            to_question_record(missing_image, Path("images"))

        with self.assertRaisesRegex(ValueError, "no reference answers"):
            to_question_record(make_row(1, 10, answers=[]), Path("images"))

        missing_answers = make_row(1, 10)
        missing_answers["answers"] = None
        with self.assertRaisesRegex(ValueError, "no reference answers"):
            to_question_record(missing_answers, Path("images"))

    def test_prepare_split_saves_a_shared_document_once(self):
        first_image = FakeImage()
        second_image = FakeImage()
        dataset = FakeDataset(
            [
                make_row(7, 70, image=first_image),
                make_row(7, 71, image=second_image),
            ]
        )

        with TemporaryDirectory() as directory:
            records = prepare_split(dataset, [0, 1], Path(directory))

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["image_path"], records[1]["image_path"])
            self.assertEqual(first_image.save_count + second_image.save_count, 1)
            self.assertTrue(Path(records[0]["image_path"]).is_file())


class PreparationTests(unittest.TestCase):
    def test_prepare_dataset_uses_validation_only(self):
        validation = FakeDataset(
            [
                make_row(1, 10),
                make_row(1, 11),
                make_row(2, 12),
                make_row(3, 13),
                make_row(4, 14),
                make_row(5, 15),
                make_row(6, 16),
                make_row(7, 17),
            ]
        )

        def fake_load_dataset(name, data_files, split):
            self.assertEqual(name, "parquet")
            self.assertEqual(data_files, {"validation": VALIDATION_FILES})
            self.assertEqual(split, "validation")
            return validation

        with TemporaryDirectory() as directory, patch(
            "ocr_vlm.dataset.load_dataset", side_effect=fake_load_dataset
        ) as loader:
            root = Path(directory)
            questions_dir = root / "questions"
            images_dir = root / "images"

            prepare_dataset(
                questions_dir=questions_dir,
                images_dir=images_dir,
                dev_size=2,
                test_size=3,
                seed=42,
            )

            dev = read_jsonl(questions_dir / "dev.jsonl")
            final = read_jsonl(questions_dir / "test.jsonl")

            self.assertEqual(loader.call_count, 1)
            self.assertEqual(len(dev), 2)
            self.assertEqual(len(final), 3)
            self.assertTrue(
                {record["page_id"] for record in dev}.isdisjoint(
                    {record["page_id"] for record in final}
                )
            )
            self.assertTrue(all(set(record) == QUESTION_FIELDS for record in dev + final))
            self.assertTrue(
                all(Path(record["image_path"]).is_file() for record in dev + final)
            )


if __name__ == "__main__":
    unittest.main()
