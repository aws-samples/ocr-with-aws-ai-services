"""Synchronous Textract image calls use bytes and do not require S3."""

from typing import Any, Dict

import pytest
from PIL import Image

import engines.textract_engine as textract_engine
from engines.textract_engine import TextractEngine


class FakeTextract:
    """Record synchronous calls and return a minimal successful response."""

    def __init__(self) -> None:
        self.detect_calls: list[Dict[str, Any]] = []
        self.analyze_calls: list[Dict[str, Any]] = []

    @staticmethod
    def _response() -> Dict[str, Any]:
        return {
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                {
                    "BlockType": "LINE",
                    "Text": "hello",
                    "Geometry": {
                        "BoundingBox": {
                            "Left": 0.1,
                            "Top": 0.1,
                            "Width": 0.2,
                            "Height": 0.1,
                        }
                    },
                }
            ],
        }

    def detect_document_text(self, **kwargs) -> Dict[str, Any]:
        self.detect_calls.append(kwargs)
        return self._response()

    def analyze_document(self, **kwargs) -> Dict[str, Any]:
        self.analyze_calls.append(kwargs)
        return self._response()


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> FakeTextract:
    client = FakeTextract()

    def get_client(service_name: str):
        if service_name == "textract":
            return client
        raise AssertionError(f"Image processing must not request {service_name}")

    monkeypatch.setattr(textract_engine, "get_aws_client", get_client)
    return client


@pytest.fixture
def image() -> Image.Image:
    return Image.new("RGB", (40, 40), "white")


def test_text_detection_sends_image_bytes_without_s3(runtime, image) -> None:
    result = TextractEngine().process_image(
        image, {"s3_bucket": "", "feature_types": None})

    document = runtime.detect_calls[0]["Document"]
    assert isinstance(document["Bytes"], bytes)
    assert result["operation_type"] == "textract_detect"


def test_analysis_sends_image_bytes_and_features(runtime, image) -> None:
    result = TextractEngine().process_image(
        image, {"s3_bucket": "", "feature_types": ["FORMS"]})

    call = runtime.analyze_calls[0]
    assert isinstance(call["Document"]["Bytes"], bytes)
    assert call["FeatureTypes"] == ["FORMS"]
    assert result["operation_type"] == "textract_analyze"


def test_pdf_without_a_bucket_fails_before_building_an_aws_client(
    runtime, tmp_path
) -> None:
    document = tmp_path / "document.pdf"
    document.write_bytes(b"%PDF-1.4 fake")

    with pytest.raises(ValueError, match="requires an S3 bucket"):
        TextractEngine().process_image(str(document), {"s3_bucket": ""})
