"""Tests for cost components: the arithmetic behind each engine's dollar figure.

The comparison table shows one number per engine, which hides that two of the
three engines are billed by two services at once - a per-page OCR charge plus a
per-token structuring charge. `CostComponent` carries each charge with the
arithmetic that produced it so the tooltip can show the run's own numbers.

The invariant these tests defend is that the components sum to the total the table
shows. If the two ever disagree, the tooltip is lying about the figure it explains.
"""

from typing import Any, Dict, List

import pytest

from processor import process_engine_result
from shared.config import API_COSTS, POSTPROCESSING_MODEL
from shared.cost_calculator import (
    CostComponent,
    describe_bedrock_cost,
    describe_textract_cost,
)

# A plausible token count for structuring a seven-page claim form.
TOKEN_USAGE: Dict[str, int] = {
    "inputTokens": 21432,
    "outputTokens": 3166,
    "totalTokens": 24598,
}


def total_of(components: List[CostComponent]) -> float:
    """Sum a component list the way the table does.

    Args:
        components: The charges making up one engine's cost.

    Returns:
        float: The total in USD.
    """
    return sum(component.amount for component in components)


class TestBedrockComponents:
    """Per-token charges, quoted at the published per-1K rate."""

    def test_the_amount_is_input_plus_output_at_their_own_rates(self) -> None:
        """Input and output are priced differently, so both are read separately."""
        rates = API_COSTS['bedrock'][POSTPROCESSING_MODEL]
        expected = (21432 / 1000) * rates['input'] + (3166 / 1000) * rates['output']

        components = describe_bedrock_cost(
            model_id=POSTPROCESSING_MODEL,
            token_usage=TOKEN_USAGE,
            purpose="JSON structuring")

        assert total_of(components) == pytest.approx(expected)

    def test_the_formula_quotes_the_runs_own_token_counts(self) -> None:
        """A generic description of per-token pricing would not be checkable."""
        components = describe_bedrock_cost(
            model_id=POSTPROCESSING_MODEL,
            token_usage=TOKEN_USAGE,
            purpose="JSON structuring")

        assert "21,432 input tokens" in components[0].formula
        assert "3,166 output tokens" in components[0].formula

    def test_the_purpose_names_the_step_being_billed(self) -> None:
        """The same model bills for extraction on one engine and structuring on two."""
        structuring = describe_bedrock_cost(
            model_id=POSTPROCESSING_MODEL, token_usage=TOKEN_USAGE,
            purpose="JSON structuring")
        extraction = describe_bedrock_cost(
            model_id=POSTPROCESSING_MODEL, token_usage=TOKEN_USAGE,
            purpose="Bedrock extraction")

        assert structuring[0].label.startswith("JSON structuring")
        assert extraction[0].label.startswith("Bedrock extraction")
        assert POSTPROCESSING_MODEL in structuring[0].label

    def test_no_token_usage_means_no_component(self) -> None:
        """An engine that reported no tokens is not billed a guessed amount."""
        assert describe_bedrock_cost(
            model_id=POSTPROCESSING_MODEL, token_usage=None, purpose="x") == []

    def test_an_unpriced_model_means_no_component(self) -> None:
        """A model with no published rate yields no charge rather than a zero one."""
        assert describe_bedrock_cost(
            model_id="some.model-with-no-price", token_usage=TOKEN_USAGE,
            purpose="x") == []


class TestTextractComponents:
    """Per-page charges, whose rate depends on which API was called."""

    def test_text_detection_is_the_flat_per_page_rate(self) -> None:
        """Seven pages of DetectDocumentText at $0.0015 is $0.0105."""
        components = describe_textract_cost(
            operation_type='textract_async', page_count=7, feature_types=[])

        assert total_of(components) == pytest.approx(0.0105)
        assert "7 pages" in components[0].formula

    def test_the_label_names_the_api_that_appears_on_the_bill(self) -> None:
        """'textract_async' is this app's word for it; the bill says otherwise."""
        components = describe_textract_cost(
            operation_type='textract_async', page_count=1, feature_types=[])
        assert components[0].label == "Textract StartDocumentTextDetection"

    def test_the_formula_says_no_features_were_selected(self) -> None:
        """The cheapest rate is worth explaining, since features cost far more."""
        components = describe_textract_cost(
            operation_type='textract_detect', page_count=1, feature_types=[])
        assert "text detection only" in components[0].formula

    def test_analysis_features_cost_more_per_page_than_text_detection(self) -> None:
        """Billing an AnalyzeDocument run at the detection rate understates it."""
        detection = describe_textract_cost(
            operation_type='textract_async', page_count=7, feature_types=[])
        analysis = describe_textract_cost(
            operation_type='textract_analyze_async', page_count=7,
            feature_types=['FORMS', 'TABLES'])

        assert total_of(analysis) > total_of(detection)

    def test_the_analysis_formula_itemises_the_selected_features(self) -> None:
        """Which feature contributed what is the whole question with a bundle rate."""
        components = describe_textract_cost(
            operation_type='textract_analyze', page_count=3,
            feature_types=['FORMS', 'TABLES'])

        assert "3 pages" in components[0].formula
        assert "FORMS" in components[0].formula and "TABLES" in components[0].formula

    def test_an_unknown_operation_has_no_component(self) -> None:
        """An unpriced operation is reported as unpriced, not as free."""
        assert describe_textract_cost(
            operation_type='textract_teleport', page_count=1) == []

    def test_every_component_cites_a_pricing_page(self) -> None:
        """A figure that cannot be checked against a published price is not useful."""
        components = describe_textract_cost(
            operation_type='textract_async', page_count=1, feature_types=[])
        assert components[0].source.startswith("https://aws.amazon.com/")


class TestComponentsSumToTheReportedTotal:
    """The invariant: the tooltip explains the number the table shows."""

    def _result(self, **overrides: Any) -> Dict[str, Any]:
        """Build a successful engine result dict.

        Args:
            **overrides: Fields to set or replace on the base result.

        Returns:
            Dict[str, Any]: A result dict shaped like an engine's return value.
        """
        result: Dict[str, Any] = {
            "text": "some extracted text",
            "json": {"field": "value"},
            "process_time": 50.533,
            "pages": 7,
            "operation_type": "textract_async",
        }
        result.update(overrides)
        return result

    def test_textract_total_is_its_page_charge_plus_its_token_charge(self) -> None:
        """Two services bill one Textract run, and both belong in the total."""
        processed = process_engine_result(
            "Textract",
            self._result(feature_types=[], token_usage=TOKEN_USAGE),
            truth_data=None,
            truth_exists=False)

        assert len(processed["cost_breakdown"]) == 2
        assert processed["cost"] == pytest.approx(total_of(processed["cost_breakdown"]))
        assert processed["pages"] == 7

    def test_textract_without_structuring_is_the_page_charge_alone(self) -> None:
        """With structured output off there are no tokens and no second charge."""
        processed = process_engine_result(
            "Textract",
            self._result(feature_types=[], token_usage=None),
            truth_data=None,
            truth_exists=False)

        assert len(processed["cost_breakdown"]) == 1
        assert processed["cost"] == pytest.approx(0.0105)

    def test_bedrock_total_is_one_per_token_charge(self) -> None:
        """One Converse call does OCR and structuring, so there is no page charge."""
        processed = process_engine_result(
            "Bedrock",
            self._result(
                operation_type="bedrock",
                model_id=POSTPROCESSING_MODEL,
                token_usage=TOKEN_USAGE),
            truth_data=None,
            truth_exists=False)

        assert len(processed["cost_breakdown"]) == 1
        assert processed["cost"] == pytest.approx(total_of(processed["cost_breakdown"]))

    def test_bda_standard_output_bills_every_page_plus_structuring(self) -> None:
        """The fixed defect: seven pages billed as seven, not as one."""
        processed = process_engine_result(
            "BDA",
            self._result(
                operation_type="bda",
                use_blueprint=False,
                field_count=0,
                token_usage=TOKEN_USAGE),
            truth_data=None,
            truth_exists=False)

        page_charge = next(
            component for component in processed["cost_breakdown"]
            if component.label.startswith("BDA"))
        assert page_charge.amount == pytest.approx(0.07)
        assert "7 pages" in page_charge.formula
        assert processed["cost"] == pytest.approx(total_of(processed["cost_breakdown"]))

    def test_bda_with_a_blueprint_is_billed_no_structuring_charge(self) -> None:
        """The blueprint returns schema-shaped JSON, so no LLM runs afterwards."""
        processed = process_engine_result(
            "BDA",
            self._result(
                operation_type="bda",
                use_blueprint=True,
                field_count=10,
                token_usage=TOKEN_USAGE),
            truth_data=None,
            truth_exists=False)

        assert all(
            not component.label.startswith("JSON structuring")
            for component in processed["cost_breakdown"])

    def test_bda_reporting_no_pages_raises_rather_than_billing_nothing(self) -> None:
        """A successful BDA run with no page count cannot be costed."""
        with pytest.raises(ValueError, match="cannot be calculated"):
            process_engine_result(
                "BDA",
                self._result(operation_type="bda", pages=0, use_blueprint=False),
                truth_data=None,
                truth_exists=False)

    def test_a_failed_run_has_no_charge_and_no_components(self) -> None:
        """A run that returned nothing must not report a cost it cannot substantiate."""
        processed = process_engine_result(
            "BDA",
            {"text": "BDA Error: boom", "operation_type": "error", "process_time": 1.2,
             "pages": 0},
            truth_data=None,
            truth_exists=False)

        assert processed["cost"] == 0.0
        assert processed["cost_breakdown"] == []
        assert processed["pages"] == 0

    def test_the_banner_itemises_a_two_service_charge(self) -> None:
        """The status line already split this; it now comes from the components."""
        processed = process_engine_result(
            "Textract",
            self._result(feature_types=[], token_usage=TOKEN_USAGE),
            truth_data=None,
            truth_exists=False)

        assert "Textract StartDocumentTextDetection: $" in processed["status_html"]
        assert "JSON structuring: $" in processed["status_html"]
