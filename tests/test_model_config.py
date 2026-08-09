"""
Tests that the model list and the price list agree with each other

`calculate_bedrock_cost()` and `BedrockEngine.get_cost()` both report $0.00 for a
model that has no entry in `API_COSTS['bedrock']`, rather than raising. On a
cost-comparison benchmark that is the worst possible failure mode: the model appears
free, which is a plausible-looking number, so nothing prompts anyone to check. Adding
a model to `BEDROCK_MODELS` without adding its price is a one-line mistake with no
visible symptom, and these tests are the only thing that catches it.
"""

import pytest

from shared.config import (
    API_COSTS,
    BEDROCK_MODELS,
    MANTLE_ENDPOINT_TEMPLATE,
    MANTLE_MODEL_IDS,
    POSTPROCESSING_MODEL,
)
from shared.cost_calculator import calculate_bedrock_cost

# Enough tokens that any real per-token rate produces a cost well clear of the
# rounding used in the reported figures.
SAMPLE_TOKEN_USAGE = {"inputTokens": 100_000, "outputTokens": 20_000, "totalTokens": 120_000}


# --- the hazard these tests exist for ----------------------------------------


def test_an_unpriced_model_is_reported_as_free_not_as_an_error() -> None:
    """
    An unknown model costs $0.00 silently

    This is the behaviour that makes the coverage tests below necessary rather than
    redundant, so it is asserted directly: if this ever started raising, the tests
    that follow would be belt-and-braces instead of load-bearing.
    """
    _, cost = calculate_bedrock_cost("openai.gpt-not-a-real-model", SAMPLE_TOKEN_USAGE)

    assert cost == 0.0


# --- every selectable model has a price --------------------------------------


@pytest.mark.parametrize(
    "display_name,model_id", sorted(BEDROCK_MODELS.items())
)
def test_every_selectable_model_has_a_price(display_name: str, model_id: str) -> None:
    """
    Each model offered in the UI can be priced

    Args:
        display_name (str): Label shown in the model dropdown.
        model_id (str): Bedrock model ID the label maps to.
    """
    assert model_id in API_COSTS["bedrock"], (
        f"'{display_name}' is selectable but has no entry in API_COSTS['bedrock'], so "
        f"every run with it would report a cost of $0.00."
    )


@pytest.mark.parametrize(
    "display_name,model_id", sorted(BEDROCK_MODELS.items())
)
def test_every_selectable_model_produces_a_non_zero_cost(
    display_name: str, model_id: str
) -> None:
    """
    Each model's rates actually yield a charge

    A model present in the price list with zeroed rates would pass the membership
    test above and still report as free.

    Args:
        display_name (str): Label shown in the model dropdown.
        model_id (str): Bedrock model ID the label maps to.
    """
    _, cost = calculate_bedrock_cost(model_id, SAMPLE_TOKEN_USAGE)

    assert cost > 0.0, f"'{display_name}' prices out at $0.00 for a 120,000-token run."


def test_the_postprocessing_model_has_a_price() -> None:
    """
    The structuring model can be priced

    Textract and BDA both charge a second Bedrock call to this model, and that cost is
    reported separately from the engine's own, so it needs its own entry - which it
    would not necessarily have if it were ever set to a model absent from
    BEDROCK_MODELS.
    """
    assert POSTPROCESSING_MODEL in API_COSTS["bedrock"]


def test_no_prices_are_left_for_models_that_were_removed() -> None:
    """
    The price list carries no entries for models no longer offered

    Stale rates are not merely dead weight: read later they look like current AWS
    pricing for a model this app supports, and the retired Claude 3.x entries had
    already drifted from the published rates.
    """
    priced_models = set(API_COSTS["bedrock"])
    live_models = set(BEDROCK_MODELS.values()) | {POSTPROCESSING_MODEL}

    assert priced_models == live_models, (
        f"Priced but not selectable: {sorted(priced_models - live_models)}. "
        f"Selectable but not priced: {sorted(live_models - priced_models)}."
    )


def test_rates_are_per_thousand_tokens_not_per_token() -> None:
    """
    A million input tokens costs what the pricing page says, to the cent

    Both cost functions compute (tokens / 1000) * rate, so the table has to hold
    dollars per 1,000 tokens. Every entry used to carry an extra "/ 1000", making the
    stored value a per-token rate and every reported Bedrock cost 1/1000 of the truth.
    A real Luna run billed at $0.0048 was displayed as $0.000005.

    Anchored on Claude Sonnet 5 at its published $2 per 1M input tokens, because a
    relative assertion would hold equally well at the wrong scale.
    """
    _, cost = calculate_bedrock_cost(
        "us.anthropic.claude-sonnet-5",
        {"inputTokens": 1_000_000, "outputTokens": 0, "totalTokens": 1_000_000},
    )

    assert cost == pytest.approx(2.00)


def test_a_typical_run_costs_a_realistic_amount() -> None:
    """
    A measured run prices out in the range a human would sanity-check against

    Taken from an actual GPT-5.6 Luna run over a 9-page claim form: 6,221 input and
    2,604 output tokens. Fractions of a cent is right; millionths of a cent is the
    scale the old rates produced, and dollars would mean the rates were entered as
    per-1M figures.
    """
    _, cost = calculate_bedrock_cost(
        "openai.gpt-5.6-luna",
        {"inputTokens": 6_221, "outputTokens": 2_604, "totalTokens": 8_825},
    )

    assert 0.0001 < cost < 0.10


@pytest.mark.parametrize("model_id", sorted(API_COSTS["bedrock"]))
def test_output_tokens_are_never_cheaper_than_input_tokens(model_id: str) -> None:
    """
    Each model's output rate is at least its input rate

    True of every model AWS publishes. A pair of rates accidentally transposed while
    transcribing the pricing page would otherwise go unnoticed, since both orderings
    produce a believable total.

    Args:
        model_id (str): Bedrock model ID to check the rates of.
    """
    rates = API_COSTS["bedrock"][model_id]

    assert rates["output"] >= rates["input"]


# --- the mantle split --------------------------------------------------------


def test_mantle_models_are_all_selectable() -> None:
    """
    Every model marked as a Mantle model is one the UI actually offers

    MANTLE_MODEL_IDS is what routes a request away from boto3, so an ID here that
    matches nothing in BEDROCK_MODELS is dead configuration, and one missing from here
    would be sent to `converse()`, which rejects it.
    """
    assert MANTLE_MODEL_IDS <= set(BEDROCK_MODELS.values())


@pytest.mark.parametrize("model_id", sorted(MANTLE_MODEL_IDS))
def test_mantle_models_carry_no_cross_region_prefix(model_id: str) -> None:
    """
    Mantle model IDs are bare, with no "us." inference-profile prefix

    Unlike every bedrock-runtime model in this app, these have no geo or global
    inference profile, so a "us." prefix copied from the neighbouring entries would be
    rejected as an unrecognised model.

    Args:
        model_id (str): Mantle model ID to check.
    """
    assert not model_id.startswith("us.")


def test_the_mantle_endpoint_is_region_specific() -> None:
    """
    The endpoint template takes the region as a parameter

    The region is part of the hostname rather than a header, so a hardcoded host would
    silently send every request to one region regardless of configuration.
    """
    assert "{region}" in MANTLE_ENDPOINT_TEMPLATE

    formatted = MANTLE_ENDPOINT_TEMPLATE.format(region="eu-west-1")
    assert formatted.startswith("https://")
    assert "eu-west-1" in formatted
