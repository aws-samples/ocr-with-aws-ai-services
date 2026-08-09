from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import (
    API_COSTS,
    POSTPROCESSING_MODEL,
    TEXTRACT_FEATURE_BUNDLE_COSTS,
    TEXTRACT_FEATURE_COSTS,
)

# Operation types that go through the AnalyzeDocument family of APIs and are
# therefore priced per selected feature rather than at a flat per-page rate.
TEXTRACT_ANALYZE_OPERATIONS = ('textract_analyze', 'textract_analyze_async')

# Where each rate in API_COSTS comes from. Quoted in cost tooltips so a figure
# that looks wrong can be checked against the published price rather than
# against this file.
TEXTRACT_PRICING_URL = "https://aws.amazon.com/textract/pricing/"
BEDROCK_PRICING_URL = "https://aws.amazon.com/bedrock/pricing/"
BDA_PRICING_URL = "https://aws.amazon.com/bedrock/pricing/"

# Number of blueprint fields included in BDA's custom-output page rate. Fields
# beyond this are charged individually, per page.
BDA_INCLUDED_BLUEPRINT_FIELDS = 30


@dataclass(frozen=True)
class CostComponent:
    """
    One charge making up an engine's total, with the arithmetic that produced it.

    The comparison table shows one dollar figure per engine, which hides the fact
    that two of the three engines are billed by two different services at once -
    a per-page OCR charge plus a per-token structuring charge. Keeping the
    components lets the tooltip show how the total was reached using the run's own
    numbers, rather than a generic description of the pricing model.

    Attributes:
        label (str): What is being charged for, e.g. "Textract DetectDocumentText".
        formula (str): How `amount` was reached, e.g. "7 pages x $0.001500 per page".
        amount (float): The charge in USD.
        source (str): URL of the published price this rate came from.
    """

    label: str
    formula: str
    amount: float
    source: str


def describe_bedrock_cost(
    *, model_id: str, token_usage: Optional[Dict[str, Any]], purpose: str
) -> List[CostComponent]:
    """
    Describe a Bedrock charge as a component carrying its own token arithmetic.

    Args:
        model_id: The Bedrock model ID that was billed.
        token_usage: Token counts from the response, or None when the call did not
            report any.
        purpose: What the call was for, e.g. "Bedrock extraction" or
            "JSON structuring". Named because the same model is billed for two
            quite different steps depending on the engine.

    Returns:
        A single-element list, or an empty list when there is nothing to bill -
        no token usage, or a model with no published rate in API_COSTS.
    """
    if not token_usage or model_id not in API_COSTS.get('bedrock', {}):
        return []

    model_costs = API_COSTS['bedrock'][model_id]
    input_tokens = token_usage.get('inputTokens', 0)
    output_tokens = token_usage.get('outputTokens', 0)

    input_cost = (input_tokens / 1000) * model_costs['input']
    output_cost = (output_tokens / 1000) * model_costs['output']

    return [
        CostComponent(
            label=f"{purpose} ({model_id})",
            formula=(
                f"{input_tokens:,} input tokens x ${model_costs['input']:.6f}/1K "
                f"= ${input_cost:.6f}; "
                f"{output_tokens:,} output tokens x ${model_costs['output']:.6f}/1K "
                f"= ${output_cost:.6f}"
            ),
            amount=input_cost + output_cost,
            source=BEDROCK_PRICING_URL,
        )
    ]


def describe_textract_cost(
    *, operation_type: str, page_count: int, feature_types: Optional[List[str]] = None
) -> List[CostComponent]:
    """
    Describe the Textract charge for a run as a component with its page arithmetic.

    Args:
        operation_type: One of 'textract_detect', 'textract_async',
            'textract_analyze' or 'textract_analyze_async'.
        page_count: Pages Textract read.
        feature_types: Selected feature types; required by the analyze operations
            and ignored by the text-detection ones.

    Returns:
        A single-element list, or an empty list when the operation has no
        published rate.
    """
    # The API name is what appears on the bill, so the tooltip names it rather
    # than the app's internal operation_type string.
    api_names = {
        'textract_detect': "Textract DetectDocumentText",
        'textract_async': "Textract StartDocumentTextDetection",
        'textract_analyze': "Textract AnalyzeDocument",
        'textract_analyze_async': "Textract StartDocumentAnalysis",
    }

    if operation_type in TEXTRACT_ANALYZE_OPERATIONS:
        cost_per_page, rate_breakdown = resolve_textract_feature_rate(feature_types or [])
        formula = (
            f"{page_count} pages x ${cost_per_page:.6f} per page "
            f"({'; '.join(rate_breakdown)})"
        )
    elif operation_type in API_COSTS:
        cost_per_page = API_COSTS[operation_type]
        formula = (
            f"{page_count} pages x ${cost_per_page:.6f} per page "
            f"(text detection only, no analysis features selected)"
        )
    else:
        return []

    return [
        CostComponent(
            label=api_names.get(operation_type, f"Textract {operation_type}"),
            formula=formula,
            amount=cost_per_page * page_count,
            source=TEXTRACT_PRICING_URL,
        )
    ]


def describe_bda_cost(
    *, use_blueprint: bool, document_type: str, page_count: int, field_count: int = 0
) -> List[CostComponent]:
    """
    Describe the BDA charge for a run, including any extra-field charge.

    Args:
        use_blueprint: Whether custom output against a blueprint was used, which
            is billed at a different and higher page rate than standard output.
        document_type: 'document' or 'image'.
        page_count: Pages, or images, BDA processed.
        field_count: Fields in the blueprint. Only the count beyond
            BDA_INCLUDED_BLUEPRINT_FIELDS is charged, and only on the custom path.

    Returns:
        One component for the page charge, plus a second for extra fields when the
        blueprint defines more than the included number.
    """
    if document_type not in ('document', 'image'):
        raise ValueError(
            f"BDA prices documents and images differently, so document_type must be "
            f"one of 'document' or 'image', got {document_type!r}")

    tier = 'custom' if use_blueprint else 'standard'
    cost_per_unit = API_COSTS['bda'][tier][document_type]
    unit = "pages" if document_type == 'document' else "images"
    # The rate is charged per page, not per document: "7 pages x $0.010000 per
    # document" read as though a 7-page PDF were billed once.
    rate_unit = "page" if document_type == 'document' else "image"
    tier_name = "custom output (blueprint)" if use_blueprint else "standard output"

    components = [
        CostComponent(
            label=f"BDA {tier_name}",
            formula=f"{page_count} {unit} x ${cost_per_unit:.6f} per {rate_unit}",
            amount=cost_per_unit * page_count,
            source=BDA_PRICING_URL,
        )
    ]

    if use_blueprint and field_count > BDA_INCLUDED_BLUEPRINT_FIELDS:
        extra_fields = field_count - BDA_INCLUDED_BLUEPRINT_FIELDS
        extra_rate = API_COSTS['bda']['custom']['extra_field']
        components.append(
            CostComponent(
                label="BDA blueprint fields beyond the first "
                      f"{BDA_INCLUDED_BLUEPRINT_FIELDS}",
                formula=(
                    f"{extra_fields} extra fields x {page_count} {unit} "
                    f"x ${extra_rate:.6f} per field per page"
                ),
                amount=extra_rate * extra_fields * page_count,
                source=BDA_PRICING_URL,
            )
        )

    return components


def merge_cost_components(
    *, components: Sequence[CostComponent], document_count: int
) -> List[CostComponent]:
    """
    Combine the components of several documents into one per kind of charge.

    A batch run incurs the same charges once per document, so each document's own
    formula ("7 pages x $0.001500 per page") describes only its own share and
    would be misleading against a summed total. Merging keeps the label and the
    pricing source - which do generalise across the batch - and replaces the
    formula with the number of documents summed, so the total still explains
    itself without claiming one document's arithmetic produced it.

    Components are grouped by label and kept in first-seen order, so the batch
    row lists its charges in the same order a single-document row does.

    Args:
        components: Every component from every document, in any order.
        document_count: Documents the components were collected from, quoted in
            the merged formula.

    Returns:
        One component per distinct label, with the amounts summed.
    """
    merged: Dict[str, CostComponent] = {}

    for component in components:
        existing = merged.get(component.label)
        running_total = component.amount + (existing.amount if existing else 0.0)
        merged[component.label] = CostComponent(
            label=component.label,
            formula=f"summed across {document_count} documents in this batch",
            amount=running_total,
            source=component.source,
        )

    return list(merged.values())


def calculate_bedrock_cost(model_id, token_usage):
    """
    Calculate the cost of a Bedrock API call
    
    Args:
        model_id: The Bedrock model ID
        token_usage: Token usage dictionary
    
    Returns:
        Tuple of (HTML representation of cost, actual cost value)
    """
    if not token_usage:
        return '<div class="cost-none">No cost data available</div>', 0.0
    
    # Get cost per token for the model from the nested structure
    if model_id not in API_COSTS.get('bedrock', {}):
        return '<div class="cost-none">No cost data available for this model</div>', 0.0
    
    # Get cost per token for the model from the correct structure
    model_costs = API_COSTS['bedrock'][model_id]
    cost_per_1k_input = model_costs['input']
    cost_per_1k_output = model_costs['output']
    
    # Calculate cost
    input_tokens = token_usage.get('inputTokens', 0)
    output_tokens = token_usage.get('outputTokens', 0)
    
    input_cost = (input_tokens / 1000) * cost_per_1k_input
    output_cost = (output_tokens / 1000) * cost_per_1k_output
    total_cost = input_cost + output_cost
    
    # Format HTML output
    html = f'''
    <div class="cost-container">
        <div class="cost-total">${total_cost:.6f} total</div>
        <div class="cost-breakdown">
            <span>${input_cost:.6f} for {input_tokens} input tokens (${cost_per_1k_input:.6f}/1K tokens)</span><br>
            <span>${output_cost:.6f} for {output_tokens} output tokens (${cost_per_1k_output:.6f}/1K tokens)</span>
        </div>
    </div>
    '''
    
    # Return both the HTML and the actual cost value
    return html, total_cost


def resolve_textract_feature_rate(feature_types: List[str]) -> Tuple[float, List[str]]:
    """
    Resolve the per-page AnalyzeDocument rate for a set of Textract features

    AWS prices most feature combinations additively, but publishes discounted
    bundle rates for a few of them, and LAYOUT is free when requested alongside
    TABLES. Both cases are handled here so callers never have to know about them.

    Args:
        feature_types: Selected Textract feature types, e.g. ['FORMS', 'TABLES']

    Returns:
        Tuple of (per-page rate in USD, list of human-readable rate explanations)

    Raises:
        ValueError: If feature_types is empty or contains an unknown feature
    """
    if not feature_types:
        raise ValueError(
            "AnalyzeDocument pricing requires at least one feature type; "
            "use 'textract_detect'/'textract_async' for text-only extraction"
        )

    unknown = sorted(set(feature_types) - set(TEXTRACT_FEATURE_COSTS))
    if unknown:
        raise ValueError(f"Unknown Textract feature type(s): {unknown}")

    selected = frozenset(feature_types)

    # A published bundle rate wins over the sum of its parts.
    if selected in TEXTRACT_FEATURE_BUNDLE_COSTS:
        rate = TEXTRACT_FEATURE_BUNDLE_COSTS[selected]
        return rate, [f"${rate:.4f} per page for {' + '.join(sorted(selected))} (bundle rate)"]

    # LAYOUT is included at no charge when TABLES is also requested.
    billable = set(selected)
    breakdown = []
    if 'LAYOUT' in billable and 'TABLES' in billable:
        billable.discard('LAYOUT')
        breakdown.append("LAYOUT included at no charge with TABLES")

    rate = 0.0
    for feature in sorted(billable):
        feature_rate = TEXTRACT_FEATURE_COSTS[feature]
        rate += feature_rate
        breakdown.append(f"${feature_rate:.4f} per page for {feature}")

    return rate, breakdown


def calculate_textract_analyze_cost(feature_types: List[str], page_count: int = 1) -> Tuple[str, float]:
    """
    Calculate the cost of an AnalyzeDocument / StartDocumentAnalysis call

    Args:
        feature_types: Selected Textract feature types, e.g. ['FORMS', 'TABLES']
        page_count: Number of pages processed

    Returns:
        Tuple of (HTML string with cost information, cost value)
    """
    cost_per_page, breakdown = resolve_textract_feature_rate(feature_types)
    total_cost = cost_per_page * page_count

    breakdown_html = "<br>".join(f"<span>{line}</span>" for line in breakdown)
    html = f'''
    <div class="cost-container">
        <div class="cost-total">${total_cost:.6f} total</div>
        <div class="cost-breakdown">
            <span>${cost_per_page:.4f} per page × {page_count} pages</span><br>
            {breakdown_html}
        </div>
    </div>
    '''

    return html, total_cost


def calculate_textract_cost(operation_type='textract_detect', page_count=1, feature_types=None):
    """
    Calculate the cost of a Textract API call

    Args:
        operation_type: The Textract operation type ('textract_detect', 'textract_async',
                        'textract_analyze' or 'textract_analyze_async')
        page_count: Number of pages processed
        feature_types: Selected Textract feature types; required for the analyze
                       operation types, ignored for the text-detection ones

    Returns:
        Tuple: (HTML string with cost information, cost value)
    """
    if operation_type in TEXTRACT_ANALYZE_OPERATIONS:
        return calculate_textract_analyze_cost(feature_types or [], page_count)

    if operation_type not in API_COSTS:
        return '<div class="cost-none">No cost data available for this operation</div>', 0.0

    cost_per_page = API_COSTS[operation_type]
    total_cost = cost_per_page * page_count

    # Format HTML output
    html = f'''
    <div class="cost-container">
        <div class="cost-total">${total_cost:.6f} total</div>
        <div class="cost-breakdown">
            <span>${cost_per_page:.6f} per page \u00d7 {page_count} pages</span>
        </div>
    </div>
    '''
    
    return html, total_cost

def calculate_full_textract_cost(result):
    """
    Calculate the total cost including LLM processing
    
    Args:
        result: Result dictionary from Textract processing
        
    Returns:
        Total cost value
    """
    pages = result.get("pages", 1)
    operation_type = result.get("operation_type", "textract_detect")
    feature_types = result.get("feature_types") or []

    # Get base textract cost
    _, textract_base_cost = calculate_textract_cost(operation_type, pages, feature_types)
    
    # Add LLM cost if applicable
    total_cost = textract_base_cost
    token_usage = result.get("token_usage")
    if token_usage:
        _, llm_cost = calculate_bedrock_cost(POSTPROCESSING_MODEL, token_usage)
        total_cost += llm_cost
    
    return total_cost

def calculate_bda_cost(use_blueprint, document_type, page_count=1, field_count=0):
    """
    Calculate the cost of a BDA API call
    
    Args:
        use_blueprint: Whether Custom Output (Blueprint) was used
        document_type: 'document' or 'image'
        page_count: Number of pages/images processed
        field_count: Number of fields defined in blueprint (only relevant for custom output)
        
    Returns:
        HTML string with cost information, and the actual cost value
    """
    if document_type not in ['document', 'image']:
        document_type = 'document'  # Default to document type
        
    if use_blueprint:
        # Custom Output with blueprint
        cost_per_unit = API_COSTS['bda']['custom'][document_type]
        
        # Calculate additional cost for extra fields beyond 30
        extra_field_cost = 0
        if field_count > 30:
            extra_field_cost = API_COSTS['bda']['custom']['extra_field'] * (field_count - 30) * page_count
            
        total_cost = (cost_per_unit * page_count) + extra_field_cost
        
        # Format HTML output
        html = f'''
        <div class="cost-container">
            <div class="cost-total">${total_cost:.6f} total</div>
            <div class="cost-breakdown">
                <span>${cost_per_unit:.6f} per {document_type} \u00d7 {page_count} {document_type}s</span>
                {f"<br><span>${extra_field_cost:.6f} for additional {field_count-30} fields</span>" if field_count > 30 else ""}
            </div>
        </div>
        '''
    else:
        # Standard Output
        cost_per_unit = API_COSTS['bda']['standard'][document_type]
        total_cost = cost_per_unit * page_count
        
        # Format HTML output
        html = f'''
        <div class="cost-container">
            <div class="cost-total">${total_cost:.6f} total</div>
            <div class="cost-breakdown">
                <span>${cost_per_unit:.6f} per {document_type} \u00d7 {page_count} {document_type}s</span>
            </div>
        </div>
        '''
    
    return html, total_cost