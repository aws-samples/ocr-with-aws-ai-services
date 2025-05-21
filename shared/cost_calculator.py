from .config import API_COSTS, POSTPROCESSING_MODEL

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


def calculate_textract_cost(operation_type='textract_detect', page_count=1):
    """
    Calculate the cost of a Textract API call
    
    Args:
        operation_type: The Textract operation type ('textract_detect' or 'textract_analyze_tables_forms')
        page_count: Number of pages processed
        
    Returns:
        Tuple: (HTML string with cost information, cost value)
    """
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
    
    # Get base textract cost
    _, textract_base_cost = calculate_textract_cost(operation_type, pages)
    
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