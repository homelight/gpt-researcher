import tiktoken

# Per OpenAI Pricing Page: https://openai.com/api/pricing/
ENCODING_MODEL = "o200k_base"
INPUT_COST_PER_TOKEN = 0.00000015
OUTPUT_COST_PER_TOKEN = 0.0000006
IMAGE_INFERENCE_COST = 0.003825
EMBEDDING_COST = 0.02 / 1000000 # Assumes new ada-3-small


# Cost estimation is via OpenAI libraries and models. May vary for other models
def estimate_llm_cost(input_content: str, output_content: str) -> float:
    encoding = tiktoken.get_encoding(ENCODING_MODEL)
    input_tokens = encoding.encode(input_content)
    output_tokens = encoding.encode(output_content)
    input_costs = len(input_tokens) * INPUT_COST_PER_TOKEN
    output_costs = len(output_tokens) * OUTPUT_COST_PER_TOKEN
    # print('\n\n--------- llm cost ---------')
    # print(f"Input Costs: {input_costs}")
    # print(f"Output Costs: {output_costs}")
    # print(f"{len(input_tokens)} input_content: {input_content}")
    # print(f"{len(output_tokens)} output_content: {output_content}")
    return input_costs + output_costs

def estimate_embedding_cost(model, docs):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = sum(len(encoding.encode(str(doc))) for doc in docs)

    # print('\n\n--------- embedding cost ---------')
    # print(f"Total Tokens: {total_tokens}")
    # print(f"cost: {total_tokens * EMBEDDING_COST}")

    return total_tokens * EMBEDDING_COST

