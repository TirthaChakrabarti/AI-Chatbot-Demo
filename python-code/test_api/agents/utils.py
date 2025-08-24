import json

# Getting response from bedrock Llama
def get_chatbot_response(client, model_id, messages, temperature=0.0):
    formatted_prompt = convert_message_to_llama3_prompt(messages)

    input_messages = []
    for message in messages: 
        input_messages.append({"role": message["role"], "content": message["content"]})

    print("Attempting to get LM response...")

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "max_gen_len": 2000,
            "temperature": temperature,
            "top_p": 0.8,
            "prompt": formatted_prompt
        })
    )

    response_body = json.loads(response['body'].read())
    return response_body.get("generation", "").strip()

# Converting OpenAI-like prompt to Bedrock llama3 prompt
def convert_message_to_llama3_prompt(messages):
    prompt = "<|begin_of_text|>\n"
    for message in messages:
        role = message["role"]
        content = message["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n"
        prompt += content + "\n"
        prompt += f"<|eot_id|>\n"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n"

    return prompt

def double_check_json_output(client,model_id,json_string):
    prompt = f""" 
    Return ONLY valid JSON that Python's json.loads() will accept.
    - All keys and string values must be double quoted.
    - No comments, no text outside the JSON.
    - No markdown fences.

    Here is the json string to fix:
    {json_string}
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_chatbot_response(client,model_id,messages)

    return response