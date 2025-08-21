def get_chatbot_response(client, model_name, messages, temperature=0.0):
    input_messages = []
    for message in messages: 
        input_messages.append({"role": message["role"], "content": message["content"]})

    print("Attempting to get LM response...")

    response = client.chat.completions.create(
        model=model_name,
        messages=input_messages,
        temperature=temperature,  # no randomness desired as agent-based system will depend on each other
        top_p=0.4,
        max_tokens=2000  # word or sub-word
        # max_tokens=512
    ).choices[0].message.content

    return response


def get_embedding_vector(prompt, embedding_model_name):
    embedding_vector = model.encode(prompt)

    embedding = []

    for embedding_obj in embedding_vector:
        embedding.append(embedding_obj)

    return embedding

def double_check_json_output(client,model_name,json_string):
    prompt = f""" 
    Return ONLY valid JSON that Python's json.loads() will accept.
    - All keys and string values must be double quoted.
    - No comments, no text outside the JSON.
    - No markdown fences.

    Here is the json string to fix:
    {json_string}
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_chatbot_response(client,model_name,messages)

    return response