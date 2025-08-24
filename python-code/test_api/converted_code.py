import json
import boto3

client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

model_id = "meta.llama3-1-8b-instruct-v1:0"
model_inference_profile = 'arn:aws:bedrock:us-east-1:823413233438:inference-profile/us.meta.llama3-1-8b-instruct-v1:0'

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

def get_bedrock_response(client, model_id, messages, temperature=0.0):
    formatted_prompt = convert_message_to_llama3_prompt(messages)

    print("Converting OpenAI prompt to Bedrock prompt:", formatted_prompt)

    response = client.invoke_model(
        modelId=model_inference_profile,
        body=json.dumps({
            "max_gen_len": 2000,
            "temperature": temperature,
            "top_p": 0.8,
            "prompt": formatted_prompt
        })
    )

    response_body = json.loads(response['body'].read())
    return response_body.get("generation", "").strip()

system_prompt = """
Your output should be in a structured JSON format exactly like the one below. 
You are not allowed to write anything other than JSON object:
[
    {
        "Discovery": name of discovery,
        "Introduction": brief introduction
    }
]

DO NOT WRITE ANYTHING OTHER THAN JSON OBJECT
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What are Einstein's contributions? Name top 5."}
]

response = get_bedrock_response(client, model_id, messages)

print(response)