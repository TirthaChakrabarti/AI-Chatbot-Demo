import boto3
import json

client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

model_id = "meta.llama3-1-8b-instruct-v1:0"
model_inference_profile = 'arn:aws:bedrock:us-east-1:823413233438:inference-profile/us.meta.llama3-1-8b-instruct-v1:0'

system_prompt = """
You are a helpful assistant that answers questions about the capitals of countries.

Your output should be in a structured JSON format exactly like the one below. You are not allowed to write anything other than JSON object:

[
    {
        "country": the country that you will get the capital of,
        "capital": the capital of the country stated
    }
]
"""

user_prompt = "India, France, Germany, The USA, Russia, China"

# formatted_prompt = f"""
# <|begin_of_text|><|start_header_id|>user<|end_header_id|>
# {prompt}
# <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# """

formatted_prompt = f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

payload = {
    "prompt": formatted_prompt,
    "max_gen_len": 512,
    "temperature": 0.0,
    "top_p": 0.9,
}

body = json.dumps(payload)

response = client.invoke_model(
    modelId=model_inference_profile,
    body=body,
    accept='application/json',
    contentType='application/json'
)

# print(response.get('body').read().decode('utf-8'))

result = json.loads(response.get('body').read())
response_text = result.get('generation', '')

print(response_text)