import boto3
import json
import re
from copy import deepcopy   # deepcopy copies by value and not by reference
import os
from .utils import get_chatbot_response
import dotenv

dotenv.load_dotenv()

class ClassificationAgent():
    def __init__(self):
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )

        self.model_id = "meta.llama3-1-8b-instruct-v1:0"
        self.model_inference_profile = 'arn:aws:bedrock:us-east-1:823413233438:inference-profile/us.meta.llama3-1-8b-instruct-v1:0'

    def get_response(self, messages):
        messages = deepcopy(messages)

        system_prompt = """
        You are a strict JSON-only classification agent for a coffee shop chatbot. 
        Your ONLY job is to decide which agent should handle the user's message and return a JSON.

        CRITICALLY IMPORTANT: 
        - DO NOT generate any other text.
        - DO NOT write any text, explanations, apologies, order confirmations, or recommendations.
        - If you output anything outside JSON, the system will FAIL.

        **We have 3 agents to choose from:

        1. details_agent: for answering questions about the coffee shop, its location, delivery places, working hours, menu items (that we have or serve) and their details, prices. And also to answer greetings and goodbyes.
        2. order_taking_agent: for taking ORDERS from the user. It's responsible to have a conversation with the user about the order untill it's complete.
        3. recommendation_agent: for giving recommendations and suggestions to the user about what to buy. If the user asks for a recommendation or suggestion, this agent should be used.
        
        OUTPUT RULES:
        - Always output ONLY valid JSON (no extra text).
        - JSON format (exactly):
        {
        "Reason": "short reasoning",
        "decision": "details_agent" or "order_taking_agent" or "recommendation_agent",
        "message": ""
        }

        - Keys and values must be strings.

        If the user's message is unclear, use DECISION HELPER:

        1. If the user is ASKING a question (ends with "?" or starts with "do you", "what", "where", "when", "how", "is", "are") or wanting to know (statements like "tell me", "tell me about", "tell me more about", "give me details", "give me more details", "explain", "explain more", "what's the difference between"), chose "details_agent".
        2. If the user is REQUESTING or ORDERING something (statements like "order", "I want", "I'll have", "get me", "give me", "send me", "please bring", "buy", "need" or just the name of product), chose "order_taking_agent".
        3. If the user is ASKING for a suggestion or recommendation ("what do you recommend", "any specials", "suggest me something"), chose "recommendation_agent".
        4. Greetings or goodbyes ("hi", "hello", "bye", "thanks") chose "details_agent".
        5. If unsure: default to "details_agent".

        VITAL NOTE:
        NEVER generate any other text other than the JSON.
        """
        
        input_messages = [
            {"role": "system", "content": system_prompt},
        ]

        # input_messages += messages[-3:]
        # Only pass the last user message, not assistant messages
        if messages and messages[-1]['role'] == 'user':
            input_messages.append(messages[-1])

        input_messages.insert(0, {"role": "system", "content": 'CRITICAL: Your response will be parsed by json.loads(). If it is not valid JSON, the program will crash.'})

        print('input_messages(classification agent):', input_messages)

        chatbot_output =get_chatbot_response(self.client,self.model_inference_profile,input_messages)
        print('chatbot_output(classification agent):', chatbot_output)

        output = self.postprocess(chatbot_output)
        print('processed output(classification agent):', output)

        return output

    def postprocess(self,output):
        # print("classification output: ", output)
        # output = json.loads(output)

        # dict_output = {
        #     "role": "assistant",
        #     "content": output['message'],
        #     "memory": {"agent":"classification_agent",
        #                "classification_decision": output['decision']
        #               }
        # }
        # return dict_output
        if not output or not output.strip():
            raise ValueError("Chatbot output is empty")

        # Try to find the first valid JSON object in the output
        json_obj = None
        brace_count = 0
        start_idx = None
        for i, ch in enumerate(output):
            if ch == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    try:
                        json_obj = json.loads(output[start_idx:i+1])
                        break
                    except json.JSONDecodeError:
                        continue

        if json_obj is None:
            print("⚠ No valid JSON found. Using fallback.")
            return {
                "role": "assistant",
                "content": "Sorry I am not sure about it. Please try again.",
                "memory": {
                    "agent": "Classification_Agent",
                    "classification_decision": "unsure"
                }
            }

        return {
            "role": "assistant",
            "content": json_obj.get("message", ""),
            "memory": {
                "agent": "Classification_Agent",
                "classification_decision": json_obj.get("decision", "")
            }
        }