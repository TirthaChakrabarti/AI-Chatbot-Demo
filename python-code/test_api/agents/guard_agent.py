import boto3
import json
from copy import deepcopy
from .utils import get_chatbot_response
import dotenv

dotenv.load_dotenv()

class GuardAgent():
    def __init__(self):
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )

        self.model_id = "meta.llama3-1-8b-instruct-v1:0"
        self.model_inference_profile = 'arn:aws:bedrock:us-east-1:823413233438:inference-profile/us.meta.llama3-1-8b-instruct-v1:0'

    def get_response(self, message):
        print('Calling Guard agent to validate query...')
        
        message = deepcopy(message)

        system_prompt = """
        You are an JSON-only agent 
        YOUR ONLY ROLE: Decide if a user's message is ALLOWED or NOT ALLOWED.

        You MUST generate a JSON of this structure exactly:
        {
        "Reason": "Brief reasoning showing which rule matched.",
        "decision": "allowed" or "not allowed",
        "message": "" if allowed, otherwise "Sorry, being a Coffee Shop AI, I am unable to proceed with that request. Kindly ensure it's about the coffee shop and its services. Thank you."
        }

        RULES:

        ALLOWED if the message is about ANY of the following:

        1. Coffee shop details: location, hours, services (like delivery, events etc.)
        2. Menu or avaliable items or options or list of items: Coffee, Chocolate, Pastries, Bakeries, Flavours, Non-alcoholic drinks
        3. ingredients and descriptions of items
        4. prices of the items
        5. Ordering items (want to order, buy, need, give, provide, send etc.)
        6. Asking for recommendations or suggestions on what to buy (if item or category not specified, assume it's about recommendations from our menu)
        7. Asking about the AI assistant's purpose, what it can do, and how it helps in the coffee shop.
        8. Greetings, thankings, farewells etc. -- even without anything specific to the coffee shop — unless they clearly mention an unrelated topic.

        NOT ALLOWED if the message is:

        1. About completely unrelated or irrelevant queries, topics, items, or services
        2. About coffeeshop's employees 
        3. About how to make an item (recipes, preparation steps)
        4. Asking recommendation for unrelated items

        IMPORTANT:
        - Do not try to answer questions. 
        - Only decide if the message is ALLOWED or NOT ALLOWED. 
        - DO NOT output normal text. Only output valid JSON.
        - No explanations outside JSON.
        - Never output extra text before or after the JSON.
        - Keys and values in JSON must be strings.
        - Follow the JSON format EXACTLY as shown.
        - Decision rules must be applied strictly.
        """

        input_messages = [{"role": "system", "content": system_prompt}] + message[-3:]

        # for message in message:
        #     input_messages.append({"role": message["role"], "content": message["content"]})

        max_turns = 3
        turn = 1

        for turn in range(1, max_turns + 1):
            system_prompt_with_retry = system_prompt
            
            if turn > 1:

                retry_prompt = f"""
            This is turn no. {turn} of maximum {max_turns} turns.:
            *Last message was not valid JSON thus rejected
            *Generate JSON only
            *Do not try to answer questions on your own
            *Only decide if the message is ALLOWED or NOT ALLOWED
            *Follow the JSON format EXACTLY as shown
            """
                
                system_prompt_with_retry += retry_prompt
            
            input_messages = [{"role": "system", "content": system_prompt_with_retry}]

            print("Input Messages (Guard):", input_messages)

            chatbot_output = get_chatbot_response(self.client, self.model_inference_profile, input_messages)
            print("Chatbot output (Guard):", chatbot_output)

            output = self.postprocess(chatbot_output)
            print("Processed Output (Guard):", output)

            if output["memory"]["guard_decision"] in ("allowed", "not allowed"):
                return output
            
            print(f"⚠ Attempt {turn}/{max_turns}: Guard agent did not produce valid JSON. Retrying...")

        # Final fallback: 
        # If the agent fails to produce valid JSON after max_turns, return a default response
        return {
                "role": "assistant",
                "content": "Sorry, your request could not be processed. Please try again.",
                "memory": {
                    "agent": "Guard",
                    "guard_decision": "not allowed"
                }
            }
    
    def postprocess(self, output):
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
                        print("Expected JSON (Guard):", json_obj)
                        break
                    except json.JSONDecodeError:
                        continue

        if json_obj is None:
            print("⚠ No valid JSON found. Returning default response.")

            return {
                "role": "assistant", 
                "content": "", 
                "memory": {
                    "agent": "Guard", 
                    "guard_decision": ""
                }
            }

        return {
            "role": "assistant",
            "content": json_obj.get("message", ""),
            "memory": {
                "agent": "Guard",
                "guard_decision": json_obj.get("decision", "")
            }
        }
