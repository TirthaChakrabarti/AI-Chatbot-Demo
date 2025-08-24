import boto3
import os
import json 
from .utils import get_chatbot_response, double_check_json_output
from copy import deepcopy
from dotenv import load_dotenv

class OrderTakingAgent:
    def __init__(self, recommendation_agent):
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )

        self.model_id = "meta.llama3-1-8b-instruct-v1:0"
        self.model_inference_profile = 'arn:aws:bedrock:us-east-1:823413233438:inference-profile/us.meta.llama3-1-8b-instruct-v1:0'

        self.recommendation_agent = recommendation_agent

    def get_response(self, messages):
        messages = deepcopy(messages)

        system_prompt = """
            You are an Order taking agent for a coffee shop called "Merry's way".

            STRICT OUTPUT RULES:
            - You MUST reply **only in valid JSON**.
            - Any other output will crash the system.
            - NEVER include conversational text outside the JSON object.
            - If you need to ask questions, do it inside the "response" field.

            Format (Follow exactly):

            {
                "chain of thought": Short reasoning as a string
                "step number": <number>
                "order": 
                    [{
                        "item": "item name", 
                        "quantity": <number>, 
                        "price": "<item total price>" 
                    }]
                "response": "Ask about additional items or finalize the bill."
            }

            You're task is as follows:

            1. Take the User's Order
                IMPORTANT:
                *STRICTLY catch the order item (even if small spelling mistake is present)
                *STRICTLY catch the order quantity
                *You MUST not make mistake in catching the items' names and quantities

            2. Validate that all their items are in the menu. Here is the menu for this coffee shop.

                Cappuccino - $4.50
                Jumbo Savory Scone - $3.25
                Latte - $4.75
                Chocolate Chip Biscotti - $2.50
                Espresso shot - $2.00
                Hazelnut Biscotti - $2.75
                Chocolate Croissant - $3.75
                Dark chocolate (Drinking Chocolate) - $5.00
                Cranberry Scone - $3.50
                Croissant - $3.25
                Almond Croissant - $4.00
                Ginger Biscotti - $2.50
                Oatmeal Scone - $3.25
                Ginger Scone - $3.50
                Chocolate syrup - $1.50
                Hazelnut syrup - $1.50
                Carmel syrup - $1.50
                Sugar Free Vanilla syrup - $1.50
                Dark chocolate (Packaged Chocolate) - $3.00

            3. if an item is not in the menu let the user know (and repeat back the remaining valid order if any)
            4. IMPORTANT: Ask them if they need anything else. 
            5. If they do: repeat starting from step 3
            6. If they don't want anything else: 
                Using the "order" object that is in the output, 
                *Make sure to hit all three points
                1. List down all ordered items and their prices
                2. SUPER CRITICAL: Calculate the total price without any mistake
                3. Thank the user for the order and close the conversation with no more questions

            The user message will contain a section called memory. This section will contain the following:
            "order"
            "step number"

            please utilize this information to determine the next step in the process.

            IMPORTANT: 
            - DO NOT tell the user to go to the cash counter
            - If the user adds a new item, APPEND it to the existing "order" array.
            - If the user says "done", finalize the order and include the total.

            CRITICAL: 
            - No comments
            - No trailing commas 
            - No extra explanations
            - ONLY output valid JSON
            - The system parses your output with json.loads(). If you output anything else, it will crash.

        """

        last_order_taking_status = ""
        asked_recommendation_before = False

        for message_index in range(len(messages)-1, 0, -1):
            message = messages[message_index]
            
            agent_name = message.get("memory", {}).get("agent", "")
            print('agent name: ', agent_name)
            if message.get("role") == "assistant" and agent_name == "order_taking_agent":
                step_number = message['memory']['step number']
                order = message['memory']['order']
                asked_recommendation_before = message['memory']['asked_recommendation_before']

                print('step number: ', step_number)
                print('order: ', order)
                print('asked recommendation before: ', asked_recommendation_before)

                last_order_taking_status = f"""
                step number: {step_number}
                order: {order}
                """

                break

        # messages[-1]['content'] = last_order_taking_status + "\n" + messages[-1]['content']

        # input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]
        input_messages = [{"role": "system", "content": system_prompt}]

        # Add last memory if exists
        if last_order_taking_status:
            input_messages.append({
                "role": "system",
                "content": f"Previous state:\n{last_order_taking_status}"
            })

        # Add latest user message
        input_messages.append(messages[-1])

        print('input_messages (order taking): ', input_messages)

        chatbot_response = get_chatbot_response(self.client, self.model_inference_profile, input_messages)
        print('chatbot_response (before double check): ', chatbot_response)
        
        chatbot_response = double_check_json_output(self.client, self.model_inference_profile, chatbot_response)
        print('chatbot_response (after double check): ', chatbot_response)

        output = self.postprocess(chatbot_response, messages, asked_recommendation_before)
        print('processed output JSON (order taking): ', output)

        return output
    
    def postprocess(self, output, messages, asked_recommendation_before):
        # output = json.loads(output)
        print('output (chatbot_response fed to postprocess from get_response): ', output)

        output_json = self.safe_json_load(output)
        print('output_json (in postprocess): ', output_json)

        output_json = self.safe_json_load(output)

        # Final fallback
        if not output_json:
            print("Invalid JSON received. Skipping response.")
            return {
                "role": "assistant",
                "content": "Sorry, I couldn't process that order. Could you repeat it?",
                "memory": {
                    "agent": "order_taking_agent",
                    "step number": 1,
                    "asked_recommendation_before": False,
                    "order": []
                }
            }

        # if type(output['order']) == str:
        #     output['order'] = json.loads(output['order'])

        # Parse order
        if isinstance(output_json.get("order"), str):
            try:
                output_json["order"] = json.loads(output_json["order"])
            except json.JSONDecodeError:
                output_json["order"] = []


        # response = output['response']
        response = output_json.get("response", "").strip()
        order_list = output_json.get("order", [])

        print("In postprocess:")
        print('response: ', response)
        print('order list: ', order_list)

        # if not asked_recommendation_before and len(output['order']) > 0:
        #     recommendation_output = self.recommendation_agent.get_recommendations_from_order(messages, output['order'])
        #     response = recommendation_output['content']
        #     asked_recommendation_before = True

        # dict_output = {
        #     "role": "assistant",
        #     "content": response,
        #     "memory": {
        #         "agent": "order_taking_agent",
        #         "step number": output.get("step number", 1),
        #         "asked recommendation before": asked_recommendation_before,
        #         "order": output['order']
        #     }
        # }

        # return dict_output

         # Only fetch recommendations if:
        # - We haven’t already asked before
        # - There is at least one valid item

        if not asked_recommendation_before and order_list:
            rec_output = self.recommendation_agent.get_recommendations_from_order(messages, order_list)
            print('rec_output: ', rec_output)

            if isinstance(rec_output, dict) and "content" in rec_output:
                rec_text = rec_output["content"].strip()
                print('rec_text: ', rec_text)

                # Simple sanity check to avoid "apocalypse" repeats
                if rec_text and len(rec_text.split()) > 3 and rec_text.lower() != response.lower():
                    response += "\n\n" + rec_text
                    asked_recommendation_before = True

        return {
            "role": "assistant",
            "content": response,
            "memory": {
                "agent": "order_taking_agent",
                "step number": output_json.get("step number", 1),
                "asked_recommendation_before": asked_recommendation_before,
                "order": order_list
            }
        }
    
    def safe_json_load(self, output: str):
        """Extract the first valid JSON object from a string."""

        print("output (fed to safe_json_load from postprocess):", output)

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
                        print("json_obj (in safe_json_load):", json_obj)
                        return json_obj
                    except json.JSONDecodeError:
                        continue
        # return None

        # Final fallback: ask model to repair JSON
        repaired = get_chatbot_response(
            self.client,
            self.model_inference_profile,
            [
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON. Nothing else."
                },
                {"role": "user", "content": output}
            ]
        )
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            print("safe_json_load: Model failed to repair JSON")
            return None
