import boto3
import json
import uuid
from copy import deepcopy
from .utils import get_chatbot_response, double_check_json_output


class OrderTakingAgent:
    def __init__(self, recommendation_agent):
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )
        self.model_id = "meta.llama3-3-70b-instruct-v1:0"
        self.model_inference_profile = (
            "arn:aws:bedrock:us-east-1:823413233438:inference-profile/us.meta.llama3-3-70b-instruct-v1:0"
        )
        self.recommendation_agent = recommendation_agent

    # ---------------------------
    # Public Method
    # ---------------------------
    def get_response(self, messages):
        messages = deepcopy(messages)

        # print("messages:", messages)

        # Get previous memory or default values
        memory = self._extract_last_memory(messages)
        order = memory.get("order", [])
        step_number = memory.get("step_number", 1)
        asked_recommendation_before = memory.get("asked_recommendation_before", False)
        order_id = memory.get("order_id", str(uuid.uuid4()))
        order_finalized = memory.get("order_finalized", False)

        user_message = messages[-1]["content"].lower()

        # ---------------------------
        # Intent classification using LLM
        # ---------------------------
        system_prompt_for_intent_classification = self._build_system_prompt_for_order_intents_classification(user_message)
        input_messages_for_intent_classification = [{"role": "system", "content": system_prompt_for_intent_classification}] + messages

        # print("Input messages (intent classification):", input_messages_for_intent_classification)

        chatbot_response_for_intent_classification = get_chatbot_response(self.client, self.model_inference_profile, input_messages_for_intent_classification)
        chatbot_response_for_intent_classification = double_check_json_output(self.client, self.model_inference_profile, chatbot_response_for_intent_classification)
        output_json_for_intent_classification = self._safe_json_load(chatbot_response_for_intent_classification)

        # print("Output JSON (intent classification):", output_json_for_intent_classification)

        thought = output_json_for_intent_classification.get("chain_of_thought", None)
        intent = output_json_for_intent_classification.get("intent", None)
        item = output_json_for_intent_classification.get("details").get("item", None)
        quantity = output_json_for_intent_classification.get("details").get("quantity", None)
        change = output_json_for_intent_classification.get("details").get("change", None)
        price = output_json_for_intent_classification.get("details").get("price", None)
        # response = output_json_for_intent_classification.get("response", None)

        print("Thought:", thought)
        print("Intent:", intent)
        print("Item:", item)
        print("Quantity:", quantity)
        print("Change:", change)
        print("Price:", price)
        # print("Response:", response)

        # ---------------------------
        # Fallback if JSON fails
        # ---------------------------
        if not output_json_for_intent_classification:
            return self._generate_response(
                "⚠️ Sorry, I couldn't process your order. Could you please repeat?",
                order, order_id, step_number, order_finalized
            )
        
        # ----------------------------------
        # Creating order based on intent identified by LLM
        # ----------------------------------

        if intent == "ADD_NEW_ITEM":
            order.append({"item": item, "quantity": quantity, "price": price})
        elif intent == "INCREASE_LAST_ITEM":
            order[-1]["quantity"] += quantity
        elif intent == "DECREASE_LAST_ITEM":
            order[-1]["quantity"] -= quantity
        elif intent == "INCREASE_QUANTITY":
            order.append({"item": item, "quantity": quantity, "price": price})
        elif intent == "DECREASE_QUANTITY":
            order.append({"item": item, "quantity": -quantity, "price": price})
        elif intent == "UPDATE_QUANTITY":
            for o in order:
                if o["item"] == item:
                    o["quantity"] = quantity
        elif intent == "CANCEL":
            order = [o for o in order if o["item"] != item]
        elif intent == "SHOW_LIST":
            total_price = self._calculate_total(order)
            summary = self._generate_order_summary(order, total_price)
            return self._generate_response(
                f"🧾 Here's your order summary:\n{summary}\n\nTotal: ${total_price:.2f}\n\nYou would like to add anything else or finalize the order?",
                order, order_id, step_number + 1, order_finalized
            )
        elif intent == "FINALIZE_ORDER":
            total_price = self._calculate_total(order)
            order_finalized = True
            summary = self._generate_order_summary(order, total_price)
            return self._generate_response(
                f"🧾 Here's your order summary:\n{summary}\n\nTotal: ${total_price:.2f}\n\nThank you for ordering from Merry's Way! ☕",
                order, order_id, step_number + 1, order_finalized
            )
        elif intent == "UNCLEAR":
            return self._generate_response(
                "I'm sorry, I didn't understand. Could you please clarify?",
                order, order_id, step_number, order_finalized
            )
        elif intent == "UNAVAILABLE":
            return self._generate_response(
                "I'm sorry, this item currently unavailable. Please try again later.",
                order, order_id, step_number, order_finalized
            )

        print("Order:", order)


        # ---------------------------
        # Update memory
        # ---------------------------
        memory = {
            "order": order,
            "step_number": step_number,
            "asked_recommendation_before": asked_recommendation_before,
            "order_id": order_id,
            "order_finalized": order_finalized
        }
        messages[-1]["memory"] = memory

        # ---------------------------
        # Generate response using LLM
        # ---------------------------
        
        system_prompt = self._build_system_prompt(order)
        input_messages = [{"role": "system", "content": system_prompt}]

        print("Input messages (response generation feed):", input_messages)

        chatbot_response = get_chatbot_response(self.client, self.model_inference_profile, input_messages)
        chatbot_response = double_check_json_output(self.client, self.model_inference_profile, chatbot_response)
        output_json = self._safe_json_load(chatbot_response)

        print("Output JSON (order taking):", output_json)

        response = output_json.get("response", None)

        # print("Thought:", thought)
        # print("Intent:", intent)
        # print("Item:", item)
        # print("Quantity:", quantity)
        # print("Change:", change)
        # print("Price:", price)
        # print("Response:", response)

        # ---------------------------
        # Fallback if JSON fails
        # ---------------------------
        # if not output_json:
        #     return self._generate_response(
        #         "⚠️ Sorry, I couldn't process your order. Could you please repeat?",
        #         order, order_id, step_number, order_finalized
        #     )

        # ---------------------------
        # Smart Recommendations (Apriori)
        # ---------------------------

        items = [
            "Cappuccino",
            "Jumbo Savory Scone",
            "Latte",
            "Chocolate Chip Biscotti",
            "Espresso shot",
            "Hazelnut Biscotti",
            "Chocolate Croissant",
            "Dark chocolate (Drinking Chocolate)",
            "Cranberry Scone",
            "Croissant",
            "Almond Croissant",
            "Ginger Biscotti",
            "Oatmeal Scone",
            "Ginger Scone",
            "Chocolate syrup",
            "Hazelnut syrup",
            "Carmel syrup",
            "Sugar Free Vanilla syrup",
            "Dark chocolate (Packaged Chocolate)"
        ]


        if not asked_recommendation_before and order[0]["item"] in items and order[0]["quantity"] is not None and order[0]["quantity"] > 0:
            rec_output = self.recommendation_agent.get_recommendations_from_order(messages, order)
            # response = output_json.get("response", "").strip()
            total_price = self._calculate_total(order)
            summary = self._generate_order_summary(order, total_price)
            # response = f"🧾 Here's your order summary:\n{summary}"
            if isinstance(rec_output, dict) and "content" in rec_output:
                rec_text = rec_output["content"].strip()
                if rec_text:
                    response = f"\n\nBased on your order, here are some recommendations:\n{rec_text}"
                    asked_recommendation_before = True
        # else:
        #     response = output_json.get("response", "").strip()

        return self._generate_response(
            response, order, order_id, step_number + 1, order_finalized, asked_recommendation_before
        )
        
    # ---------------------------
    # Helper Functions
    # ---------------------------
    def _extract_last_memory(self, messages):
        """Get last state from memory if exists."""
        for msg in reversed(messages):
            mem = msg.get("memory", {})
            if mem.get("agent") == "order_taking_agent":
                return mem
        return {}

    def _build_system_prompt_for_order_intents_classification(self, user_message):

            return f"""
            You are a chatbot that tries to understand the order intent of a user.
            Your task is to determine the intent of the new message and respond accordingly.

            New Message:
            {user_message}

            INTENTS:

            -ADD_NEW_ITEM
            -INCREASE_LAST_ITEM
            -DECREASE_LAST_ITEM
            -INCREASE_QUANTITY
            -DECREASE_QUANTITY
            -UPDATE_QUANTITY
            -CANCEL
            -SHOW_LIST
            -FINALIZE_ORDER
            -UNCLEAR
            -UNAVAILABLE

            CRITICAL: 
            Return only a JSON of this structure:
            {{
                "intent": "INTENT_NAME",
                "chain_of_thought": "reasoning"
                "details": {{
                    "item": "string" | null, 
                    "quantity": number | null, 
                    "price": float | null
                }}
            }}

            If not sure of the intent, use some of these styles of talking to understand the intent of the user message:

            1) ADD_NEW_ITEM: "want", "need", "i'd like", "order something", "buy", "give me", "add" or only item name
            2) UPDATE_QUANTITY: "sorry four", "update", "change", "wait five actually"
            3) INCREASE_LAST_ITEM: "increase", "add", "more", "another", "1 more", "one more", "two actually"
            4) DECREASE_LAST_ITEM: "reduce", "less", "decrease", "subtract two", "delete"
            4) INCREASE_QUANTITY: "more", "add", "add", "one more" WITH SPECIFIC ITEM NAME
            5) DECREASE_QUANTITY: "reduce", "less", "decrease", "subtract two", "delete" WITH SPECIFIC ITEM NAME
            6) CANCEL_ITEM: "cancel", "remove", "delete", "remove" WITH SPECIFIC ITEM NAME
            7) SHOW_LIST: "show", "list", "show order summary", "show order"
            8) FINALIZE_ORDER: "finalize", "final", "done", "confirm", "place", "order", "nothing else", "that's all", "checkout"
            9) UNCLEAR: "my usual", "repeat order", "give something", "order a coffee" (not specific item) etc. i.e. when order is unclear or generic.  
            10) UNAVAILABLE: if user's order is specific but that item is unavailable in the menu

            **CRITICAL: If user does NOT want to add or change the order (saying just "no", "nothing else" etc.), intent is "FINALIZE_ORDER"

            **IMPORTANT: DO NOT overinterprete when user message is unclear. If unclear, intent is "UNCLEAR". Always validate against the menu.

            Menu:
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
            """

    def _build_system_prompt(self, order):
        return f"""
        You are an Order Taking Agent for Merry's Way Coffee Shop.
        Here is your current order:
        {order}

        Always respond in valid JSON only, no extra text.

        Format:
        {{
            "order": [{{"item": "item name", "quantity": <quantity>, "price": <price>}}],
            "response": "Give details of the order and ask for additional items or ask to finalize."
        }}
        """

    def _calculate_total(self, order):
        total = sum((item["price"] * item["quantity"]) for item in order)
        return total

    def _generate_order_summary(self, order, total=None):
        print("Order:", order)
        for item in order:
            print("Item:", item["item"], "Quantity:", item["quantity"], "Price:", item["price"])

        summary = "\n".join([f"- {item['quantity']} x {item['item']} = ${item['quantity'] * float(item['price']):.2f}" for item in order])
        if total is not None:
            summary += f"\n\nTotal = ${total:.2f}"
        return summary

    def _generate_response(self, response, order, order_id, step_number, order_finalized=False, asked_recommendation_before=False):
        return {
            "role": "assistant",
            "content": response,
            "memory": {
                "agent": "order_taking_agent",
                "step_number": step_number,
                "order": order,
                "order_id": order_id,
                "order_finalized": order_finalized,
                "asked_recommendation_before": asked_recommendation_before
            }
        }

    def _safe_json_load(self, output: str):
        try:
            return json.loads(output)
        except:
            try:
                start_idx = output.find("{")
                end_idx = output.rfind("}")
                return json.loads(output[start_idx:end_idx + 1])
            except:
                return None
