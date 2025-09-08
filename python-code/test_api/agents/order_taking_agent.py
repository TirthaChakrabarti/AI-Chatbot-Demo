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
        # Intent classification using LLM (extract one or multiple actions)
        # ---------------------------
        system_prompt_for_intent_classification = self._build_system_prompt_for_order_intents_classification(user_message)
        input_messages_for_intent_classification = [{"role": "system", "content": system_prompt_for_intent_classification}] + messages

        # print("Input messages (intent classification):", input_messages_for_intent_classification)

        chatbot_response_for_intent_classification = get_chatbot_response(self.client, self.model_inference_profile, input_messages_for_intent_classification)
        chatbot_response_for_intent_classification = double_check_json_output(self.client, self.model_inference_profile, chatbot_response_for_intent_classification)
        output_json_for_intent_classification = self._safe_json_load(chatbot_response_for_intent_classification)

        print("Output JSON (intent classification):", output_json_for_intent_classification)

        # ---------------------------
        # Fallback if JSON fails
        # ---------------------------
        if not output_json_for_intent_classification:
            return self._generate_response(
                "⚠️ Sorry, I couldn't process your order. Could you please rephrase?",
                order, order_id, step_number, order_finalized
            )
        
        # ----------------------------------
        # Creating order based on intent identified by LLM
        # ----------------------------------

        actions = []
        isArray = False

        if "actions" in output_json_for_intent_classification and isinstance(output_json_for_intent_classification["actions"], list):
            actions = output_json_for_intent_classification["actions"]
            isArray = True
        else:
            # fallback to single-intent style
            intent = output_json_for_intent_classification.get("intent")
            details = output_json_for_intent_classification.get("details", {}) 
            actions = self._translate_intent_to_action(intent, details)

        # apply action sequentially
        for act in actions:
            act_type = act.get("type")
            item = act.get("item")
            qty = act.get("quantity")
            price = act.get("price")

            # normalize
            if isinstance(item, str):
                item = item.strip()

            # Safety checks
            if act_type == "add":
                if not item or not qty or qty <= 0:
                    continue
                self._add_or_merge_item(order, item, qty, price)
            elif act_type == "update":
                if not item:
                    continue
                if qty is None:
                    continue
                self._set_item_quantity(order, item, qty)
            elif act_type == "remove":
                if not item:
                    continue
                order = [o for o in order if o["item"].lower() != item.lower()]
            elif act_type == "increase_last":
                if not order or qty is None:
                    continue
                order[-1]["quantity"] += qty
            elif act_type == "decrease_last":
                if not order or qty is None:
                    continue
                order[-1]["quantity"] = max(0, order[-1]["quantity"] - qty)
            elif act_type == "unavailable":
                if not item:
                    continue
                if isArray:
                    return self._generate_response(
                        f"I'm sorry, {item} is currently unavailable. Please try again later. \n\nOther items have been added to your order.",
                        order, order_id, step_number, order_finalized
                    )
                return self._generate_response(
                    f"I'm sorry, {item} is currently unavailable. Please try again later. Would you like to try something else?",
                    order, order_id, step_number, order_finalized
                )
            elif act_type == "negotiation":
                return self._generate_response(
                    f"I'm sorry, {item} is not available in the asked price.",
                    order, order_id, step_number, order_finalized
                )
            elif act_type == "nothing_else":
                return self._generate_response(
                    "Ok, I guess that's all for now. Would you like to finalize the order?",
                    order, order_id, step_number + 1, order_finalized
                )
            elif act_type == "show_list":
                if not order:
                    return self._generate_response(
                        "🧾 Your order book is empty. Would you like to order something now?", 
                        order, order_id, step_number + 1, order_finalized
                    )
                total_price = self._calculate_total(order)
                summary = self._generate_order_summary(order, total_price)
                return self._generate_response(
                    f"🧾 Here's your order summary:\n{summary}\n\nWould you like to add anything else or finalize the order?\n\n**Once finalized, order cannot be changed.",
                    order, order_id, step_number + 1, order_finalized
                )
            elif act_type == "finalize":
                if not order:
                    return self._generate_response(
                        "🧾 Your order book is empty. Would you like to order something now?", 
                        order, order_id, step_number + 1, order_finalized
                    )
                total_price = self._calculate_total(order)
                order_finalized = True
                summary = self._generate_order_summary(order, total_price)
                return self._generate_response(
                    f"Your order has been confirmed! \n\n🧾 Here's your order summary:\n{summary}\n\nThank you for ordering from Merry's Way! ☕",
                    order, order_id, step_number + 1, order_finalized
                )
            elif act_type == "unclear":
                return self._generate_response(
                    "I'm sorry, I didn't understand. Could you please clarify?",
                    order, order_id, step_number, order_finalized
                )

            # ignore unknown action types silently (or collect errors if we want)

        # Clean: remove any zero-quantity items
        order = [o for o in order if o.get("quantity", 0) > 0]

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

        # ---------------------------
        # Fallback if JSON fails
        # ---------------------------
        if not output_json:
            return self._generate_response(
                "⚠️ Sorry, something went wrong generating the response. Could you repeat?",
                order, order_id, step_number, order_finalized
            )

        print("Output JSON (order taking):", output_json)

        response = output_json.get("response", None)

        # ---------------------------
        # Smart Recommendations (Apriori)
        # ---------------------------

        if not asked_recommendation_before and order and len(order) == 1 and order[0].get("item"):
            first_item = order[0]["item"]
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
            if any(first_item.lower() == it.lower() for it in items):
                rec_output = self.recommendation_agent.get_recommendations_from_order(messages, order)
                # response = output_json.get("response", "").strip()
                total_price = self._calculate_total(order)
                summary = self._generate_order_summary(order, total_price)
                response = f"🧾 Here's your order summary:\n{summary}"
                if isinstance(rec_output, dict) and "content" in rec_output:
                    rec_text = rec_output["content"].strip()
                    if rec_text:
                        response += f"\n{rec_text}"
                        asked_recommendation_before = True


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

    def _translate_intent_to_action(self, intent, details):
        if not intent:
            return []
        it = intent.upper()
        if details is not None:
            item = details.get("item")
            qty = details.get("quantity")
            price = details.get("price")

        if it == "ADD_NEW_ITEM":
            return [{"type": "add", "item": item, "quantity": qty, "price": price}]
        if it == "INCREASE_LAST_ITEM":
            return [{"type": "increase_last", "quantity": qty or 1}]
        if it == "DECREASE_LAST_ITEM":
            return [{"type": "decrease_last", "quantity": qty or 1}]
        if it == "INCREASE_QUANTITY":
            return [{"type": "add", "item": item, "quantity": qty or 1, "price": price}]
        if it == "DECREASE_QUANTITY":
            return [{"type": "update", "item": item, "quantity": (0 if qty is None else max(0, -qty))}]
        if it == "UPDATE_QUANTITY":
            return [{"type": "update", "item": item, "quantity": qty}]
        if it == "CANCEL":
            return [{"type": "remove", "item": item}]
        if it == "UNAVAILABLE":
            return [{"type": "unavailable", "item": item}]
        if it == "NEGOTIATION":
            return [{"type": "negotiation", "item": item}]
        if it == "NOTHING_ELSE":
            return [{"type": "nothing_else"}]
        if it == "SHOW_LIST":
            return [{"type": "show_list"}] 
        if it == "FINALIZE_ORDER":
            return [{"type": "finalize"}]
        if it == "UNCLEAR":
            return [{"type": "unclear"}]
        return []

    def _build_system_prompt_for_order_intents_classification(self, user_message):

            return f"""
            You are a intent-classifier AI.
            Your output must be a valid JSON only.

            User Message:
            {user_message}

            CRITICAL: 
            Based on the user message, 
            You must return either a single-intent JSON of this structure:

            {{
                "intent": "ADD_NEW_ITEM",
                "details": {{"item": "latte", "quantity": 2, "price": 4.75}} or null,
                "chain_of_thought": "reasoning"
            }}

            ALLOWED INTENTS: ADD_NEW_ITEM, INCREASE_LAST_ITEM, DECREASE_LAST_ITEM, INCREASE_QUANTITY, DECREASE_QUANTITY, UPDATE_QUANTITY, CANCEL, UNAVAILABLE, NEGOTIATION, NOTHING_ELSE, SHOW_LIST, FINALIZE_ORDER, UNCLEAR

            OR

            for user messages containing multiple changes, return an "action" array of atomic operations in the order they should be applied. Structure:
            {{
                "actions": [
                    {{"type": "add", "item": "latte", "quantity": 2, "price": 4.75}},
                    {{"type": "add", "item": "espresso", "quantity": 1, "price": 2.0}},
                    {{"type": "update", "item": "cappuccino", "quantity": 5}},
                    {{"type": "unavailable", "item": "mocha"}},
                    {{"type": "update", "item": "caramel syrup", "quantity": 3}},
                    {{"type": "remove", "item": "croissant"}}
                ],
                "chain_of_thought": "reasoning"
            }}

            Allowed ACTIONS: add, update, remove, increase_last, decrease_last, finalize, unavailable, negotiation, unclear

            VITAL: Array must include only the latest requests, not repeat previous ones

            **If user does NOT want to add anything or change the order (saying just "no", "nothing else" etc.), intent is "NOTHING_ELSE".

            IMPORTANT: 
            - validate every single item against this menu; 
            - If item not present, return intent "UNAVAILABLE"
            - If user requests for lesser price than specified in menu, return intent "NOGOTIATION"
            - If user says words like "my usual", "repeat order", "give something", "order a coffee" (not specific item) etc. i.e. when order is unclear or generic, return intent "UNCLEAR"

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
            "response": "Give details of the order and ask for additional items or ask to finalize. Do not calculate the total."
        }}
        """

    def _add_or_merge_item(self, order, item_name, quantity, price=None):
        # merge case-insensitively
        for o in order:
            if o["item"].lower() == item_name.lower():
                o["quantity"] = int(o.get("quantity", 0)) + int(quantity)
                if price is not None:
                    o["price"] = price
                return
        # new item
        order.append({"item": item_name, "quantity": int(quantity), "price": float(price) if price is not None else 0.0})

    def _set_item_quantity(self, order, item_name, quantity):
        for o in order:
            if o["item"].lower() == item_name.lower():
                o["quantity"] = int(quantity)
                return
        # if item not found, create it (choice: you can also ignore instead)
        order.append({"item": item_name, "quantity": int(quantity), "price": 0.0})

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
