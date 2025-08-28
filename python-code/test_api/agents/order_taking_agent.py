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
        self.model_id = "meta.llama3-1-8b-instruct-v1:0"
        self.model_inference_profile = (
            "arn:aws:bedrock:us-east-1:823413233438:inference-profile/us.meta.llama3-1-8b-instruct-v1:0"
        )
        self.recommendation_agent = recommendation_agent

    # ---------------------------
    # Public Method
    # ---------------------------
    def get_response(self, messages):
        messages = deepcopy(messages)

        # Get previous memory or default values
        memory = self._extract_last_memory(messages)
        order = memory.get("order", [])
        step_number = memory.get("step_number", 1)
        asked_recommendation_before = memory.get("asked_recommendation_before", False)
        order_id = memory.get("order_id", str(uuid.uuid4()))
        order_finalized = memory.get("order_finalized", False)

        user_message = messages[-1]["content"].lower()

        # ---------------------------
        # Handle order status inquiry
        # ---------------------------
        if any(kw in user_message for kw in ["ordered?", "did you place", "what's my order", "my bill"]):
            if order_finalized:
                return self._generate_response(
                    "✅ Your order has already been placed and finalized. Thank you! ☕",
                    order, order_id, step_number, True
                )
            elif order:
                order_summary = self._generate_order_summary(order)
                return self._generate_response(
                    f"So far, you have ordered:\n{order_summary}\n\nWould you like to add more or finalize?",
                    order, order_id, step_number, False
                )
            else:
                return self._generate_response(
                    "You haven't ordered anything yet. Would you like to start?",
                    order, order_id, step_number, False
                )

        # ---------------------------
        # Reset if previous order finalized and new one starts
        # ---------------------------
        if order_finalized and any(kw in user_message for kw in ["order", "want", "give me", "buy", "need"]):
            order = []
            step_number = 1
            asked_recommendation_before = False
            order_finalized = False
            order_id = str(uuid.uuid4())

        # ---------------------------
        # Build prompt for LLM
        # ---------------------------
        system_prompt = self._build_system_prompt(order)
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_response = get_chatbot_response(self.client, self.model_inference_profile, input_messages)
        chatbot_response = double_check_json_output(self.client, self.model_inference_profile, chatbot_response)
        output_json = self._safe_json_load(chatbot_response)

        # ---------------------------
        # Fallback if JSON fails
        # ---------------------------
        if not output_json:
            return self._generate_response(
                "⚠️ Sorry, I couldn't process your order. Could you please repeat?",
                order, order_id, step_number, order_finalized
            )

        # ---------------------------
        # Merge new items into order list
        # ---------------------------

        # From LLM output
        llm_order_items = output_json.get("order", [])
        if isinstance(llm_order_items, dict):
            llm_order_items = [llm_order_items]

        print("LLM order items:", llm_order_items)

        # Convert to lowercase for safety
        new_items = [
            {
                "item": item["item"].strip(),
                "quantity": item.get("quantity", 1),
                "price": item.get("price", 0)
            }
            for item in llm_order_items
        ]

        print("New items:", new_items)

        for new_item in new_items:
            found = False
            for existing_item in order:
                if existing_item["item"].lower() == new_item["item"].lower():
                    print("Found existing item:", existing_item)
                    print("New item:", new_item)
                    # ✅ Overwrite price calculation
                    existing_item["quantity"] = new_item["quantity"]
                    existing_item["price"] = new_item["price"]
                    # existing_item["price"] = existing_item["quantity"] * (new_item["price"] / new_item["quantity"])
                    found = True
                    break
            if not found:
                order.append(new_item)

        memory["order"] = order

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
        # If user finalizes the order
        # ---------------------------
        if any(kw in user_message for kw in ["done", "checkout", "finalize", "that's all", "no more", "nothing else"]):
            total_price = self._calculate_total(order)
            order_finalized = True
            summary = self._generate_order_summary(order, total_price)
            return self._generate_response(
                f"🧾 Here's your order summary:\n{summary}\n\nTotal: ${total_price:.2f}\n\nThank you for ordering from Merry's Way! ☕",
                order, order_id, step_number + 1, order_finalized
            )

        # ---------------------------
        # Smart Recommendations (Apriori)
        # ---------------------------
        if not asked_recommendation_before and new_items:
            rec_output = self.recommendation_agent.get_recommendations_from_order(messages, order)
            response = output_json.get("response", "").strip()
            if isinstance(rec_output, dict) and "content" in rec_output:
                rec_text = rec_output["content"].strip()
                if rec_text:
                    response += f"\n\n{rec_text}"
                    asked_recommendation_before = True
        else:
            response = output_json.get("response", "").strip()

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

    def _build_system_prompt(self, order):
        return f"""
        You are an Order Taking Agent for Merry's Way Coffee Shop.
        Always respond in valid JSON only, no extra text.

        Format:
        {{
            "step_number": <number>,
            "order": [{{"item": "Latte", "quantity": 2, "price": 9.5}}],
            "response": "Ask for additional items or finalize."
        }}

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

        Previous Order: {json.dumps(order, ensure_ascii=False)}
        """

    def _calculate_total(self, order):
        total = sum((item["price"] * item["quantity"]) for item in order)
        return total

    def _generate_order_summary(self, order, total=None):
        summary = "\n".join([f"- {item['quantity']} x {item['item']} = ${item['quantity'] * float(item['price']):.2f}" for item in order])
        if total is not None:
            summary += f"\n\nTotal = ${total:.2f}"
        return summary

    def _generate_response(self, response, order, order_id, step_number, finalized=False, asked_recommendation_before=False):
        return {
            "role": "assistant",
            "content": response,
            "memory": {
                "agent": "order_taking_agent",
                "step_number": step_number,
                "order": order,
                "order_id": order_id,
                "order_finalized": finalized,
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
