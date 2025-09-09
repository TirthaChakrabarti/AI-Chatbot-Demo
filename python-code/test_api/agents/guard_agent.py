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
        You are a JSON-only content filter for a coffee shop AI assistant.

        YOUR TASK: Analyze the user's message in context and determine if it should be ALLOWED or NOT ALLOWED.

        IMPORTANT: Consider the conversation history to understand what the user is asking for. Look at the LATEST USER MESSAGE but use previous context to understand references like "it", "that", "more", etc.

        Output ONLY this JSON structure:
        {
            "chain_of_thought": "Explain what the user is actually asking for (considering context) and which rule applies",
            "decision": "allowed" or "not allowed",
            "message": "" if allowed, otherwise "Return a polite, engaging reply based on the reason. 
                        Examples:
                        - If user asks for non-coffee items → 'Oops! We don't have that here. Want me to recommend something from our menu instead?'
                        - If user asks for unrelated jokes → 'Haha, I wish I knew jokes, but I'm better at coffee recommendations! Want me to suggest something?'
                        - If user sends inappropriate content → 'I'm here to help with coffee orders only. Let's keep it friendly 😊'
                        - If user asks about alcohol or drugs → 'We don't serve that, but we have amazing non-alcoholic drinks!'
                        Always keep it friendly, natural, and conversational."
        }

        ALLOWED REQUESTS:
        1. Coffee shop information: location, hours, services, delivery areas
        2. Menu items: coffee, tea, chocolate, pastries, bakery items, flavors, non-alcoholic drinks
        3. Item details: ingredients, descriptions, prices
        4. Orders: add, remove, modify quantities, change sizes/options
        5. Recommendations: asking for suggestions (assume coffee shop items if unspecified)
        6. AI assistant questions: asking about capabilities, purpose
        7. Social interactions: greetings, thanks, goodbyes (unless clearly unrelated)

        NOT ALLOWED REQUESTS:
        1. Unrelated topics, items, or services outside coffee shop scope
        2. Employee personal information
        3. Recipes or preparation instructions
        4. Recommendations for non-coffee shop items
        5. Harmful content: adult content, violence, hate speech, drugs, alcohol

        CONTEXT EXAMPLES:
        Conversation: 
        User: "I'd like a cappuccino"
        Assistant: "One cappuccino added to order. Anything else?"
        User: "No...Nothing else"
        → ALLOWED (order completed)

        Conversation:
        User: "What pastries do you have?"  
        Assistant: "We have croissants, scones..."
        User: "Add 2 croissants to my order"
        → ALLOWED (adding pastries to order)

        Conversation:
        User: "Hi there"
        Assistant: "Hello! How can I help?"
        User: "Tell me a joke"
        → NOT ALLOWED (unrelated request)

        Analyze the conversation context to understand what the user actually wants.
        """

        # Get contextual conversation (last 6 messages or all if fewer)
        # This includes enough context for ordering conversations
        context_messages = message[-6:] if len(message) > 6 else message
        
        # Find the latest user message for focused analysis
        latest_user_message = None
        for msg in reversed(message):
            if msg["role"] == "user":
                latest_user_message = msg["content"]
                break
        
        if not latest_user_message:
            return self._create_error_response("No user message found")

        # Create input messages with context
        input_messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation context (but limit to avoid token overflow)
        for msg in context_messages:
            input_messages.append({
                "role": msg["role"], 
                "content": msg["content"]
            })
        
        print("Input Messages (Guard):", input_messages)

        max_turns = 3
        
        for turn in range(1, max_turns + 1):
            if turn > 1:
                retry_prompt = f"""
                RETRY ATTEMPT {turn}/{max_turns}:
                Your previous response was not valid JSON. 

                REQUIREMENTS:
                - Output ONLY valid JSON
                - No markdown formatting  
                - No extra text before or after JSON
                - Consider the conversation context
                - Focus on the latest user message: '{latest_user_message}'
                - Provide accurate reasoning based on what user actually asked

                Remember: Use context to understand references but focus classification on the latest message.
                """
                input_messages[0]["content"] = system_prompt + "\n\n" + retry_prompt
            
            try:
                chatbot_output = get_chatbot_response(self.client, self.model_inference_profile, input_messages)
                print(f"Chatbot output (Guard - Turn {turn}):", chatbot_output)

                output = self.postprocess(chatbot_output, latest_user_message, context_messages)
                print(f"Processed Output (Guard - Turn {turn}):", output)

                # Validate that the decision is correct
                if output["memory"]["guard_decision"] in ("allowed", "not allowed"):
                    # Additional validation: check if reasoning makes sense with context
                    if self._validate_contextual_reasoning(latest_user_message, context_messages, 
                                                         output["memory"].get("reason", ""), 
                                                         output["memory"]["guard_decision"]):
                        return output
                    else:
                        print(f"⚠ Turn {turn}: Reasoning doesn't match context. Retrying...")
                else:
                    print(f"⚠ Turn {turn}: Invalid decision format. Retrying...")
                    
            except Exception as e:
                print(f"⚠ Turn {turn}: Error processing response: {e}")

        # Final fallback
        return self._create_error_response("Guard agent failed to provide valid response after multiple attempts")
    
    def _validate_contextual_reasoning(self, user_message, context_messages, reasoning, decision):
        """Validate reasoning considering conversation context"""
        user_lower = user_message.lower()
        reasoning_lower = reasoning.lower()
        
        # Get conversation context
        context_text = " ".join([msg["content"].lower() for msg in context_messages[-4:]])
        
        # Check for order-related context
        order_indicators = ["order", "add", "remove", "change", "modify", "make it", "i want", "i'd like"]
        has_order_context = any(indicator in context_text for indicator in order_indicators)
        
        # Validate order modifications
        modification_patterns = ["make it", "change", "add", "remove", "modify", "more", "less", "large", "small"]
        if any(pattern in user_lower for pattern in modification_patterns):
            if has_order_context and "order" not in reasoning_lower and "modif" not in reasoning_lower:
                return False
        
        # Validate contextual references
        contextual_refs = ["it", "that", "this", "them", "those"]
        if any(ref in user_lower for ref in contextual_refs):
            if not has_order_context and decision == "allowed":
                # If there's a contextual reference but no clear coffee shop context, be more careful
                coffee_context = any(word in context_text for word in 
                                   ["coffee", "latte", "cappuccino", "espresso", "pastry", "menu", "drink"])
                if not coffee_context:
                    return False
        
        # Original validation logic
        greeting_patterns = ["hi", "hello", "who are you", "what can you", "how can you help"]
        if any(pattern in user_lower for pattern in greeting_patterns):
            if "greeting" not in reasoning_lower and "introduction" not in reasoning_lower and "assistant" not in reasoning_lower:
                return False
        
        recommendation_patterns = ["recommend", "suggest", "what should i"]
        if any(pattern in user_lower for pattern in recommendation_patterns):
            if "recommend" not in reasoning_lower and "suggest" not in reasoning_lower:
                return False
                
        return True
    
    def _create_error_response(self, error_msg):
        """Create a standard error response"""
        return {
            "role": "assistant",
            "content": "Sorry, your message could not be processed. Please try again.",
            "memory": {
                "agent": "Guard",
                "guard_decision": "not allowed",
                "reason": error_msg
            }
        }
    
    def _extract_conversation_summary(self, messages):
        """Extract key context from conversation for better understanding"""
        summary_parts = []
        
        for msg in messages[-4:]:  # Last 4 messages for context
            role = msg["role"]
            content = msg["content"][:100]  # Truncate for brevity
            summary_parts.append(f"{role}: {content}")
            
        return " | ".join(summary_parts)
    
    def postprocess(self, output, user_message, context_messages):
        if not output or not output.strip():
            raise ValueError("Chatbot output is empty")

        # Clean the output first
        output = output.strip()
        
        # Try to find and extract JSON
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
                        json_str = output[start_idx:i+1]
                        json_obj = json.loads(json_str)
                        print("Extracted JSON (Guard):", json_obj)
                        break
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        continue

        if json_obj is None:
            print("⚠ No valid JSON found in output")
            raise ValueError("No valid JSON found")

        # Validate required fields
        required_fields = ["chain_of_thought", "decision"]
        for field in required_fields:
            if field not in json_obj:
                raise ValueError(f"Missing required field: {field}")

        decision = json_obj.get("decision", "").lower().strip()
        if decision not in ("allowed", "not allowed"):
            raise ValueError(f"Invalid decision value: {decision}")

        return {
            "role": "assistant",
            "content": json_obj.get("message", ""),
            "memory": {
                "agent": "Guard",
                "guard_decision": decision,
                "reason": json_obj.get("chain_of_thought", ""),
                "context_summary": self._extract_conversation_summary(context_messages)
            }
        }