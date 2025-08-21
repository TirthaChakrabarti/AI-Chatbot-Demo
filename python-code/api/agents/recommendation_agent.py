from openai import OpenAI
import pandas as pd
import json
import re
from copy import deepcopy   # deepcopy copies by value and not by reference
import os
from .utils import get_chatbot_response, double_check_json_output
import dotenv

dotenv.load_dotenv()

class RecommendationAgent:
    def __init__(self, apriori_recommendation_path, popularity_recomendation_path):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"  
        )
        self.model_name = "phi3"

        # loading JSON object of the apriori algorithm
        with open(apriori_recommendation_path, 'r') as json_file:
            self.apriori_recommendations = json.load(json_file)

        # print('apriori recommendations path:', apriori_recommendation_path)
        # print('apriori recommendations from data:', self.apriori_recommendations)

        self.popularity_recommendations = pd.read_csv(popularity_recomendation_path)

        # print('popularity recommendations path:', popularity_recomendation_path)
        # print('popularity recommendations from data:', self.popularity_recommendations)

        self.products = self.popularity_recommendations['product'].tolist()
        self.product_categories = list(set(self.popularity_recommendations['product_category'].tolist()))

        print('products:', self.products)
        print('product categories:', self.product_categories)

    def get_apriori_recommendations(self, products, top_k=5):
        recommendation_list = []

        for product in products:
            if product in self.apriori_recommendations:
                # recommendation_list.extend(self.apriori_recommendations[product][:top_k])
                print('ordered product one:', product)

                recommendation_list += self.apriori_recommendations[product]
                print('apriori recommendations list:', self.apriori_recommendations[product])
                print('recommendation list:', recommendation_list)

        # Sort recommendation list based on "confidence" in descending order
        recommendation_list = sorted(recommendation_list, key=lambda x: x['confidence'], reverse=True)
        print('sorted recommendation list:', recommendation_list)

        recommendations = []
        recommendations_per_category = {}

        for recommendation in recommendation_list:
            if recommendation in recommendations:
                continue

            # Limit 2 recommendations per category
            product_category = recommendation['product_category']
            if product_category not in recommendations_per_category:
                recommendations_per_category[product_category] = 0

            if recommendations_per_category[product_category] >= 2:
                continue
            
            recommendations_per_category[product_category] += 1

            # Add the recommendation to the list
            recommendations.append(recommendation['product'])
            
            if len(recommendations) >= top_k:
                break

        return recommendations

    def get_popular_recommendations(self, product_categories=None, top_k=5):
        recommendation_df = self.popularity_recommendations

        print('product_categories:', product_categories)
        print('type(product_categories):', type(product_categories))

        if type(product_categories) == str:
            product_categories = [product_categories]

        if product_categories is not None:
            recommendation_df = self.popularity_recommendations[self.popularity_recommendations['product_category'].isin(product_categories)]

        print('recommendation_df:', recommendation_df)

        # sort by number of transactions (most popular at the top)
        recommendation_df = recommendation_df.sort_values('number_of_transactions', ascending=False)
        print('sorted recommendation_df:', recommendation_df)

        if recommendation_df.shape[0] == 0:
            return []

        recommendation = recommendation_df['product'].tolist()[:top_k]

        return recommendation
    
    def recommendation_classification(self, message):

        system_prompt = """ 
            You are a JSON-only API. Always output a single valid JSON object.
            Your ONLY TASK: Determine the type of recommendation based on user's message. 

            CRITICAL: Check user message very carefully to find if it mentions any ONE or MULTIPLE of the following categories:
            Coffees, Bakery (includes scones, croissants and biscotti), Flavours (includes syrups), Chocolates

            We have 3 types of recommendations:
            
            1. Popular: If the user does NOT mention any category, choose 'popular'.
            2. Popular by Category: If the user mentions a category (or related to a category), choose 'popular by category'.
            3. Apriori: If user already has an order, choose 'apriori'.

            Rules for output:
            - You MUST respond in **valid JSON only**.
            - No extra text, no explanations outside the JSON.
            - Use **exact strings** from the provided item and category lists.
            - The JSON must always include all three keys: "chain_of_thought", "recommendation_type", and "parameters".

            STRICTLY follow the JSON format:
            {
                "chain_of_thought": "Brief reasoning for your choice.",
                "recommendation_type": "apriori" | "popular" | "popular by category",
                "parameters": []  
                    // for 'popular': leave empty, for 'popular by category': specified categories [MUST be from: Bakery, Coffee, Flavours, Chocolate], for 'apriori': items
            }
            
            Here is the list of items in the coffee shop:
            """+ ",".join(self.products) + """
            Here is the list of Categories we have in the coffee shop:
            """ + ",".join(self.product_categories) + """

        """

        input_messages = [{"role": "system", "content": system_prompt}] + message
        # print('input messages (rec classification):', input_messages)

        chatbot_response = get_chatbot_response(self.client,self.model_name,input_messages)
        print('chatbot response (rec classification):', chatbot_response)

        chatbot_response = double_check_json_output(self.client,self.model_name,chatbot_response)
        print('chatbot response after double check (rec classification):', chatbot_response)

        output = self.postprocess_classfication(chatbot_response)
        print('final output (classification):', output)

        return output
    
    def postprocess_classfication(self, output):

        # output = json.loads(output)
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            print("Invalid JSON from model:", output)
            return {
                "recommendation_type": "popular",   # fallback   
                "parameters": []
            }

        # dict_output = {
        #     "recommendation_type": output['recommendation_type'],
        #     "parameters": output['parameters']
        # }

        # chain_of_thought = output['chain of thought']
        # recommendation_type = output['recommendation_type']
        # parameters = output['parameters']

        # return dict_output

        recommendation_type = output.get('recommendation_type')
        parameters = output.get('parameters', [])
        chain_of_thought = output.get('chain_of_thought', None)

        if recommendation_type not in ["apriori", "popular", "popular by category"]:
            print("Unexpected or missing recommendation_type:", output)
            return {
                "chain_of_thought": chain_of_thought,
                "recommendation_type": "popular",   # fallback
                "parameters": parameters
            }

        return {
            "chain_of_thought": chain_of_thought,
            "recommendation_type": recommendation_type,
            "parameters": parameters
        }
    
    def get_recommendations_from_order(self, messages, order): 
        messages = deepcopy(messages)

        products = []
        for item in order:
            products.append(item.get('product') or item.get('item'))

        recommendations = self.get_apriori_recommendations(products)
        print('recommendations (from order):', recommendations)

        recommendations_str = ", ".join(recommendations)
        print('recommendations_str (from order):', recommendations_str)

        system_prompt = f"""
        You are a helpful AI assistant for a coffee shop application.
        your task is to recommend items to the user based on their already placed order. 
        Put it in an unordered list with very small descriptions. 
        Wrap with brief (1-2 sentences) and suitable words.
        """

        prompt = f"""
        {messages[-1]['content']}

        Please recommend these items exactly: {recommendations_str}

        IMPORTANT: 
        - Keep the response in an unordered list with a small description for each item
        - Keep the look clean and simple
        - Keep the response short but elegant
        - DO NOT use irrelevant words like "endlist" or such

        """

        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        chatbot_response = get_chatbot_response(self.client,self.model_name,input_messages)
        
        output = self.postprocess(chatbot_response)

        return output
    
    def get_response(self, messages):
        messages = deepcopy(messages)

        print('Calling Recommendation Classifier to understand user intent...')
        recommendation_classification = self.recommendation_classification(messages)
        recommendation_type = recommendation_classification['recommendation_type']
        # parameters = recommendation_classification['parameters']

        print('recommendation_classification:', recommendation_classification)
        print('recommendation_type:', recommendation_type)
        print('parameters:', recommendation_classification['parameters'])

        recommendations = []

        if recommendation_type == "apriori":
            recommendations = self.get_apriori_recommendations(recommendation_classification['parameters'])
        elif recommendation_type == "popular":
            recommendations = self.get_popular_recommendations()
        elif recommendation_type == "popular by category":
            recommendations = self.get_popular_recommendations(product_categories= recommendation_classification['parameters'])
        
        if recommendations == []:
            return {"role": "assistant", "content": "I'm sorry, I can't help with that recommendation. Can I help you with something else?"}
        
        print('recommendations (get_response):', recommendations)

        # Respond to user
        recommendation_str = ", ".join(recommendations)
        print('recommendation_str (get_response):', recommendation_str)

        system_prompt = f"""
        You are a helpful AI assistant for a coffee shop.
        your task is to recommend items to the user like a coffee shop waiter. 
        Put recommendations in an unordered list.

        IMPORTANT: DO NOT use irrelevant texts like "inquiries" or such
        """

        prompt = f"""
        {messages[-1]['content']}

        Please recommend these items exactly: {recommendation_str}        
        """

        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        print('input_messages (recommendation get_response):', input_messages)

        chatbot_response = get_chatbot_response(self.client,self.model_name,input_messages)
        print('chatbot_response (recommendation):', chatbot_response)

        output = self.postprocess(chatbot_response)
        print('post-processed output (recommendation):', output)

        # return chatbot_response
        return output
    
    def postprocess(self, output):
        output = { 
            "role": "assistant", 
            "content": output, 
            "memory": 
                {"agent": "recommendation_agent"} 
            } 
        
        return output

        # try:
        #     data = json.loads(output)
        #     return {
        #         "role": "assistant",
        #         "content": data,  # structured, clean
        #         "memory": {"agent": "recommendation_agent"}
        #     }
        # except json.JSONDecodeError:
        #     return {
        #         "role": "assistant",
        #         "content": {"recommendations": []},
        #         "memory": {"agent": "recommendation_agent"}
        #     }