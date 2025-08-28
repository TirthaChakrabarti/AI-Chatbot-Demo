from agents import (GuardAgent, 
                    ClassificationAgent, 
                    DetailsAgent,
                    RecommendationAgent,
                    OrderTakingAgent,
                    AgentProtocol
                    )
import os
from typing import Dict
import pathlib
import re

folder_path = pathlib.Path(__file__).parent.resolve()

def main():

    guard_agent = GuardAgent()
    classification_agent = ClassificationAgent()
    recommendation_agent = RecommendationAgent(
            os.path.join(folder_path, "recommendation_objects/apriori_recommendations.json"),
            os.path.join(folder_path, "recommendation_objects/popularity_recommendation.csv")
        )

    agent_dict: Dict[str, AgentProtocol] = {
        "details_agent": DetailsAgent(),
        "recommendation_agent": recommendation_agent,
        "order_taking_agent": OrderTakingAgent(recommendation_agent)
    }

    messages = []

    while True:
        # clear previous inputs
        # os.system('cls' if os.name == 'nt' else 'clear')

        print("\nStart Conversation.......")
        for message in messages:
            print(f"\n{message['role']}: {message['content']}")

        # get user input
        user_prompt = input("\nEnter your message (type 'exit' or 'quit' to end the conversation): ")

        if user_prompt.lower() in ["exit", "quit"]:
            print("\nExiting conversation. Goodbye!")
            break   # exit the while loop

        messages.append({"role": "user", "content": user_prompt})

        # get guard agent's response
        guard_agent_response = guard_agent.get_response(messages)
        print("\nGuard Agent's Response: ", guard_agent_response)

        if guard_agent_response["memory"]["guard_decision"] == "not allowed":
            messages.append(guard_agent_response)
            continue

        # get classification agent's response
        classification_agent_response = classification_agent.get_response(messages)

        if classification_agent_response["memory"]["classification_decision"] == "unsure":
            messages.append(classification_agent_response)
            continue

        chosen_agent = classification_agent_response["memory"]["classification_decision"]
        print("\nChosen Agent: ", chosen_agent)

        # get the chosen agent's response
        agent = agent_dict[chosen_agent]
        agent_response = agent.get_response(messages)
        print("\nAgent's Response: ", agent_response)

        messages.append(agent_response)
    


if __name__ == "__main__":
    main()
    
