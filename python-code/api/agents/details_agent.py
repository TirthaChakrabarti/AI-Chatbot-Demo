from copy import deepcopy
import os

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from .utils import get_chatbot_response

load_dotenv()

class DetailsAgent:
    def __init__(self):
        # Local LLM client
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        self.model_name = "phi3"

        # Embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Pinecone setup
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "ns1")

    def get_closest_results(self, input_embeddings, top_k=2):
        """Query Pinecone index for the closest matching documents."""
        index = self.pc.Index(self.index_name)
        results = index.query(
            namespace=self.namespace,
            vector=input_embeddings,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        return results

    def get_response(self, messages):
        """Generate a chatbot response using retrieved Pinecone knowledge and the local Ollama LLM."""
        print("Geneting a response using retrieved Pinecone knowledge...")

        messages = deepcopy(messages)
        user_message = messages[-1]["content"]

        # Create embeddings
        embeddings = self.embedding_model.encode(user_message).tolist()
        print('embeddings:', embeddings)

        # Retrieve similar docs
        result = self.get_closest_results(embeddings)
        print('result (details_agent):', result)

        source_knowledge = "\n".join(
            [doc['metadata']['text'].strip() for doc in result['matches']]
        )

        print('source_knowledge (details_agent):', source_knowledge)

        # Construct RAG prompt
        prompt = f"""
        Using the contexts below, answer the query as a coffee shop waiter.
        Answers should be concise but complete.

        Contexts:
        {source_knowledge}

        Query:
        {user_message}
        """

        # System role
        system_prompt = """
        RULES:
        - You are "AIndrilla"- an artificially intelligent friendly customer support agent for a coffee shop called 'Marry's Way'.
        - You answer queries, recommend food items, provide information about the coffee shop, take orders etc.
        - For generic greetings, thankings and farewell, respond very briefly (2-3 elegant sentences) that elevates the interest of the user or end the conversation in a nice way. 
        - Do not provide too much information beyond the context (prices etc.) unless asked.
        - Do not include irrelevant words, text or paragraphs
        - Answers should be concise but complete.

        When providing a list of items: 
        - Keep the response in an unordered list with very short description for each item
        - Keep the look clean and simple
        - Keep the response short but elegant
        - DO NOT use irrelevant words like "endlist", `end of list` or undesirable signs like [] etc.
        """

        # Inject new content into conversation
        messages[-1]['content'] = prompt
        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]

        # print('input_messages (details_agent):', input_messages)

        # Generate output
        chatbot_output = get_chatbot_response(self.client, self.model_name, input_messages)
        print('chatbot_output (details_agent):', chatbot_output)

        output = self.postprocess(chatbot_output)
        print('processed output (details_agent):', output)

        return output

    def postprocess(self, output):
        """Format output in the required structure."""
        return {
            "role": "assistant",
            "content": output,
            "memory": {"agent": "details_agent"}
        }
