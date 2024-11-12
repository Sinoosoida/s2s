# File: open_api_model_server_handler.py
import logging
import time
import httpx
import os
from nltk import sent_tokenize
from rich.console import Console
from openai import OpenAI
from baseHandler import BaseHandler
from utils.constants import end_of_data
from utils.data import ImmutableDataChain
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import json
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from pyprojroot import here
from dotenv import load_dotenv
import numpy as np
from langchain import LLMChain, PromptTemplate

logger = logging.getLogger(__name__)
console = Console()

class OpenApiModelServerHandler(BaseHandler):
    """
    Handles the language model part, integrated with a Q&A system.
    """
    def setup(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key must be provided or set in the OPENAI_API_KEY environment variable.")

        self.http_client = httpx.Client()
        self.client = OpenAI(api_key=api_key, http_client=self.http_client)
        
        # Setup Q&A system components
        load_dotenv()
        model_name = os.getenv("gpt_deployment_name")
        proxy_url = "http://RGHu6U:WP6Z4s@168.80.200.166:8000"

        # Initialize ChatOpenAI with proxy
        self.qna_model = ChatOpenAI(
            api_key=api_key,
            http_client=httpx.Client(proxy=proxy_url),
            model_name=model_name,
            temperature=0.0
        )

        # Load data for Q&A system
        self.preprocessed_data = pd.read_excel(here('data/for_upload/preprocessed_3mo_personal.xlsx'))
        self.price_list = pd.read_excel(here('data/for_upload/price_list_person.xlsx'))
        self.jargon_data = pd.read_excel(here('data/for_upload/jargons.xls'))

        # Ensure that column names are correct
        self.jargon_data.columns = ['Жаргонизмы', 'Слово_значение', 'Примечание']
        self.jargon_terms = dict(zip(self.jargon_data['Жаргонизмы'], self.jargon_data['Слово_значение']))
        self.jargon_text = "\n".join([f"- {term}: {definition}" for term, definition in self.jargon_terms.items()])

        chroma_db_path = here('data/chroma_collections')
        client_chromadb = chromadb.PersistentClient(path=str(chroma_db_path))

        self.preprocessed_collection = client_chromadb.get_collection(name="product_embeddings_preprocessed")
        self.price_list_collection = client_chromadb.get_collection(name="product_embeddings_price_list")

        # Initialize tools and create ReAct agent for Q&A system
        self.tools = self.create_tools()
        self.qna_agent = create_react_agent(self.qna_model, tools=self.tools)

        self.system_prompt = {
            "role": "system",
            "content": """
            You are an intelligent assistant helping users find information about products in a database.
            When a user provides a product-related query, follow these rules:
            1. After initially getting a user's query about a particular product, use the 'extract_parameters' tool to normalize the query.
            2. Perform a semantic search (RAG) first in the preprocessed data collection (`preprocessed_collection`).
            3. If the number of matching results after using 'rag_search_products' tool is zero in `preprocessed_collection`, switch to searching in the price list collection (`price_list_collection`).
            4. If the number of matching results after using 'rag_search_products' tool is zero in `price_list_collection`, inform the user that the product is not available.
            5. If the number of matching results after using 'rag_search_products' tool is greater than five, ask the user for clarification on one or two parameters that are currently null to refine your search. You do this directly without using any tool.
            6. After the user provides additional information following your request to clarify one or two parameters, use the 'extract_parameters' tool again to normalize the user's response.
            7. Use the 'merge_params' tool to combine the original parameters with the new parameters obtained from the user's clarification.
            8. With the updated parameters, repeat the search in the same document collection where the initial search was conducted, refining the results.
            9. If there are matches between one and five after using 'rag_search_products' tool, use the 'get_prices' tool and provide the results to the user.
            10. Assist the user in adding selected products to the cart and calculating the total price.
            """
        }

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        response = self.client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello  "},
            ],
            stream=False
        )
        end = time.time()
        logger.info(f"{self.__class__.__name__}: warmed up! time: {(end - start):.3f} s")

    def process(self, data: ImmutableDataChain):
        logger.debug("call api language model...")

        chat_history = [self.system_prompt]
        user_input = data.get("messages")[-1]['content']
        logger.debug(f"User input: {user_input}")

        # Add user message to chat history
        chat_history.append({"role": "user", "content": user_input})
        
        # Invoke the Q&A agent
        inputs = {"messages": chat_history}
        response = self.qna_agent.stream(inputs, stream_mode="values")

        # Stream response back to the client
        first_chunk = True
        first_sentence = True
        generated_text, printable_text = "", ""
        for s in response:
            reply = s["messages"][-1].content
            logger.debug(f"Agent reply: {reply}")
            generated_text += reply
            printable_text += reply
            sentences = sent_tokenize(printable_text)
            if len(sentences) > 1:
                if first_sentence:
                    logger.debug(f"First sentence received")
                    first_sentence = False
                yield data.add_data(sentences[0], "llm_sentence")
                printable_text = ""

        logger.debug(f"All chunks received")
        # Don't forget last sentence
        yield data.add_data(printable_text, "llm_sentence")
        yield data.add_data(end_of_data, "llm_sentence")

    def load_embeddings(self):
        for idx, row in self.preprocessed_data.iterrows():
            row_data = row.to_dict()
            text_representation = " ".join([str(value) for key, value in row_data.items() if pd.notna(value)])
            embedding = self.get_embedding(text_representation)
            normalized_embedding = self.normalize_embedding(embedding).tolist()
            self.preprocessed_collection.add(embeddings=[normalized_embedding], documents=[json.dumps(row_data, ensure_ascii=False)], ids=[str(idx)])

        for idx, row in self.price_list.iterrows():
            row_data = row.to_dict()
            text_representation = " ".join([str(value) for key, value in row_data.items() if pd.notna(value)])
            embedding = self.get_embedding(text_representation)
            normalized_embedding = self.normalize_embedding(embedding).tolist()
            self.price_list_collection.add(embeddings=[normalized_embedding], documents=[json.dumps(row_data, ensure_ascii=False)], ids=[str(idx)])

    def get_embedding(self, text, model="text-embedding-3-large"):
        response = self.client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding

    def normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def create_tools(self):
        from langchain_core.tools import tool

        @tool
        def extract_parameters(user_query: str, jargon_text: str) -> dict:
            """
            Extract parameters from the user query using an LLM.
            """
            prompt_template = PromptTemplate(
                input_variables=['user_input', 'jargon_text'],
                template="""
Ты опытный помощник в сфере продаж охлажденной мясной продукции, который точно и корректно извлекает значения конкретных параметров из пользовательских запросов.

**Параметры для извлечения:**

- Цена
- Животное
- Объект
- Дополнительная информация
- Количество
- Желаемое число единиц выбранного объекта
- Способ упаковки
- Термическое состояние
- Производитель

**Инструкции:**

- Проанализируй пользовательский запрос и извлеки значения указанных выше параметров.
- Если параметр не упомянут в запросе, установи его значение как null.
- Обязательно учитывай жаргонизмы при анализе.

Жаргонизмы и их определения: 
{jargon_text}

Пользовательский запрос: "{user_input}"
Извлеченные параметры:
"""
            )
            chain = prompt_template | self.qna_model

            response = chain.invoke({
                "user_input": user_query,
                "jargon_text": jargon_text
            })
            response_content = response['content'] if isinstance(response, dict) else response.content
            return json.loads(response_content)

        @tool
        def rag_search_products(params: dict, threshold: float = 1.04, collection_name: str = "preprocessed") -> dict:
            """
            Use semantic search (RAG) to find the most relevant products based on the given parameters.
            """
            if not params:
                raise ValueError("Parameters are missing for the RAG search.")

            if "Цена" in params:
                del params["Цена"]

            non_null_params = {key: value for key, value in params.items() if pd.notna(value) and value != "null"}
            if not non_null_params:
                raise ValueError("No valid parameters provided for the RAG search.")

            query_string = " ".join(non_null_params.values())
            query_embedding = self.get_embedding(query_string)
            normalized_query_embedding = self.normalize_embedding(query_embedding).tolist()

            collection = self.preprocessed_collection if collection_name == "preprocessed" else self.price_list_collection
            results = collection.query(query_embeddings=[normalized_query_embedding], n_results=10)

            filtered_results = [doc for doc, distance in zip(results["documents"][0], results["distances"][0]) if distance < threshold]
            cleaned_results = [json.loads(doc) for doc in filtered_results]

            return {"matches": cleaned_results, "count": len(cleaned_results)}

        @tool
        def merge_params(original_params: dict, new_params: dict) -> dict:
            """
            Merge the original parameters with new parameters.
            """
            merged_params = original_params.copy()
            merged_params.update({k: v for k, v in new_params.items() if v is not None})
            return merged_params

        @tool
        def get_prices(codes: list) -> dict:
            """
            Get prices for the given product codes from the price list.
            """
            if not codes:
                raise ValueError("No product codes provided.")

            filtered_prices = self.price_list[self.price_list['КодНоменклатуры'].isin(codes)]
            if filtered_prices.empty:
                raise ValueError("No prices found for provided codes.")

            return dict(zip(filtered_prices['КодНоменклатуры'], filtered_prices['Price']))

        @tool
        def calculate_total_price(cart: list) -> float:
            """
            Calculate the total price for the products in the cart.
            """
            total_price = 0.0

            for item in cart:
                code = item['НоменклатураКод']
                quantity = item.get('Количество', 1)

                if code in self.price_list['КодНоменклатуры'].values:
                    price = self.price_list[self.price_list['КодНоменклатуры'] == code]['Price'].values[0]
                    total_price += price * quantity

            return total_price

        @tool
        def add_to_cart(products: list, cart: list) -> list:
            """
            Add selected products to the cart.
            """
            updated_cart = cart.copy()

            for product in products:
                code = product.get('НоменклатураКод')
                quantity = product.get('Количество', 1)

                existing_item = next((item for item in updated_cart if item['НоменклатураКод'] == code), None)
                if existing_item:
                    existing_item['Количество'] += quantity
                else:
                    updated_cart.append({
                        "НоменклатураКод": code,
                        "Количество": quantity
                    })

            return updated_cart

        return [extract_parameters, rag_search_products, merge_params, get_prices, calculate_total_price, add_to_cart]