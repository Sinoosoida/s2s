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
from langchain_core.tools import tool


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

        # proxy_url = "http://RGHu6U:WP6Z4s@168.80.200.166:8000"
        # proxies = {
        #     "http://": proxy_url,
        #     "https://": proxy_url,
        # }

        self.http_client = httpx.Client()
        self.client = OpenAI(api_key=api_key, http_client=self.http_client)
        
        # Setup Q&A system components
        load_dotenv()
        model_name = os.getenv("gpt_deployment_name")
        # Initialize ChatOpenAI with proxy
        self.qna_model = ChatOpenAI(
            api_key=api_key,
            # http_client=httpx.Client(proxy=proxy_url),
            http_client=self.http_client,
            model_name=model_name,
            temperature=0.0
        )

        # logger.info("DEBBBBBB: start INIT")

        # Load data for Q&A system
        self.preprocessed_data = pd.read_excel(here('data/for_upload/preprocessed_3mo_personal.xlsx'))
        self.price_list = pd.read_excel(here('data/for_upload/preprocessed_price_list.xlsx'))
        self.jargon_data = pd.read_excel(here('data/for_upload/jargons.xls'))

        # Ensure that column names are correct
        self.jargon_data.columns = ['Жаргонизмы', 'Слово_значение', 'Примечание']
        self.jargon_terms = dict(zip(self.jargon_data['Жаргонизмы'], self.jargon_data['Слово_значение']))
        self.jargon_text = "\n".join([f"- {term}: {definition}" for term, definition in self.jargon_terms.items()])

        chroma_db_path = here('data/chroma_collections')
        self.client_chromadb = chromadb.PersistentClient(path=str(chroma_db_path))

        self.preprocessed_collection = client_chromadb.get_collection(name="product_embeddings_preprocessed")
        self.price_list_collection = client_chromadb.get_collection(name="product_embeddings_price_list")

        # logger.info("DEBBBBBB: before agent's init")

        # Initialize tools and create ReAct agent for Q&A system
        self.tools = self.create_tools()
        self.qna_agent = create_react_agent(self.qna_model, tools=self.tools)

        self.system_message = {
            "role": "system",
            "content": """
        You are an intelligent assistant helping users find information about products in a database. You always communicate in Russian.

        1. Your main goal is to tell the user about the products with their prices found in the database at the user's request that the user needs.
        2. Refine the user's request in case you have too many suitable items in between.
        3. Inform about replenishment, creation, filling of the user's order cart.

        Mandatory rules for using tools to place an order to a user:
        1. When a user makes an initial request for a product of interest, always use the 'extract_parameters' tool to normalize the request.
        2. ALWAYS use the exact output - dict "params" from the 'extract_parameters' tool without modification when sending it to the 'rag_search_products' tool. This ensures consistency and prevents validation errors. Don't change **None** from the values of output dict from 'extract_parameters' tool to **null**.  
        3. When a user makes an initial query about products of interest, the initial semantic search with the 'rag_search_products' tool is performed on 'personal_collection'.
        4. If the number of matching results from the 'rag_search_products' tool is 0 in 'personal_collection' or 'temporary_collection', initiate a search with the 'rag_search_products' tool in 'price_list_collection'.
        5. If the number of matching results from the 'rag_search_products' tool is 0 in the 'price_list_collection', inform the user that the product is out of stock.
        6. If the number of matching results from the 'rag_search_products' tool is between 1 and 5 in any collection where you are searching, use the 'get_prices' tool. Once you have prices for the items found that match the user's query, offer them all to the user.
        7. If the number of matching results from the 'rag_search_products' tool is greater than 5 in any collection where you search, ask the user to specify 1 or 2 parameters that are null in the normalized form for the product of interest.
        8. After the user has refined some parameters, use the 'extract_parameters' tool to normalize the query again.
        9. After the user has refined some parameters and you have normalized them, initiate the 'merge_params' tool to combine the current non-null parameters with the ones the user has just refined.
        10. With the updated parameters after refinement, initiate the 'rag_search_products' tool with the 'temporary_collection' string as the second argument.
        11. Use the exact output from the 'merge_params' tool without modification when sending it to the 'rag_search_products' tool. This ensures consistency and prevents validation errors.
        12. Using 'add_to_cart' and 'calculate_total_price', based on the user's desires, assemble their shopping cart and give the order total.
        13. Once the user has added everything they want to the cart, use the 'determine_delivery_days' tool to determine the delivery day for each product in the user's cart based on their region and the current time of interaction.
        """
        }

        # logger.info("DEBBBBBB: after agent's init before warmup")
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        # logger.info("DEBBBBBB: starting warmup")

        start = time.time()
        response = self.client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello  "},
            ],
            stream=False
        )
        # logger.info("DEBBBBBB: ending warmup")

        end = time.time()
        
        logger.info(f"{self.__class__.__name__}: warmed up! time: {(end - start):.3f} s")

    def process(self, data: ImmutableDataChain):
        logger.debug("call api language model...")
        sticky_system_prompt = self.system_prompt

        chat_history = [self.system_prompt]
        user_input = data.get("messages")[-1]['content']
        logger.debug(f"User input: {user_input}")

        # Add user message to chat history
        chat_history.append({"role": "user", "content": user_input})
        chat_history.insert(0, sticky_system_prompt)

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

    def get_embedding(self, text, model="text-embedding-3-large"):
        response = self.client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding

    def normalize_embedding(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def create_tools(self):
        # Tool to extract parameters from user query
        @tool
        def extract_parameters(user_query: str, jargon_text: str) -> dict:
            """
            Extract parameters from the user query using an LLM.
            Args:
                user_query (str): The query provided by the user.
                jargon_text (str): A list of jargon terms to help extract parameters.
            Returns:
                dict: A dictionary containing the extracted parameters.
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
        - Уменьшительно ласкательные приводи в нормальную форму. Множественное число тоже приводи в нормальную форму.
        - Параметр "Животное" отвечает за то, из чего сделана продукция, не путай с "Объект".
        - Параметр "Объект" отвечает за наименование продукции, не путай с "Животное", не указывай животное в этом параметре.
        - Параметр "Количество" отвечает за число грамм или килограмм в одной единице продукции.
        - Выведи результат в формате корректного JSON, используя названия параметров в качестве ключей и извлеченные значения в качестве значений.
        - В выводе результата не используй никаких дополнительных маркеров, таких как ```json```

        Жаргонизмы и их определения: 
        {jargon_text}

        **Примеры:**

        Пользовательский запрос: "Почём у вас охлажденная вакуумная тушка бройлера?"
        Извлеченные параметры:
        "Цена": null,
        "Животное": "Курица",
        "Объект": "Тушка ЦБ",
        "Дополнительная информация": null,
        "Количество": null,
        "Желаемое число единиц выбранного объекта": null,
        "Способ упаковки": "В/У",
        "Термическое состояние": "охл",
        "Производитель": null

        Пользовательский запрос: "Хочу пару коробов филейки свиной"
        Извлеченные параметры:
        "Цена": null,
        "Животное": "Свинья",
        "Объект": "Вырезка свиная",
        "Дополнительная информация": null,
        "Количество": null,
        "Желаемое число единиц выбранного объекта": "2",
        "Способ упаковки": "пак",
        "Термическое состояние": null,
        "Производитель": null

        Пользовательский запрос: "Какая у вас печень замороженная есть от КПД?"
        Извлеченные параметры:
        "Цена": null,
        "Животное": "null",
        "Объект": "Печень",
        "Дополнительная информация": null,
        "Количество": "null",
        "Желаемое число единиц выбранного объекта": null,
        "Способ упаковки": null,
        "Термическое состояние": "зам",
        "Производитель": "КПД"

        **Теперь обработай следующий пользовательский запрос:**

        Пользовательский запрос: "{user_input}"
        Извлеченные параметры:
        """
        )

            chain = prompt_template | self.qna_model

            # Invoke the chain
            response = chain.invoke({
                "user_input": user_query,
                "jargon_text": jargon_text
            })

            response_content = response['content'] if isinstance(response, dict) else response.content

            # Debug output to inspect response content
            print("DEBUG: Response Content:", response_content)

            # Parse the result
            try:
                norm_input = json.loads(response_content)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON response: " + response_content)

            # Debug output to inspect parsed result
            print("DEBUG: Parsed JSON:", norm_input)

            # Convert parsed JSON to a dictionary with standard Python None values for null entries
            params = {key: (None if value == 'null' else value) for key, value in norm_input.items()}

            # Debug output to inspect cleaned result
            print("DEBUG: Cleaned Output:", params)

            return params

        @tool
        def rag_search_products(params: dict, jargon_text: str, collection_name: str = "personal_collection") -> dict:
            """
            Use semantic search (RAG) to find the most relevant products based on the given parameters.
            Args:
                params (dict): Parameters extracted from the user's query (e.g., Животное, Объект, Термическое состояние).
                jargon_text (str): A list of jargon terms to help in semantic search.
                collection_name (str): Name of the collection to search in ("personal_collection" or "price_list_collection" or "temporary_collection").
            Returns:
                dict: A dictionary containing the matching products and the count of matches.
            """

            if not params or all(value is None or value == "" for value in params.values()):
                raise ValueError("Parameters are missing or invalid for the RAG search.")
            
            print("params from extract_param:):", params)
            # Убираем параметр "Цена", если он есть, так как поиск по цене не применим для семантического поиска
            if "Цена" in params:
                del params["Цена"]
            if "Желаемое число единиц выбранного объекта" in params:
                del params["Желаемое число единиц выбранного объекта"]

            # Преобразуем параметры в JSON-строку
            json_params = json.dumps(params, ensure_ascii=False)
            query_embedding = self.get_embedding(json_params)
            normalized_query_embedding = self.normalize_embedding(query_embedding).tolist()

            print("DEBUG: norm string params(JSON):", json_params)

            # Determine which collection to search in
            if collection_name == "personal_collection":
                collection = self.client_chromadb.get_collection(name="personal_collection")
            elif collection_name == "price_list_collection":
                collection = self.client_chromadb.get_collection(name="price_list_collection")
            elif collection_name == "temporary_collection":
                # Проверяем, создана ли временная коллекция ранее
                if "temporary_collection" not in [col.name for col in self.client_chromadb.list_collections()]:
                    raise ValueError("Temporary collection does not exist. Make sure it is created before this step.")
                # Используем уже существующую временную коллекцию
                collection = self.client_chromadb.get_collection(name="temporary_collection")
            else:
                raise ValueError("Invalid collection name specified.")

            print("DEBUG: COLLECTION:", collection.name)
            
            # Search in the specified collection
            top_semantic = collection.query(query_embeddings=[normalized_query_embedding], n_results=20)
            print("DEBUG: top_semantic:", top_semantic)

            # Обработка результатов семантического поиска
            docs_from_results = top_semantic.get('documents', [[]])[0]
            print("DEBUG: docs_from_results:", docs_from_results)

            if not docs_from_results:
                return {"matches": [], "count": 0}

            non_null_params = {key: value for key, value in params.items() if pd.notna(value) and value != "null"}
            non_null_params_str = json.dumps(non_null_params, ensure_ascii=False)

            print("non null params string:", non_null_params_str)

            prompt_template = PromptTemplate(
                input_variables=['docs_from_results', 'non_null_params', 'jargon_text'],
                template="""
            **ЗАДАЧА:**

            У тебя есть список продуктов (в формате словарей) и заданные параметры для сравнения. Твоя задача — для каждого продукта определить, соответствует ли он всем заданным параметрам, учитывая смысловую близость и однокоренные слова.

            **ИНСТРУКЦИИ:**

            1. **Рассматривай каждый продукт отдельно.** Сосредоточься на сравнении только тех ключей, которые присутствуют в заданных параметрах для сравнения.

            2. **Критерии соответствия:**
            - Если значения продукта по **всем** заданным параметрам **более-менее совпадают** с заданными (учитывая смысловую близость, однокоренные слова, синонимы, разные формы слов), то для этого продукта выведи **"1"**.
            - Если хотя бы по одному из заданных параметров есть **явное критическое несоответствие**, выведи **"0"**.

            3. **Не будь придирчивым:**
            - Учитывай синонимы, однокоренные слова, разные склонения и падежи, множественное или единственное число, регистр букв.
            - Жаргонизмы и профессиональные термины должны быть учтены (см. раздел **ЖАРГОНИЗМЫ**).
            -'Бедро хребет' - это 'Бедро'. На этом примере пойми специфику.

            4. **Формат вывода:**
            - Выведи результат в виде массива строк, состоящего только из символов **"0"** и **"1"**.
            - Не добавляй никаких дополнительных символов, текста или маркеров или ```json.
            - Не добавляй никаких объяснений, каких-либо ещё слов.
            - Пример: `["1", "0", "1"]`

            **ЖАРГОНИЗМЫ:**

            {jargon_text}

            **ПРИМЕРЫ:**

            **Пример 1:**

            Продукция на рассмотрение:
            [
                '{{"НоменклатураКод": "01-00000151", "Номенклатура": "Печень ЦБ мон охл Черкизово ТД (код 1010406265)", "Животное": "Курица", "Объект": "Печень ЦБ", "Дополнительная информация": NaN, "Количество": NaN, "Способ упаковки": "мон", "Термическое состояние": "охл", "Производитель": "Черкизово ТД"}}',
                '{{"НоменклатураКод": "01-00000321", "Номенклатура": "Паштет из печени птицы \\"Казачья Ферма\\" запеченный в/у ВТД ТД", "Животное": "Птица", "Объект": "Паштет из печени", "Дополнительная информация": "запеченный", "Количество": NaN, "Способ упаковки": "В/У", "Термическое состояние": NaN, "Производитель": "ВТД ТД"}}',
                '{{"НоменклатураКод": "ЦБ-00001028", "Номенклатура": "Фарш свин-говяж  охл.ВТД ТД", "Животное": "Свинина, Говядина", "Объект": "Фарш", "Дополнительная информация": NaN, "Количество": NaN, "Способ упаковки": "НаN", "Термическое состояние": "охл", "Производитель": "ВТД ТД"}}',
            ]
            Значения параметров для сравнения:
            {{
                "Объект": "Печень"
            }}
            Результат фильтрации:
            [
                "1", "1", "0"
            ]

            **Пример 2:**

            Продукция на рассмотрение:
            [
                '{{"НоменклатураКод": "ЦБ-00000497", "Номенклатура": "Грудка с кожей Ц/Б охл. 4кг ВТД ТД", "Животное": "Курица", "Объект": "Грудка с кожей", "Дополнительная информация": NaN, "Количество": "4кг", "Способ упаковки": NaN, "Термическое состояние": "охл", "Производитель": "ВТД ТД"}}',
                '{{"НоменклатураКод": "ЦБ-00001820", "Номенклатура": "Грудка ЦБ лот охл Куриное Царство (код 1010207832)", "Животное": "Курица", "Объект": "Грудка ЦБ", "Дополнительная информация": NaN, "Количество": NaN, "Способ упаковки": "лот", "Термическое состояние": "охл", "Производитель": "Куриное Царство"}}',
                '{{"НоменклатураКод": "ЦБ-00001803", "Номенклатура": "Голень Ц/Б охл.мон.ВТД ТД Гофра", "Животное": "Курица", "Объект": "Голень Ц/Б", "Дополнительная информация": NaN, "Количество": NaN, "Способ упаковки": "мон", "Термическое состояние": "охл", "Производитель": "ВТД ТД Гофра"}}',
                '{{"НоменклатураКод": "ЦБ-00002762", "Номенклатура": "Окорочок из мяса птицы охл. \\"Домашний\\" пакет ВТД ТД", "Животное": "Курица", "Объект": "Окорочок из мяса птицы", "Дополнительная информация": "Домашний", "Количество": NaN, "Способ упаковки": "пакет", "Термическое состояние": "охл", "Производитель": "ВТД ТД"}}',
                '{{"НоменклатураКод": "ЦБ-00001809", "Номенклатура": "Окорочок с хребтом Ц/Б охл.мон.ВТД ТД Гофра", "Животное": "Курица", "Объект": "Окорочок с хребтом", "Дополнительная информация": NaN, "Количество": NaN, "Способ упаковки": "мон", "Термическое состояние": "охл", "Производитель": "ВТД ТД"}}',
                '{{"НоменклатураКод": "ЦБ-00000669", "Номенклатура": "Грудка ЦБ 5кг лот охл Благояр", "Животное": "Курица", "Объект": "Грудка ЦБ", "Дополнительная информация": NaN, "Количество": "5кг", "Способ упаковки": "лот", "Термическое состояние": "охл", "Производитель": "Благояр"}}'
            ]
            Значения параметров для сравнения:
            {{
                "Объект": "Грудка"
            }}
            Результат фильтрации:
            [
                "1", "1", "0", "0", "0", "1"
            ]

            **Пример 3:**

            Продукция на рассмотрение:
            [
                '{{"НоменклатураКод": "ЦБ-00000848", "Номенклатура": "Бедро хребет Ц/Б охл. пакет ВТД ТД", "Животное": "Курица", "Объект": "Бедро хребет Ц/Б", "Дополнительная информация": NaN, "Количество": NaN, "Способ упаковки": "пакет", "Термическое состояние": "охл", "Производитель": "ВТД ТД"}}',
                '{{"НоменклатураКод": "01-00000149", "Номенклатура": "Окорочок ЦБ без хребта мон охл Черкизово ТД (код 1010225431)", "Животное": "Курица", "Объект": "Окорочок ЦБ без хребта", "Дополнительная информация": NaN, "Количество": NaN, "Способ упаковки": "мон", "Термическое состояние": "охл", "Производитель": "Черкизово ТД"}}',
                '{{"НоменклатураКод": "ЦБ-00000668", "Номенклатура": "Голень ЦБ 5 кг лот охл Благояр", "Животное": "Курица", "Объект": "Голень ЦБ", "Дополнительная информация": NaN, "Количество": "5 кг", "Способ упаковки": "лот", "Термическое состояние": "зам", "Производитель": "Благояр"}}',
                '{{"НоменклатураКод": "01-00000143", "Номенклатура": "Бедро ЦБ без хребта мон охл Черкизово ТД (код 1010239665)", "Животное": "Курица", "Объект": "Бедро ЦБ без хребта", "Дополнительная информация": NaN, "Количество": NaN, "Способ упаковки": "мон", "Термическое состояние": "охл", "Производитель": "Черкизово ТД"}}',
                '{{"НоменклатураКод": "ЦБ-00001990", "Номенклатура": "Бедро Ц/Б охл.мон.ВТД ТД Гофра", "Животное": "Курица", "Объект": "Бедро Ц/Б", "Дополнительная информация": NaN, "Количество": NaN, "Способ упаковки": "мон", "Термическое состояние": "охл", "Производитель": "ВТД ТД"}}'
            ]
            Значения параметров для сравнения:
            {{
                "Животное": "Курица",
                "Объект": "Голень",
                "Термическое состояние": "охл"
            }}
            Результат фильтрации:
            [
                "0", "0", "0", "0", "0", "0"
            ]

            **Теперь отфильтруй следующий список:**
            {docs_from_results}

            **Значения параметров для сравнения:**
            {non_null_params}

            **Твоя оценка:**
            """
            )
            chain = prompt_template | self.qna_model

            # Вызов LLM модели для фильтрации
            try:
                response = chain.invoke({
                    "docs_from_results": docs_from_results,
                    "non_null_params": non_null_params_str,
                    "jargon_text": jargon_text
                })
            except Exception as e:
                print("DEBUG: Error invoking LLM model for filtering:", e)
                return {"matches": [], "count": 0}
            
            # Проверяем ответ от LLM и преобразуем его в Python формат
            response_content = response.get('content') if isinstance(response, dict) else response.content
            response_content = response_content.strip('`')

            try:
                filter_results = json.loads(response_content)
            except json.JSONDecodeError as e:
                print("DEBUG: Error decoding LLM response:", e)
                print("DEBUG: Response content that caused the error:", response_content)
                return {"matches": [], "count": 0}
            
            print("answer from chain:", response_content)
            final_filtered_results = [doc for doc, flag in zip(docs_from_results, filter_results) if flag == "1"]

            if len(final_filtered_results) > 5:
                # Удаляем коллекцию, если она уже существует, чтобы избежать ошибки
                if "temporary_collection" in [col.name for col in self.client_chromadb.list_collections()]:
                    try:
                        self.client_chromadb.delete_collection(name="temporary_collection")
                        print("DEBUG: Temporary collection deleted.")
                    except Exception as e:
                        print("DEBUG: Error deleting existing temporary collection:", e)
                        return {"matches": [], "count": 0}
                
                try:
                    # Создаем новую временную коллекцию
                    temporary_collection = self.client_chromadb.create_collection(name="temporary_collection")
                    print("DEBUG: Temporary collection created.")

                    for idx, doc in enumerate(final_filtered_results):
                        doc_embedding = self.get_embedding(doc)
                        normalized_doc_embedding = self.normalize_embedding(doc_embedding).tolist()
                        temporary_collection.add(
                            embeddings=[normalized_doc_embedding],
                            documents=[doc],
                            ids=[str(idx)]
                        )
                    print("DEBUG: Documents added to temporary collection.")
                except Exception as e:
                    print("DEBUG: Error creating or populating temporary collection:", e)

            # Преобразуем строки JSON в объекты Python, чтобы избежать проблем с кодировкой Unicode
            cleaned_results = []
            for doc in final_filtered_results:
                try:
                    cleaned_results.append(json.loads(doc))
                except json.JSONDecodeError as e:
                    print("DEBUG: Error decoding document:", e)
                    print("DEBUG: Document that caused the error:", doc)

            print("DEBUG: Cleaned results:", cleaned_results, " with len:", len(final_filtered_results))

            return {"matches": cleaned_results, "count": len(final_filtered_results)}


        @tool
        def merge_params(original_params: dict, new_params: dict) -> dict:
            """
            Merge the original parameters with new parameters.
            Args:
                original_params (dict): The original parameters from the initial query.
                new_params (dict): The new parameters obtained after user clarification.
            Returns:
                dict: Merged parameters with updated values.
            """
            merged_params = original_params.copy()
            # merged_params.update({k: v for k, v in new_params.items() if v != "null"})

            # Если новый параметр не None, обновляем его
            for key, value in new_params.items():
                if value is not None:
                    merged_params[key] = value

            print("DEBUG: Merged parameters:", merged_params)
            return merged_params


        @tool
        def get_prices(codes: list) -> dict:
            """
            Get prices for the given product codes from the price list.
            Args:
                codes (list): List of product codes.
            Returns:
                dict: A dictionary with product codes as keys and their prices as values.
            """
            if not codes:
                raise ValueError("No product codes provided.")
            print("GEEET PRICIESSSSSSSSSSS")
            # Debug output to see codes being processed
            print("DEBUG: Codes received for price lookup:", codes)

            # Filter prices for provided codes
            filtered_prices = self.price_list[self.price_list['КодНоменклатуры'].isin(codes)]
            
            if filtered_prices.empty:
                raise ValueError("No prices found for provided codes.")

            # Debug output to inspect found prices
            print("DEBUG: Prices found:", filtered_prices)

            return dict(zip(filtered_prices['КодНоменклатуры'], filtered_prices['Price']))

        @tool
        def calculate_total_price(cart: list) -> float:
            """
            Calculate the total price for the products in the cart.
            Args:
                cart (list): List of products in the cart.
            Returns:
                float: The total price for the products.
            """
            total_price = 0.0

            for item in cart:
                # Проверка, что ключ 'НоменклатураКод' присутствует
                if 'НоменклатураКод' not in item:
                    print("DEBUG: Продукт без 'НоменклатураКод'", item)
                    continue

                code = item['НоменклатураКод']
                quantity = item.get('Количество', 1)

                # Проверка, что код присутствует в прайс-листе
                if code in self.price_list['КодНоменклатуры'].values:
                    price = self.price_list[self.price_list['КодНоменклатуры'] == code]['Price'].values[0]
                    total_price += price * quantity
                else:
                    print(f"DEBUG: Код {code} не найден в прайс-листе")

            # Debug: вывод итоговой суммы
            print("DEBUG: Итоговая сумма:", total_price)

            return total_price

        @tool
        def add_to_cart(products: list, cart: list) -> list:
            """
            Add selected products to the cart.
            Args:
                products (list): List of selected products with their codes and quantities.
                cart (list): The current cart to add products to.
            Returns:
                list: The updated cart.
            """
            updated_cart = cart.copy()

            for product in products:
                # Получаем код продукта из 'code' или 'НоменклатураКод'
                code = product.get('НоменклатураКод') or product.get('code')
                if not code:
                    print("DEBUG: Продукт не содержит корректный ключ для кода", product)
                    continue

                quantity = product.get('Количество', product.get('quantity', 1))
                if not isinstance(quantity, int) or quantity <= 0:
                    print("DEBUG: Неправильное количество для продукта", product)
                    quantity = 1

                # Check if the product already exists in the cart
                existing_item = next((item for item in updated_cart if item['НоменклатураКод'] == code), None)
                if existing_item:
                    # Update the quantity if the product is already in the cart
                    existing_item['Количество'] += quantity
                else:
                    # Add new product to the cart
                    updated_cart.append({
                        "НоменклатураКод": code,
                        "Количество": quantity
                    })

            # Debug: вывод обновленной корзины
            print("DEBUG: Обновленная корзина:", updated_cart)

            return updated_cart

        from datetime import datetime

        @tool
        def determine_delivery_days(user_cart: list,
                                    region: str = 'Главный',
                                    current_datetime=datetime(2024, 11, 15, 10, 25),
                                    nomen_file: str = '/home/alex/Desktop/wrk/Advanced-QA-and-RAG-Series/Q&A-and-RAG-with-SQL-and-TabularData/data/for_upload/nomen.xlsx',
                                    manufacturer_dir: str = '/home/alex/Desktop/wrk/Advanced-QA-and-RAG-Series/Q&A-and-RAG-with-SQL-and-TabularData/data/for_upload/manufacturer',
                                    personal_schedule_file: str = '/home/alex/Desktop/wrk/Advanced-QA-and-RAG-Series/Q&A-and-RAG-with-SQL-and-TabularData/data/for_upload/personal_schedule.xls') -> dict:
            """
            Determine the delivery day for products in the user's cart based on the region and the current date and time.
            Args:
                user_cart (list): A list of products in the user's cart.
                region (str): The user's region.
                current_datetime: The current datetime of user interaction.
                nomen_file (str): Path to the file containing product nomenclature and manufacturers.
                manufacturer_dir (str): Directory containing manufacturer schedules.
                personal_schedule_file (str): Path to the file containing the user's personal delivery schedule.
            Returns:
                dict: A dictionary containing delivery days for each product in the user's cart.
            """
            import pandas as pd
            from datetime import timedelta
            import os

            print("DEBUG: Starting determine_delivery_days function.")

            # Load nomenclature data
            print(f"DEBUG: Loading nomenclature data from {nomen_file}.")
            nomen_df = pd.read_excel(nomen_file)
            delivery_schedule = {}

            # Iterate over each product in the user's cart
            for item in user_cart:
                product_code = item['НоменклатураКод']
                print(f"DEBUG: Processing product with code {product_code}.")

                # Find the manufacturer for the product using Код
                product_row = nomen_df.loc[nomen_df['Код'] == product_code]
                if product_row.empty:
                    raise ValueError(f"Product code {product_code} not found in nomenclature file {nomen_file}")
                manufacturer = ' '.join(product_row['Производитель'].values[0].split())
                product_name = product_row['Наименование'].values[0]
                print(f"DEBUG: Found manufacturer {manufacturer} for product {product_name}.")

                # Locate the corresponding manufacturer's folder
                manufacturer_path = os.path.join(manufacturer_dir, ' '.join(manufacturer.split()))
                
                print(f"DEBUG: Looking for manufacturer folder at {manufacturer_path}.")
                if not os.path.exists(manufacturer_path):
                    raise ValueError(f"Manufacturer folder not found: {manufacturer_path}")

                # Select the appropriate xls file based on product type using LLM
                product_files = [file_name for file_name in os.listdir(manufacturer_path) if file_name.endswith('.xls')]
                if not product_files:
                    raise ValueError(f"No suitable product files found in {manufacturer_path}")
                print(f"DEBUG: Found product files: {product_files}.")

                # Use LLM to choose the most relevant file
                prompt_template = PromptTemplate(
                    input_variables=['product_name', 'file_names'],
                    template="""
                Выберите наиболее подходящий файл из списка файлов производителя для данного продукта.

                Продукт: {product_name}
                Список файлов производителя: {file_names}

                Выведи только название файла, которое наиболее подходит данному продукту, без каких-либо пояснений или дополнительных слов.
                """

                )
                chain = prompt_template | self.qna_model
                response = chain.invoke({
                    "product_name": product_name,
                    "file_names": ', '.join(product_files)
                })
                # Изменение строки для извлечения имени файла продукта и нормализация имени файла
                product_file_name = response.content.strip()
                product_file_name = ' '.join(product_file_name.split())  # Убираем лишние пробелы
                product_file_name = product_file_name.replace('\xa0', ' ')  # Убираем неразрывные пробелы

                # Вывод всех файлов в директории для отладки
                all_files = os.listdir(manufacturer_path)
                print(f"DEBUG: Files in directory {manufacturer_path}: {all_files}")

                # Нормализация всех имен файлов в директории
                normalized_files = [' '.join(f.split()).replace('\xa0', ' ') for f in all_files]

                # Поиск файла с нормализованным именем
                if product_file_name in normalized_files:
                    original_index = normalized_files.index(product_file_name)
                    product_file = os.path.join(manufacturer_path, all_files[original_index])
                    print(f"DEBUG: Selected product file after normalization: {all_files[original_index]}")
                else:
                    raise ValueError(f"Selected product file '{product_file_name}' not found in {manufacturer_path}. Available files: {all_files}")

                # Проверяем наличие файла
                if not os.path.exists(product_file):
                    raise ValueError(f"Selected product file '{product_file_name}' not found in {manufacturer_path}. Available files: {all_files}")

                # Load the product file
                print(f"DEBUG: Loading product file from {product_file}.")
                product_df = pd.read_excel(product_file)

                # Determine the current day and time
                current_day = current_datetime.strftime('%A').replace('Monday', 'Понедельник').replace('Tuesday', 'Вторник').replace('Wednesday', 'Среда').replace('Thursday', 'Четверг').replace('Friday', 'Пятница').replace('Saturday', 'Суббота').replace('Sunday', 'Воскресенье')
                current_time = current_datetime.time()
                print(f"DEBUG: Current day is {current_day}, current time is {current_time}.")

                # Find the delivery schedule for the region
                delivery_time = None
                next_day = current_day
                for _ in range(7):
                    time_column = f"{next_day} – время"
                    client_column = f"{next_day} – клиенту"
                    print(f"DEBUG: Checking delivery time for region {region} on {next_day}.")

                    # Check if the region has a delivery slot for the current day
                    region_row = product_df.loc[product_df['Регион'] == region]
                    if region_row.empty:
                        raise ValueError(f"Region {region} not found in product file {product_file}")

                    time_value = region_row[time_column].values[0]
                    if pd.notna(time_value) and time_value > current_time:
                        delivery_day = region_row[client_column].values[0]
                        if pd.notna(delivery_day):
                            delivery_time = delivery_day
                            print(f"DEBUG: Found delivery time on {next_day}: {delivery_time}.")
                            break
                    else:
                        # Move to the next day
                        next_day = (current_datetime + timedelta(days=1)).strftime('%A').replace('Monday', 'Понедельник').replace('Tuesday', 'Вторник').replace('Wednesday', 'Среда').replace('Thursday', 'Четверг').replace('Friday', 'Пятница').replace('Saturday', 'Суббота').replace('Sunday', 'Воскресенье')
                        current_datetime += timedelta(days=1)
                        print(f"DEBUG: Moving to next day: {next_day}.")

                # Load the personal schedule file
                print(f"DEBUG: Loading personal schedule file from {personal_schedule_file}.")
                schedule_df = pd.read_excel(personal_schedule_file)

                # Get the delivery day
                days = schedule_df.columns.tolist()
                delivery_found = False
                for day in days:
                    if day == next_day:
                        available = schedule_df.loc[1, day]
                        if available == 'да':
                            delivery_schedule[product_name] = next_day
                            delivery_found = True
                            print(f"DEBUG: Delivery available on {next_day} for product {product_name}.")
                            break
                    next_day = (current_datetime + timedelta(days=1)).strftime('%A')
                    current_datetime += timedelta(days=1)
                    print(f"DEBUG: Checking next available day: {next_day}.")

                if not delivery_found:
                    delivery_schedule[product_name] = f"{next_day} (ещё через неделю)"
                    print(f"DEBUG: Delivery not available this week, scheduled for {next_day} (next week) for product {product_name}.")

            print("DEBUG: Completed determine_delivery_days function.")
            return delivery_schedule


        return [extract_parameters, rag_search_products, merge_params, get_prices, add_to_cart, calculate_total_price, determine_delivery_days]