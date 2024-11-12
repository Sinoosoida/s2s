# File: create_embeddings.py
import os
import pandas as pd
import json
import numpy as np
import chromadb
from openai import OpenAI
from pyprojroot import here
from dotenv import load_dotenv
import httpx
import logging
from chromadb.config import Settings
from chromadb import Client

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OpenAI API key must be provided or set in the OPENAI_API_KEY environment variable.")

http_client = httpx.Client()
client = OpenAI(api_key=api_key, http_client=http_client)

# Инициализация клиента ChromaDB с указанием пути для хранения коллекций
chroma_db_path = here('data/chroma_collections')
if not os.path.exists(chroma_db_path):
    os.makedirs(chroma_db_path)

# Load data from CSV files
preprocessed_data = pd.read_excel(here('data/for_upload/preprocessed_3mo_personal.xlsx'))
price_list = pd.read_excel(here('data/for_upload/price_list_person.xlsx'))

# # Настройки для инициализации клиента ChromaDB
# settings = Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory=str(chroma_db_path)
# )

# Инициализация клиента ChromaDB
client_chromadb = chromadb.PersistentClient(path=str(chroma_db_path))

# Create collections for embeddings
preprocessed_collection = client_chromadb.create_collection(name="product_embeddings_preprocessed")
price_list_collection = client_chromadb.create_collection(name="product_embeddings_price_list")

# Function to get embeddings using OpenAI model
def get_embedding(text, model="text-embedding-3-large"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Function to normalize embeddings
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

# Add embeddings to preprocessed collection
for idx, row in preprocessed_data.iterrows():
    row_data = row.to_dict()
    text_representation = " ".join([str(value) for key, value in row_data.items() if pd.notna(value)])
    embedding = get_embedding(text_representation)
    normalized_embedding = normalize_embedding(embedding).tolist()
    preprocessed_collection.add(embeddings=[normalized_embedding], documents=[json.dumps(row_data, ensure_ascii=False)], ids=[str(idx)])

# Add embeddings to price list collection
for idx, row in price_list.iterrows():
    row_data = row.to_dict()
    text_representation = " ".join([str(value) for key, value in row_data.items() if pd.notna(value)])
    embedding = get_embedding(text_representation)
    normalized_embedding = normalize_embedding(embedding).tolist()
    price_list_collection.add(embeddings=[normalized_embedding], documents=[json.dumps(row_data, ensure_ascii=False)], ids=[str(idx)])

logger.info("ChromaDB collections for preprocessed data and price list have been created and populated with embeddings.")