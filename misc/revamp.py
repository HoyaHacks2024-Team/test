from llama_index.llms import AzureOpenAI
from llama_index.embeddings import VoyageEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index import set_global_service_context
import logging
import sys
import os

from dotenv import load_dotenv
load_dotenv()

import pymongo
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.storage.storage_context import StorageContext


# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# Configure Llama Index with Azure OpenAI
llm = AzureOpenAI(
    model=os.getenv("OPENAI_MODEL_COMPLETION"),
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_COMPLETION"),
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

# Configure Voyage Embedding
embed_model = VoyageEmbedding(model_name=os.getenv("VOYAGE_MODEL_NAME"), voyage_api_key=os.getenv("VOYAGE_API_KEY"))

# Set up service context
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
set_global_service_context(service_context)

# Load documents from the data directory
documents = SimpleDirectoryReader("data/paul_graham").load_data()


# MAIN AREA
mongodb_client = pymongo.MongoClient(os.getenv("CONNECTION_STRING"))

store = MongoDBAtlasVectorSearch(
    mongodb_client=mongodb_client,
    db_name=os.getenv("DB_NAME"),
    collection_name="paul_graham_essay",
)
storage_context = StorageContext.from_defaults(vector_store=store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)




# MAIN AREA END

# Define a query
query = "What did the author love working on?"

# Perform the query
query_engine = index.as_query_engine()
answer = query_engine.query(query)

# Print the results
print(answer.get_formatted_sources())
print("Query was:", query)
print("Answer was:", answer)