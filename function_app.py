import azure.functions as func
import logging

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.query_engine import NLSQLTableQueryEngine
import pyodbc
import urllib
from llama_index.core import SQLDatabase, ServiceContext
import pandas as pd
import openai
from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)

import os

from sqlalchemy import (
    create_engine,
    select,
)

from llama_index.core import Settings

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import logging
import sys

from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(route="sqlFunction")
def sqlFunction(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')


AZURE_OPENAI_CHATGPT_MODEL = os.getenv("AZURE_OPENAI_CHATGPT_MODEL")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_RESOURCE = os.getenv("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_TEMPERATURE = os.getenv("AZURE_OPENAI_TEMPERATURE") or "0.17"
AZURE_OPENAI_APIVERSION = os.getenv("AZURE_OPENAI_APIVERSION")
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_MODEL")
AZURE_OPENAI_EMBEDDING_APIVERSION = os.getenv("AZURE_OPENAI_EMBEDDING_APIVERSION")
AZURE_KEY_VAULT_NAME = os.getenv("AZURE_KEY_VAULT_NAME")

KVUri = f"https://{keyVaultName}.vault.azure.net"
credential = DefaultAzureCredential()
kv_client = SecretClient(vault_url=KVUri, credential=credential)

OpenEmbedingEndpoint = kv_client.get_secret("OpenEmbedingEndpoint").value
OpenEmbedingKey = kv_client.get_secret("OpenEmbedingKey").value
Open4Endpoint = kv_client.get_secret("Open4Endpoint").value
Open4key = kv_client.get_secret("Open4key").value

# params='Driver={ODBC Driver 17 for SQL Server};Server=tcp:sql-server-rvaiolttvds2o.database.windows.net,1433;Database=MJtest;UID=openvbd;PWD=Q1w2e3r4t5; Connection Timeout=30'
params = 'Driver={ODBC Driver 17 for SQL Server};' + 'Server=tcp:{ServerURL},{ServerPort};Database={DabaseName};UID={DBUsername};PWD={DBPasw}; Connection Timeout=60; Integrated Security=false'
conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
engine_azure = create_engine(conn_str)
print('connection is ok')
sql_database = SQLDatabase(engine_azure)

llm = AzureOpenAI(
    engine="gpt4",
    model="gpt-4",
    temperature=0.0,
    azure_endpoint=Open4Endpoint,
    api_key=Open4key,
    api_version="2023-07-01-preview",
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=OpenEmbedingKey,
    azure_endpoint=OpenEmbedingEndpoint,
    api_version="2023-07-01-preview",
)
Settings.llm = llm
Settings.embed_model = embed_model

tables_info = json.loads(tables_info.replace("'", '"'))
table_schema_objs = []

for table, description in tables_info.items():
    table_schema_objs.append(
        SQLTableSchema(
            table_name=table,
            context_str=description

        )

    )

table_node_mapping = SQLTableNodeMapping(sql_database)

obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex
)

query_engine = SQLTableRetrieverQueryEngine(
    sql_database, obj_index.as_retriever(similarity_top_k=3)
)

response = query_engine.query("cual es la abrebiacion de Loma la lata")
resp = {"answerNLP": response.response, "sqlQuery": response.metadata.get("sql_query")}

return func.HttpResponse(json.dumps(resp), mimetype="application/json")
