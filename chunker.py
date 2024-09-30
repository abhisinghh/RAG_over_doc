import pandas as pd
import numpy as np
import yaml
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

openai_api_key = config['openai_api_key']

#loader = PyPDFLoader("./content.pdf")
#documents = loader.load()
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="gradient")
##text_splitter = RecursiveCharacterTextSplitter(
#    # Set a really small chunk size, just to show.
#    chunk_size=512,
#    chunk_overlap=20,
#    length_function=len,
#    is_separator_regex=False,
#)


def chunk_documents(documents) :
  doc_chunks = text_splitter.split_documents(documents)
  doc_chunk_texts = [chunk.page_content for chunk in doc_chunks]
  df_chunks = pd.DataFrame(doc_chunk_texts)
  df_chunks.rename(columns = {0:'Chunks'}, inplace = True)
  #print(df_chunks.shape, df_chunks.head())
  return df_chunks
