from langchain.document_loaders import PyPDFLoader
from chunker import chunk_documents

import yaml
import os
from openai import OpenAI
import numpy as np


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

os.environ["OPENAI_API_KEY"] = config['openai_api_key']


vector_embedding_model = config["vector_embedding_model"]
openai_client = OpenAI()


def get_embedding(text, model=vector_embedding_model):
    try:
        response = openai_client.embeddings.create(input = [text], model=vector_embedding_model).data[0].embedding
        return response
    except Exception as e:
        print(f"Error generating embedding for text: {text[:50]}... - {e}")
        return None



## threshold for embedding similarity
threshold = config['embedding_similarity_threshold']
## max number of chunks to be retrieved from top 
top_k = config['top_k_retrieved_chunks']

##cosine siilarity function
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)
    return dot_product / (norm_embedding1 * norm_embedding2)


def find_most_similar(query, df):
    # Get the embedding for the query
    query_embedding = get_embedding(query)
    
    if query_embedding is None:
        print("Could not get embedding for the query.")
        return None

    # Calculate similarity between query embedding and each embedding in the DataFrame
    df['similarity'] = df['embedding'].apply(lambda emb: cosine_similarity(query_embedding, emb))
    filtered_df = df[df['similarity'] >= threshold]
    sorted_df = filtered_df.sort_values(by='similarity', ascending=False)
    top_k_df = sorted_df.head(top_k)

    return top_k_df[['chunks', 'similarity']]
    
 
