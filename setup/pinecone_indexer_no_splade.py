import os
from dotenv import load_dotenv

load_dotenv("../.env")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_API_ENV = os.environ["PINECONE_API_ENV"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
import pickle
import pinecone
import openai
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_not_exception_type,
)
from typing import List
from uuid import uuid4
import textwrap
from tqdm.auto import tqdm  # this is our progress bar
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

openai.api_key = OPENAI_API_KEY
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = "cl100k_base"


#### Set up OpenAI Embedding process


@retry(
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(6),
    retry=retry_if_not_exception_type(openai.InvalidRequestError),
)
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    return openai.Embedding.create(input=text_or_tokens, model=model)


def chunk_text(text: str, max_chunk_size: int, overlap_size: int) -> List[str]:
    """Helper function to chunk a text into overlapping chunks of specified size."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunks.append(text[start:end])
        start += max_chunk_size - overlap_size
    return chunks


def transform_record(record: dict) -> List[dict]:
    """Transform a single record as described in the prompt."""
    max_chunk_size = 500
    overlap_size = 100
    chunks = chunk_text(record, max_chunk_size, overlap_size)
    transformed_records = []
    recordId = str(uuid4())
    for i, chunk in enumerate(chunks):
        chunk_id = f"{recordId}-{i+1}"
        transformed_records.append(
            {
                "chunk_id": chunk_id,
                "chunk_parent_id": recordId,
                "chunk_text": chunk,
                "vector": get_embedding(chunk).get("data")[0]["embedding"]
                #'sparse_values': splade(chunk)
            }
        )
    return transformed_records


#### Generate Pinecone Index
index_name = PINECONE_INDEX_NAME

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV,  # may be different, check at app.pinecone.io
)

# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric="cosine",
        metadata_config={"indexed": ["unused"]},
        pod_type="p1.x1",
    )
# connect to index
index = pinecone.Index(index_name)
# view index stats
index.describe_index_stats()


#### Prepare and load data from file
with open("data/out.txt", "r", encoding="ISO-8859-1") as f:
    file = f.read()

#### Generate embeddings and Pickle the results to not spend on OpenAI
chunked_data = []
chunk_array = transform_record(file)
for chunk in chunk_array:
    chunked_data.append(chunk)


### Save data and vectors offline
# Pickle the array
with open("data/vector_data_500_100_sparse_no_splade.pickle", "wb") as f:
    pickle.dump(chunked_data, f, protocol=pickle.HIGHEST_PROTOCOL)


### Load data from local to upsert to Pinecone
with open("data/vector_data_500_100_sparse_no_splade.pickle", "rb") as f:
    vector_data = pickle.load(f)


#### Format data to load to Pinecone
def prepare_entries_for_pinecone(entries):
    """
    Prepares an array of entries for upsert to Pinecone.
    Each entry should have a 'vector' field containing a list of floats.
    """
    vectors = []
    for entry in entries:
        vector = entry["vector"]
        id = entry.get("chunk_id", "")
        metadata = entry.get(
            "metadata",
            {
                "chunk_id": entry.get("chunk_id", ""),
                "parent_id": entry.get("chunk_parent_id", ""),
                "chunk_text": entry.get("chunk_text", ""),
            },
        )
        values = [v for v in vector]
        # sparse_values = entry['sparse_values']
        # vectors.append({'id': id, 'metadata': metadata, 'values': values, 'sparse_values': sparse_values})
        vectors.append({"id": id, "metadata": metadata, "values": values})
    return {"vectors": vectors, "namespace": ""}


vectors = prepare_entries_for_pinecone(vector_data)

#### Upsert vectors (sparse and dense) and metadata to Pinecone


batch_size = 32  # process everything in batches of 32
for i in tqdm(range(0, len(vectors["vectors"]), batch_size)):
    ids_batch = [id["id"] for id in vectors["vectors"][i : i + batch_size]]
    embeds = [id["values"] for id in vectors["vectors"][i : i + batch_size]]
    meta = [id["metadata"] for id in vectors["vectors"][i : i + batch_size]]
    # sparse_values = [id['sparse_values'] for id in vectors['vectors'][i:i+batch_size]]
    upserts = []
    # loop through the data and create dictionaries for uploading documents to pinecone index
    # for _id, sparse, dense, meta in zip(ids_batch, sparse_values, embeds, meta):
    for _id, dense, meta in zip(ids_batch, embeds, meta):
        upserts.append(
            {
                "id": _id,
                # 'sparse_values': sparse,
                "values": dense,
                "metadata": meta,
            }
        )
    # upload the documents to the new hybrid index
    index.upsert(upserts)


#### Query Pinecone and OpenAI

# LIMIT = 8000
#
#
# def retrieve(query):
#    res = openai.Embedding.create(input=[query], engine=EMBEDDING_MODEL)
#
#    # retrieve from Pinecone
#    xq = res["data"][0]["embedding"]
#    # sq = splade(query)
#
#    # get relevant contexts
#    # res = index.query(xq, top_k=5, include_metadata=True, sparse_vector=sq)
#    res = index.query(xq, top_k=5, include_metadata=True)
#    contexts = [x["metadata"]["chunk_text"] for x in res["matches"]]
#
#    # build our prompt with the retrieved contexts included
#    prompt_start = (
#        "Answer the question based on the context below. If you cannot answer based on the context or general knowledge about Wells Fargo, truthfully answer that you don't know.\n\n"
#        + "Context:\n"
#    )
#    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
#    # append contexts until hitting limit
#    for i in range(1, len(contexts)):
#        if len("\n\n---\n\n".join(contexts[:i])) >= LIMIT:
#            prompt = prompt_start + "\n\n---\n\n".join(contexts[: i - 1]) + prompt_end
#            break
#        elif i == len(contexts) - 1:
#            prompt = prompt_start + "\n\n---\n\n".join(contexts) + prompt_end
#    return prompt
#
#
# def complete(prompt):
#    # query text-davinci-003
#    res = openai.Completion.create(
#        engine="text-davinci-003",
#        prompt=prompt,
#        temperature=0,
#        max_tokens=512,
#        top_p=1,
#        frequency_penalty=0,
#        presence_penalty=0,
#        stop=None,
#    )
#    return res["choices"][0]["text"].strip()


#### Langchain Memory for conversation chat style

# llm = OpenAI(
#    temperature=0, openai_api_key=OPENAI_API_KEY, model_name="text-davinci-003"
# )
# conversation_with_summary = ConversationChain(
#    llm=llm,
#    # We set a very low max_token_limit for the purposes of testing.
#    memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=650),
# )

#### Sample query to Pinecone and OpenAI
# Can be used in a notebook
# query ="What is Wells Fargo?"
## first we retrieve relevant items from Pinecone
# query_with_contexts = retrieve(query)
# print(textwrap.fill(str(conversation_with_summary.predict(input=query_with_contexts))))
# conversation_with_summary.memory.clear()


#### Loop to ask multiple questions and get answers
# while True:
#    # Prompt user for input
#    user_input = input("Enter your input (type 'quit' to exit): ")
#
#    # Check if user wants to quit
#    if user_input.lower() == "quit":
#        print("Exiting program...")
#        break
#
#    # Process user input
#    processed_input = user_input.upper()  # Convert to all uppercase letters
#    print("Processed input: ", processed_input)
#
#    query = user_input
#
#    # first we retrieve relevant items from Pinecone
#    query_with_contexts = retrieve(query)
#
#    # then we send the context and the query to OpenAI
#    print(textwrap.fill(str(conversation_with_summary.predict(input=query_with_contexts))) + '\n')
