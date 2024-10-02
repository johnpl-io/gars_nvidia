from pymilvus import MilvusClient
from tqdm import tqdm
import numpy as np
import argparse
import json
import os
from pymilvus import MilvusClient, Collection, connections
from dotenv import load_dotenv


def generate_embedding_api(client, text, model):
    """
    Generate an embedding for the given text using the specified model from OpenAI.

    Parameters:
    client (OpenAI): The OpenAI client used to generate embeddings.
    text (str): The text to generate an embedding for.
    model (str): The model to use for generating the embedding.

    Returns:
    list: The embedding vector for the given text.
    """
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def load_embedding(index, embedding_path):

    memmap_array = np.lib.format.open_memmap(embedding_path, mode="r")

    row = memmap_array[index]

    return row


def export_collection(db_client, open_ai_client, collection_name, file_name, params):
    """
    Export data from a file into a Milvus collection with generated embeddings.

    Parameters:
    db_client (MilvusClient): The Milvus client used to interact with the database.
    open_ai_client (OpenAI): The OpenAI client used to generate embeddings.
    collection_name (str): The name of the collection in Milvus.
    file_name (str): The name of the file containing data to be inserted.
    params (dict): The parameters for creating the collection and generating embeddings.
    """

    if not params["online"]:
        directory, name = os.path.split(file_name)

        new_directory = directory.replace("plain_text", "embeddings")

        base_name, _ = os.path.splitext(name)

        embedding_file_name = base_name + "_embeddings.npy"

        embedding_path = os.path.join(new_directory, embedding_file_name)

    db_client.drop_collection(collection_name)
    db_client.create_collection(
        collection_name=collection_name,
        dimension=params["embedding_size"],
        metric_type=params["metric_type"],
    )

    with open(file_name, "r") as f:
        for index, line in tqdm(enumerate(f)):
            if params["online"]:
                embedding = generate_embedding_api(
                    open_ai_client, line.strip(), params["model"]
                )
            else:
                embedding = load_embedding(index, embedding_path)

            data = [{"id": index, "vector": embedding, "plain_text": line.strip()}]

            db_client.insert(collection_name=collection_name, data=data)

        connections.connect(alias="default", uri='db/gars.db')
        collection = Collection(collection_name)
        collection.flush()


def load_db(online):
    """
    Load the database by creating collections and inserting data with generated embeddings.
    """
    params = json.load(open(os.path.join("..", "config", "db_config.json")))

    params["online"] = online
    db_client = MilvusClient("db/gars.db")
    open_ai_client =  None

    # determine if the database is allready initialized
    # TODO try to do this in a more sound way
    """  
    if set(db_client.list_collections()) == set(params["prompt_elements"]):
        return
    """

    collection_names = list(set(params["prompt_elements"]))

    """ Not needed for now
    db_client.drop_collection("User")
    db_client.create_collection(
        collection_name="User",
        dimension=params["embedding_size"] * len(params["prompt_elements"]),
        metric_type=params["metric_type"],
    )
    """

    for collection_name in collection_names:

        print(f"Loading {collection_name} to database")

        prompt_file_name = os.path.join(
            "..",
            "resources",
            "prompt_categories",
            "plain_text",
            collection_name + ".txt",
        )
        export_collection(
            db_client, open_ai_client, collection_name, prompt_file_name, params
        )


if __name__ == "__main__":

    env_path = os.path.join("..", ".env")
    load_dotenv(dotenv_path=env_path)
    parser = argparse.ArgumentParser(description="Initialize database.")

    parser.add_argument(
        "--online", help="Flag to get embeddings via api or by numpy files"
    )

    args = parser.parse_args()
    load_db(args.online)
