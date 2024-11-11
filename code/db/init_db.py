from pymilvus import MilvusClient
from tqdm import tqdm
import numpy as np
import json
import os
from pymilvus import Collection, connections


def generate_embedding_api(client, text, model):
    """
    Generate an embedding vector for the given text using a specified model from OpenAI.

    Args:
        client (OpenAI): The OpenAI client instance used to generate embeddings.
        text (str): The text input for which to generate an embedding.
        model (str): The model name to use for generating the embedding.

    Returns:
        list: A list representing the embedding vector for the given text.
    """
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def load_embedding(index, embedding_path):
    """
    Load an embedding from a memory-mapped file.

    Args:
        index (int): The index of the embedding to load.
        embedding_path (str): The path to the memory-mapped embedding file.

    Returns:
        numpy.ndarray: The embedding vector located at the specified index.
    """
    memmap_array = np.lib.format.open_memmap(embedding_path, mode="r")
    row = memmap_array[index]
    return row


def export_collection(db_client, collection_name, file_name, params):
    """
    Exports data from a text file into a Milvus collection, associating each text entry
    with a pre-generated embedding.

    Args:
        db_client (MilvusClient): The Milvus client instance used to interact with the database.
        collection_name (str): The name of the collection in Milvus where data will be stored.
        file_name (str): The file containing plain text data to be inserted into the collection.
        params (dict): Configuration parameters, including `embedding_size` and `metric_type`.

    Steps:
        1. Creates the collection in Milvus using the specified parameters.
        2. Loads each embedding from the embedding file.
        3. Inserts data into Milvus with ID, vector (embedding), and plain text.
        4. Connects to the database, flushes the collection to ensure all data is stored.
    """
    # Define paths for plain text and embedding files
    directory, name = os.path.split(file_name)
    new_directory = directory.replace("plain_text", "embeddings")
    base_name, _ = os.path.splitext(name)
    embedding_file_name = base_name + "_embeddings.npy"
    embedding_path = os.path.join(new_directory, embedding_file_name)

    # Drop and recreate the collection in Milvus
    db_client.drop_collection(collection_name)
    db_client.create_collection(
        collection_name=collection_name,
        dimension=params["embedding_size"],
        metric_type=params["metric_type"],
    )

    # Insert data and corresponding embeddings into the Milvus collection
    with open(file_name, "r") as f:
        for index, line in tqdm(
            enumerate(f), desc=f"Inserting data into {collection_name}"
        ):
            embedding = load_embedding(index, embedding_path)
            data = [
                {"id": index, "vector": embedding, "plain_text": line.strip()}
            ]
            db_client.insert(collection_name=collection_name, data=data)

        # Connect to the default database alias and flush to ensure all entries are saved
        connections.connect(alias="default", uri='db/gars.db')
        collection = Collection(collection_name)
        collection.flush()


def load_db():
    """
    Initializes the Milvus database by creating collections and populating them with data
    and pre-generated embeddings.

    Steps:
        1. Load configuration parameters for database and collections.
        2. Initialize Milvus client and retrieve collection names.
        3. For each collection, call `export_collection` to populate with embeddings.
    """
    # Load database configuration parameters
    params = json.load(open(os.path.join("..", "config", "db_config.json")))

    # Initialize Milvus client and prepare collections
    db_client = MilvusClient("db/gars.db")
    collection_names = list(set(params["prompt_elements"]))

    # Populate each collection with data and embeddings
    for collection_name in collection_names:
        print(f"Loading {collection_name} to database")
        prompt_file_name = os.path.join(
            "..",
            "resources",
            "prompt_categories",
            "plain_text",
            collection_name + ".txt",
        )
        export_collection(db_client, collection_name, prompt_file_name, params)


if __name__ == "__main__":
    # Start the database loading process when script is executed directly
    load_db()
