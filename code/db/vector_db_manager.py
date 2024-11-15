from pymilvus import MilvusClient, Collection, connections
from os.path import join
import json
from typing import List, Dict, Union
import numpy as np


class VectorDBManager:
    """
    A manager class for handling vector operations in a Milvus database.

    This class provides methods for retrieving, searching, and managing vectors
    in various collections within a Milvus database, such as finding vectors by
    ID, searching for nearest neighbors, and listing collections.

    Attributes:
        params (dict): Configuration parameters for Milvus operations.
    """

    def __init__(self):
        """
        Initializes the VectorDBManager by loading database configuration parameters.

        Establishes a connection to the Milvus database and loads settings from the
        configuration file.
        """
        # Load configuration for database
        config_path_name = join("..", "config", "db_config.json")
        self._client = MilvusClient(uri=join("db", "gars.db"))

        # Establish default connection alias for Milvus operations
        connections.connect(alias="default", uri=join("db", "gars.db"))

        # Load Milvus-specific parameters from the configuration file
        with open(config_path_name, "r") as f:
            self._params = json.load(f)

    def find_by_id(self, collection_name: str, id: int) -> Union[Dict, None]:
        """
        Retrieves a vector by its unique identifier from the specified collection.

        Args:
            collection_name (str): The name of the collection to search in.
            id (int): The unique identifier of the vector to retrieve.

        Returns:
            Union[Dict, None]: The retrieved vector data as a dictionary with vector
            data converted to a NumPy array, or None if not found.

        Raises:
            ValueError: If the specified vector is not found in the collection.
        """
        # Query the collection by ID
        res = self._client.get(collection_name=collection_name, ids=[id])

        # Handle case where vector is not found
        if res is None:
            raise ValueError(
                f"Vector with ID {id} in collection '{collection_name}' was not found!"
            )

        # Convert vector data to NumPy array for compatibility
        res[0]["vector"] = np.asarray(res[0]["vector"])
        return res[0]

    def find_knn(
        self,
        collection_name: str,
        query_vector: List[float],
        num_neighbors: int,
    ) -> List[Dict]:
        """
        Finds the k-nearest neighbors to a query vector in the specified collection.

        Args:
            collection_name (str): The name of the collection to search.
            query_vector (List[float]): The query vector to search for similar items.
            num_neighbors (int): The number of nearest neighbors to retrieve.

        Returns:
            List[Dict]: The search results containing the nearest neighbors as a list
            of dictionaries with vector data and metadata.
        """

        search_params = {
            "metric_type": self._params["metric_type"],
            "params": {},
        }

        # Perform KNN search using Milvus client
        res = self._client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="vector",
            limit=num_neighbors,
            search_param=search_params,
        )

        return res[0]  # Return nearest neighbors list

    def list_collections(self) -> List[str]:
        """
        Lists all collections in the Milvus database.

        Returns:
            List[str]: A list of all collection names in the database.
        """
        return self._client.list_collections()

    def get_collection_size(self, collection_name: str) -> int:
        """
        Retrieves the number of entries in a specified collection.

        Args:
            collection_name (str): The name of the collection to get the size for.

        Returns:
            int: The number of entries in the specified collection.
        """
        # Get collection statistics and retrieve row count
        collection_stats = self._client.get_collection_stats(
            collection_name=collection_name
        )

        return collection_stats["row_count"]
