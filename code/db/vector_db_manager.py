from pymilvus import MilvusClient, Collection, connections
from os.path import join
import json
from typing import List, Dict, Union
import numpy as np


class VectorDBManager:
    """
    A manager class for handling vector operations in a Milvus database.

    Attributes:
        params (dict): Configuration parameters for Milvus operations.
    """

    def __init__(self):
        """
        Initializes the VectorDBManager by loading database configuration parameters.
        """
        config_path_name = join("..", "config", "db_config.json")
        self.client = MilvusClient(uri=join("db", "gars.db"))
        connections.connect(alias="default", uri=join("db", "gars.db"))

        with open(config_path_name, "r") as f:
            self.params = json.load(f)

    def find_by_id(self, collection_name: str, id: int) -> Union[Dict, None]:
        """
        Retrieves a vector by its unique identifier from the specified collection.

        Args:
            collection_name (str): The name of the collection.
            id (int): The unique identifier of the vector.

        Returns:
            Union[Dict, None]: The retrieved vector data or None if not found.
        """

        res = self.client.get(collection_name=collection_name, ids=[id])

        if res is None:
            raise ValueError(
                f"{collection_name} with {collection_name}_id was not found!"
            )

        res[0]["vector"] = np.asarray(res[0]["vector"])

        return res[0]

    def find_knn(
        self, collection_name: str, query_vector: List[float], num_neighbors: int
    ) -> List[Dict]:
        """
        Finds the k-nearest neighbors to a query vector in the specified collection.

        Args:
            collection_name (str): The name of the collection.
            query_vector (List[float]): The query vector data.
            num_neighbors (int): The number of nearest neighbors to retrieve.

        Returns:
            List[Dict]: The search results containing the nearest neighbors.
        """

        collection = Collection(collection_name)

        search_params = {"metric_type": self.params["metric_type"], "params": {}}
        res = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="vector",
            limit=num_neighbors,
            search_param=search_params,
        )

        return res[0]

    def list_collections(self) -> List[str]:
        """
        Lists all collections in the database.

        Returns:
            List[str]: A list of collection names.
        """

        return self.client.list_collections()

    def get_collection_size(self, collection_name: str) -> int:
        """
        Gets number of entries in a specified collection

        Args:
            collection_name (str): The name of the collection.

        Returns:
            Int: The number of entries for the given collection
        """

        collection_stats = self.client.get_collection_stats(
            collection_name=collection_name
        )

        return collection_stats["row_count"]
