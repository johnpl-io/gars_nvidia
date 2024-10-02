import json
import os

from .gen_rec import GenRecSystem
from db.vector_db_manager import VectorDBManager
import numpy as np
from typing import List
import time
from dotenv import load_dotenv

env_path = os.path.join("..", ".env")

load_dotenv(dotenv_path=env_path)


class ArtRecSystem(GenRecSystem):
    def __init__(
        self,
        initial_preferences: dict = {},
        frozen_elements: dict = {},
        weights: List[int] = [1, 1, 1, 1],
        decay_rate: float = 0.6,
        total_iterations: int = 10,
        user_sample_stage_size: int = 3,
        max_jump: float = 1e-3,
        diffusion_steps: int = 2
    ):
        """
        Initialize the Art Recommendation System.

        Args:
            weights (list): A list of weights used for recommendations. Default is [1, 1, 1, 1, 1].
            decay_rate (float): The rate at which recommendation weights decay. Default is 0.6.
            total_iterations (int): The total number of iterations for the recommendation process. Default is 10.
            user_sample_stage_size (int): The number of iterations during which user sampling occurs. Default is 3.
            max_jump (float): The maximum number of steps to jump in the recommendation space. Default is 1e-3.
        """
        with open(os.path.join("..","config", "db_config.json")) as f:
            self._params = json.load(f)
        super().__init__(
            initial_preferences=initial_preferences,
            frozen_elements=frozen_elements,
            weights=weights,
            decay_rate=decay_rate,
            total_iterations=total_iterations,
            max_jump=max_jump,
            user_sample_stage_size=user_sample_stage_size,
            diffusion_steps=diffusion_steps
        )

        # determines whether the recommendation process is over
        self.is_done = False

        self.VDBManager = VectorDBManager()

    def adjust_user_preference(self, rating):
        """
        Adjusts the user's vector after receiving the user's rating for the current image.

        Args:
            rating (float): The rating supplied by the user for the current image.
        """
        weighted_image_embedding = self._cur_recommendation * np.asarray(
            self._weights
        ).reshape(-1, 1)
        self._cur_user_embedding += weighted_image_embedding * rating * self._max_jump
        self._cur_user_embedding *= self._decay_rate

    def recommend_prompt(self):
        """
        Uses the current user embedding to retrieve the next prompt to recommend to the user
        using K-nearest neighbors.
        """
        embedding_size = self._cur_recommendation.shape[1]
        user_preferences = np.reshape(self._cur_user_embedding, (-1, embedding_size))
        start_time = time.time()
        for index, element in enumerate(self._prompt_elements):
            vec_index = self._prompt_elements.index(element)
            if element in self._frozen_elements:
                continue
            # print(f"time to find {element} : {time.time() - start_time}")
            collection_size = self.VDBManager.get_collection_size(element)
            num_neighbors = self._get_num_neighbors(collection_size)

            if self._iteration < self._user_sample_stage_size:
                # If we are in the sampling stage, randomly select one of the prompt elements
                k_neighbors = collection_size
                chosen_prompt_id = np.random.choice(k_neighbors)
            else:
                k_neighbors = self.VDBManager.find_knn(
                    element, user_preferences[vec_index], num_neighbors
                )
                chosen_prompt_id = np.random.choice(k_neighbors)["id"]

            chosen_prompt_element = self.VDBManager.find_by_id(
                element, chosen_prompt_id
            )
            self._cur_recommendation[vec_index] = chosen_prompt_element["vector"]
            self._cur_prompt[index] = chosen_prompt_element["plain_text"]
        self._iteration += 1

    def __call__(self, rating: float = 0.0, freeze_elements: List[str] = [], preference_weights: List[float] = []) -> str:
        """
        The main function for the ArtRecSystem class. Takes in the user's rating for the current image,
        updates the user's vector, and then returns the URL of the next recommended image.

        Args:
            rating (float): The rating supplied by the user for the current image.
            weights (float): Element weight
        Returns:
            str: A URL to the next image that was generated for the user.
        """

        # Trying to make the api as stateless as possible following RESTful principles
        old_frozen_elements = self._frozen_elements.copy()
        if freeze_elements:
            self._move_to_frozen_elements(freeze_elements)
        if preference_weights:
            self._weights = preference_weights
        print(self._weights)

        start_time = time.time()
        self.adjust_user_preference(rating)
        print("Time taken to adjust user preference:", time.time() - start_time)
        self.recommend_prompt()
        print("Time taken to recommend prompt:", time.time() - start_time)
        img = self._generate_image()
        print("Current prompt", self._cur_prompt)
        print("Time taken to get img prompt:", time.time() - start_time)
        print("iteration count", self._iteration)
        # Reset frozen elements
        self._frozen_elements = old_frozen_elements
        if self._iteration >= self._total_iterations:
            self.is_done = True
        return img
