import json
import os

from .gen_rec import GenRecSystem
from db.vector_db_manager import VectorDBManager
import numpy as np
from typing import List
import time
import warnings
from numbers import Number

class ArtRecSystem(GenRecSystem):
    """
    A recommendation system for generating art prompts based on user preferences.
    Extends GenRecSystem to provide session based recommendations
    using a text2image diffusion model and vector database storing the
    embeddings of various prompt components.
    """

    def __init__(
        self,
        initial_preferences: dict = {},
        frozen_elements: dict = {},
        weights: List[int] = [1, 1, 1, 1],
        decay_rate: float = 0.6,
        total_iterations: int = 10,
        user_sample_stage_size: int = 3,
        max_jump: float = 1e-3,
        diffusion_steps: int = 2,
        dummy=False,
    ):
        """
        Initialize the Art Recommendation System with user preferences, system parameters,
        and configurations for prompt generation.

        Args:
            initial_preferences (dict): Initial user preferences for setting up embeddings.
            frozen_elements (dict): Elements to keep fixed during the recommendation process.
            weights (list): Weights applied to each prompt element. Default is [1, 1, 1, 1].
            decay_rate (float): Rate at which recommendations decay. Default is 0.6.
            total_iterations (int): Number of iterations for recommendation convergence. Default is 10.
            user_sample_stage_size (int): Initial number of iterations to perform user sampling. Default is 3.
            max_jump (float): Maximum step size for updating user embeddings. Default is 1e-3.
            diffusion_steps (int): Number of diffusion steps for image generation. Default is 2.
            dummy (bool): Whether to use a mock diffusion model for testing. Default is False.
        """
        # Load parameters from external configuration file
        with open(os.path.join("..", "config", "db_config.json")) as f:
            self._params = json.load(f)
        super().__init__(
            initial_preferences=initial_preferences,
            frozen_elements=frozen_elements,
            weights=weights,
            decay_rate=decay_rate,
            total_iterations=total_iterations,
            max_jump=max_jump,
            user_sample_stage_size=user_sample_stage_size,
            diffusion_steps=diffusion_steps,
            dummy=dummy,
        )

        self.is_done = (
            False  # Flag to determine if recommendation session is complete
        )
        self._VDBManager = (
            VectorDBManager()
        )  # Vector database manager for performing similarity search for prompt elements

    def _adjust_user_preference(self, rating: float):
        """
        Adjusts the user's embedding vector based on the user's rating of the current image.

        Args:
            rating (float): User's rating for the current image.
        """
        weighted_image_embedding = self._cur_recommendation * np.asarray(
            self._weights
        ).reshape(-1, 1)
        # Adjust user embedding with rating, scaled by max_jump
        self._cur_user_embedding += (
            weighted_image_embedding * rating * self._max_jump
        )
        # Apply decay to the user embedding
        self._cur_user_embedding *= self._decay_rate

    def _recommend_prompt(self):
        """
        Generates the next recommended prompt based on current user embedding and user preferences
        using a K-nearest neighbors approach.
        """
        embedding_size = self._cur_recommendation.shape[1]
        # reshape user vector into a matrix with each row representing their preference for
        # a corresponding prompt component (subject, style, etc.)
        user_preferences = np.reshape(
            self._cur_user_embedding, (-1, embedding_size)
        )

        # For every prompt component
        for index, element in enumerate(self._prompt_elements):
            vec_index = self._prompt_elements.index(element)
            # if the user chose to freeze the element, then we skip
            # and keep the prompt component as is
            if element in self._frozen_elements:
                continue  # Skip frozen elements

            collection_size = self._VDBManager.get_collection_size(element)
            # obtain the number of neighbors to sample from
            num_neighbors = self._get_num_neighbors(collection_size)

            if self._iteration < self._user_sample_stage_size:
                # During the initial sampling stage, we select a random prompt component
                k_neighbors = collection_size
                chosen_prompt_id = np.random.choice(k_neighbors)
            else:
                # Use user prompt component vector to query the vector database
                # for k most preferred prompt component elements
                k_neighbors = self._VDBManager.find_knn(
                    element, user_preferences[vec_index], num_neighbors
                )
                # then randomly choose from the nearest neighbors
                chosen_prompt_id = np.random.choice(k_neighbors)["id"]

            # Update current recommendation with the selected prompt element
            chosen_prompt_element = self._VDBManager.find_by_id(
                element, chosen_prompt_id
            )
            self._cur_recommendation[vec_index] = chosen_prompt_element[
                "vector"
            ]
            self._cur_prompt[index] = chosen_prompt_element["plain_text"]

        # Finally we increment our iteration count
        self._iteration += 1

    def _validate_inputs(
        self,
        rating: float,
        freeze_elements: List[str],
        preference_weights: List[float],
    ) -> None:
        """
        Validates the input parameters for the recommendation process, ensuring 
        they meet expected types, ranges, and allowable values.

        Args:
            rating (float): User rating for the current image, expected to be between -1.0 and 1.0.
            freeze_elements (List[str]): List of elements to freeze in the recommendation process.
                                        Each element must be a string and must be one of the 
                                        allowed prompt elements defined in the system.
            preference_weights (List[float]): List of weights for user preferences, where each
                                            weight is a float between 0.0 and 1.0.

        Raises:
            ValueError: If certain input parameters do not meet the expected types or 
                        constraints. Specifically:
                        - If `freeze_elements` is not a list of strings or contains elements 
                        not in the allowed prompt elements.
                        - If `preference_weights` is not a list of floats or contains values 
                        outside the 0.0 to 1.0 range.
            UserWarning: If the rating is outside of the range from -1.0 to 1.0 as the recommendation system
                        has only been tested on these values.
        """
        # Validate rating
        if not isinstance(rating, Number):
            raise ValueError("Rating must be a number.")
        if not -1.0 <= rating <= 1.0:
            warnings.warn("Our recommendation system was only tested with ratings ranging from -1.0 to 1.0. ")
        
        # Validate freeze_elements
        allowed_elements = list(set(self._prompt_elements))
        if not isinstance(freeze_elements, list):
            raise ValueError("Freeze elements must be a list.")
        if not all(isinstance(element, str) for element in freeze_elements):
            raise ValueError("Each element in freeze_elements must be a string.")
        if not all(element in allowed_elements for element in freeze_elements):
            raise ValueError(
                f"Each element in freeze_elements must be one of the allowed elements: {allowed_elements}"
            )
        
        # Validate preference_weights
        if not isinstance(preference_weights, list):
            raise ValueError("Preference weights must be a list.")
        if not all(isinstance(weight, float) or isinstance(weight, int) for weight in preference_weights):
            raise ValueError("Each weight in preference_weights must be a float.")
        if not all(0.0 <= weight <= 1.0 for weight in preference_weights):
            raise ValueError("Each weight in preference_weights must be between 0.0 and 1.0.")

    
    def __call__(
        self,
        rating: float = 0.0,
        freeze_elements: List[str] = [],
        preference_weights: List[float] = [],
    ) -> str:
        """
        Main function to process user input, adjust preferences, and generate the next image prompt.

        Args:
            rating (float): User rating for the current image, used to adjust preferences.
            freeze_elements (List[str]): Elements to keep fixed in the current recommendation.
            preference_weights (List[float]): Updated weights for specific prompt elements.

        Returns:
            str: URL of the next generated image based on the current prompt.
        Raises:
            ValueError: If any of the input parameters do not meet the expected types or 
                        constraints as specified in validate_inputs.
        """

        self._validate_inputs(rating,
                             freeze_elements,
                             preference_weights)
        
        # Save the current frozen elements and update with new ones
        old_frozen_elements = self._frozen_elements.copy()
        if freeze_elements:
            self._move_to_frozen_elements(freeze_elements)
        if preference_weights:
            self._weights = preference_weights

        print("\n=== ArtRecSystem Call ===")
        print(
            f"Starting recommendation iteration {self._iteration + 1}/{self._total_iterations}"
        )
        print(f"User Rating Received: {rating}")
        print(f"Freeze Elements: {freeze_elements}")
        print(f"Updated Weights: {self._weights}")
        print("=" * 30)

        start_time = time.time()

        # Update user preferences based on rating
        self._adjust_user_preference(rating)
        adjust_time = time.time() - start_time
        print(
            f"[{adjust_time:.2f}s] User preferences adjusted based on rating."
        )

        # Generate the next prompt based on updated preferences
        self._recommend_prompt()
        recommend_time = time.time() - start_time
        print(
            f"[{recommend_time:.2f}s] New prompt recommended based on user preferences."
        )

        # Generate the image using the diffusion model
        img = self._generate_image()
        generation_time = time.time() - start_time
        print(f"[{generation_time:.2f}s] Image generated based on prompt.")

        # Log the current prompt and iteration information
        print("\n--- Recommendation Details ---")
        print(f"Current Prompt: {' '.join(self._cur_prompt)}")
        print(f"Iteration Count: {self._iteration}")
        print(f"Total Time Taken: {generation_time:.2f}s")
        print("=" * 30)

        # Restore frozen elements to the previous state
        self._frozen_elements = old_frozen_elements

        # Check if the total iterations have been reached
        if self._iteration >= self._total_iterations:
            self.is_done = True
            print("Recommendation process complete.")

        return img
