import math
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from openai import OpenAI
from rec_system.minidiffusionpipeline import MiniDiffusionPipeline


class GenRecSystem(ABC):
    """
    An abstract class for a generative recommendation system that uses a diffusion pipeline
    to generate recommendations based on user preferences. Subclasses should define specific 
    variables and implement the abstract `__call__` method to customize the recommendation 
    process.
    """

    _params: dict  # Parameters specific to the recommendation system, defined in subclasses.

    def __init__(
        self,
        initial_preferences: dict,
        frozen_elements: dict,
        weights: list,
        decay_rate: float,
        total_iterations: int,
        max_jump: float,
        user_sample_stage_size: int,
        diffusion_steps: int,
        dummy: bool
    ) -> None:
        """
        Initializes the recommendation system with user preferences and configurations.

        Args:
            initial_preferences (dict): Initial user preferences to configure embeddings.
            frozen_elements (dict): Elements that are fixed and will not update.
            weights (list): Weights assigned to prompt elements.
            decay_rate (float): Rate at which the influence of user interactions with previous recommendations decays with iterations.
            total_iterations (int): Total number of iterations for convergence.
            max_jump (float): Maximum allowed change in user's prompt vector per iteration.
            user_sample_stage_size (int): Number of initial iterations to randomly generate images to get a sense of user's preferences before beginning recommendations.
            diffusion_steps (int): Number of steps for diffusion-based image generation.
            dummy (bool): Whether to use a dummy model for testing.
        """
        
        # Initialize core parameters and variables for recommendation process
        self._iteration = 0
        self._decay_rate = decay_rate
        self._max_jump = max_jump
        self._total_iterations = total_iterations
        self._user_sample_stage_size = user_sample_stage_size
        self.dummy = dummy  # Flag to use a dummy model for testing

        # Load prompt elements and embedding size from configuration parameters
        self._prompt_elements = self._params["prompt_elements"]
        num_elements = len(set(self._prompt_elements))
        embedding_size = self._params["embedding_size"]

        # Initialize weights, recommendation prompts, and user embeddings
        self._weights = weights
        self._cur_recommendation = np.zeros((num_elements, embedding_size))
        self._cur_prompt = [""] * len(self._prompt_elements)
        self._cur_user_embedding = np.zeros((num_elements, embedding_size))

        # Initialize user preferences and frozen elements
        self.initialize_preferences(initial_preferences)
        self._frozen_elements = frozen_elements
        if self._frozen_elements:
            self._update_frozen_elements()

        # Initialize the diffusion pipeline for image generation
        self.diffusion_pipeline = MiniDiffusionPipeline(diffusion_steps, mock=dummy)

    def _update_frozen_elements(self):
        """
        Updates prompt elements and weights based on frozen elements, ensuring that 
        frozen elements remain unchanged in recommendations.
        """
        for key, value in self._frozen_elements.items():
            if key in self._prompt_elements:
                self._cur_prompt[self._prompt_elements.index(key)] = value
                # Set weight to 0 for frozen elements to prevent updates to user vector
                self._weights[self._prompt_elements.index(key)] = 0

    def _move_to_frozen_elements(self, list_of_elements_to_freeze: List[str]):
        """
        Moves specified elements to frozen status, ensuring they remain unchanged.

        Args:
            list_of_elements_to_freeze (List[str]): List of elements to freeze in recommendations.
        """
        for element in list_of_elements_to_freeze:
            if element in self._prompt_elements:
                index = self._prompt_elements.index(element)
                self._frozen_elements[element] = self._cur_prompt[index]
        self._update_frozen_elements()

    def initialize_preferences(self, initial_preferences: dict):
        """
        Sets up user preferences by generating initial embeddings from the provided preferences.

        Args:
            initial_preferences (dict): Dictionary of user preferences by prompt category.
        """
        open_ai_client = OpenAI()  # Client for generating embeddings
        for prompt_category in initial_preferences.keys():
            index = self._prompt_elements.index(prompt_category)

            # Accumulate embeddings for each term within a prompt category
            for term in initial_preferences[prompt_category]:
                response = open_ai_client.embeddings.create(
                    input=term, model=self._params["model"]
                )
                # If user provides preferences, no needed for sampling stage
                self._user_sample_stage_size = 0
                embedding = np.asarray(response.data[0].embedding)
                # Initialize the user's latent vector as a combination of the embeddings 
                # obtained from their preferences
                self._cur_user_embedding[index] += (
                    embedding * self._params["initialize_weight"]
                )

    def get_prompt(self) -> str:
        """
        Retrieves the current generated prompt for the recommendation.

        Returns:
            str: The current prompt string.
        """
        return " ".join(self._cur_prompt)

    def _generate_image(self) -> str:
        """
        Generates an image based on the current prompt using the diffusion pipeline.

        Returns:
            str: URL of the generated image.
        """
        prompt = self.get_prompt()
        image_content = self.diffusion_pipeline.text2img(prompt)
        return image_content

    def _get_num_neighbors(self, total: int) -> int:
        """
        Determines the number of neighbors to sample in the current iteration, adjusting 
        based on the current stage of sampling and decay rate.

        Args:
            total (int): Total number of entities in the collection.

        Returns:
            int: Number of neighbors to sample.
        """
        if self._user_sample_stage_size == 0:
            # If the user specified their preferences then there's less of a need for experimentation
            total *= 0.25

        non_sample_iter = self._iteration - self._user_sample_stage_size
        r = (1 / total) ** (1 / self._total_iterations)
        return math.ceil(total * r**non_sample_iter)

    @abstractmethod
    def __call__(
        self, rating: float, freeze_elements: List[str], preference_weights: List[float]
    ) -> str:
        """
        Abstract method for handling the recommendation process, requiring subclasses to 
        implement specific behavior.

        Args:
            rating (float): User-provided rating to adjust recommendation.
            freeze_elements (List[str]): Elements to freeze in the recommendation process.
            preference_weights (List[float]): The ratings provided by the user for each aspect of the image.

        Returns:
            str: Result of the recommendation process.
        """
        pass