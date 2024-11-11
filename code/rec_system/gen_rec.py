import math
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from openai import OpenAI

from rec_system.minidiffusionpipeline import MiniDiffusionPipeline


class GenRecSystem(ABC):
    # define variables that should be defined in subclasses
    _params: dict

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

        self._iteration = 0
        self._decay_rate = decay_rate
        self._max_jump = max_jump
        self._total_iterations = total_iterations
        self._user_sample_stage_size = user_sample_stage_size
        self.dummy = dummy
        # Load parameters from the configuration file
        self._prompt_elements = self._params["prompt_elements"]
        num_elements = len(set(self._prompt_elements))
        embedding_size = self._params["embedding_size"]

        self._weights = weights
        self._cur_recommendation = np.zeros((num_elements, embedding_size))
        self._cur_prompt = [""] * len(self._prompt_elements)
        self._cur_user_embedding = np.zeros((num_elements, embedding_size))

        self.initialize_preferences(initial_preferences)

        self._frozen_elements = frozen_elements
        if self._frozen_elements:
            self._update_frozen_elements()
        
        self.diffusion_pipeline = MiniDiffusionPipeline(diffusion_steps, mock=dummy)

    def _update_frozen_elements(self):
        for key, value in self._frozen_elements.items():
            if key in self._prompt_elements:
                self._cur_prompt[self._prompt_elements.index(key)] = value
                # kill the weights assuming we want a frozen element to prevent user vector updates
                self._weights[self._prompt_elements.index(key)] = 0

    def _move_to_frozen_elements(self, list_of_elements_to_freeze: List[str]):
        for element in list_of_elements_to_freeze:
            if element in self._prompt_elements:
                index = self._prompt_elements.index(element)
                self._frozen_elements[element] = self._cur_prompt[index]
        self._update_frozen_elements()

    def initialize_preferences(self, initial_preferences: dict):
        open_ai_client = OpenAI()
        for prompt_category in initial_preferences.keys():
            index = self._prompt_elements.index(prompt_category)

            for term in initial_preferences[prompt_category]:
                response = open_ai_client.embeddings.create(
                    input=term, model=self._params["model"]
                )
                print("Setting user sample stage to 0")
                self._user_sample_stage_size = 0
                embedding = np.asarray(response.data[0].embedding)
                self._cur_user_embedding[index] += (
                    embedding * self._params["initialize_weight"]
                )

    def get_prompt(self) -> str:
        """
        Get the current prompt.

        Returns:
        str: The current prompt.

        """
        return " ".join(self._cur_prompt)

    def _generate_image(self) -> str:
        """
        Generate an image based on the current prompt.

        Returns:
            str: URL of the generated image.
        """
        prompt = self.get_prompt()
        image_content = self.diffusion_pipeline.text2img(prompt)
        return image_content

    def _get_num_neighbors(self, total: int) -> int:
        """
        Get the number of neighbors to sample from for the current iteration.

        Args:
            total (int): The total number of entities in a collection.

        Returns:
            int: The number of neighbors to sample from for the current iteration.
        """

        if self._user_sample_stage_size == 0:
            total *= 0.25
        
        non_sample_iter = self._iteration - self._user_sample_stage_size
        r = (1 / total) ** (1 / self._total_iterations)
        return math.ceil(total * r**non_sample_iter)

    @abstractmethod
    def __call__(
        self, rating: float, freeze_elements: List[str], preference_weights: List[float]
    ) -> str:
        """
        Abstract method to be implemented by subclasses to handle the recommendation process.

        Args:
            rating (float): The rating provided by the user.
            freeze_elements (list): A list of elements that you want to freeze

        Returns:
            str: The result of the recommendation process.
        """
        pass
