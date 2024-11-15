from pydoc import visiblename
from signal import valid_signals

import gradio as gr
import os
from rec_system.art_rec import ArtRecSystem
import queue


class UI:
    def __init__(self):
        # Global variables
        self.image_pipeline = None
        self.rec_system = None
        self.output_images = []  # Stores generated images throughout the session
        self.dummy = False  # Dummy flag for testing purposes
        self.validation_state = False  # Tracks if input validation is successful

    def build_ui(self):
        def check_preferences(element_checkboxes, element_preferences):
            """
            Combines selected checkboxes and custom preferences into a single list.
            Args:
                element_checkboxes (list): List of selected options from checkboxes.
                element_preferences (str): Custom preference entered by the user.
            Returns:
                list: Combined list of checkboxes and custom preferences.
            """
            return_list = []
            if len(element_checkboxes) > 0:
                return_list = element_checkboxes
            if len(element_preferences) > 0:
                return_list.append(element_preferences)
            return return_list

        def gars_session_validation(iteration_count: int):
            """
            Validates the session configuration for the GARS system.
            Args:
                iteration_count (int): Number of iterations for the session.
            Updates:
                validation_state (bool): Indicates if the validation passed.
            """
            if not isinstance(iteration_count, int):
                error_message = "Number of Iterations Must Be an Integer!"
                gr.Warning(error_message)
                self.validation_state = False
            elif iteration_count < 10 or iteration_count > 100:
                error_message = "Number of Iterations Must be Between 10 and 100!"
                gr.Warning(error_message)
                self.validation_state = False
            else:
                self.validation_state = True

        def start_gars_session(
            iteration_count,
            sdxl_dropdown,
            subjects_checkboxes,
            custom_preference_subject,
            styles_checkboxes,
            custom_preference_style,
            art_mediums_checkboxes,
            custom_preference_medium,
            progress=gr.Progress(track_tqdm=True),
        ):
            """
            Starts a new GARS (Generative Art Recommendation System) session.
            Args:
                iteration_count (int): Number of iterations for generating recommendations.
                sdxl_dropdown (str): Dropdown choice for the diffusion steps.
                subjects_checkboxes (list): Selected subjects for recommendations.
                custom_preference_subject (str): Custom subject preference.
                styles_checkboxes (list): Selected art styles.
                custom_preference_style (str): Custom style preference.
                art_mediums_checkboxes (list): Selected art mediums.
                custom_preference_medium (str): Custom art medium preference.
                progress (gr.Progress): Gradio progress tracker.
            Returns:
                dict: Updates UI elements based on session initialization.
            """
            self.output_images = []
            gars_session_validation(iteration_count)
            initial_preferences = {
                "subjects": check_preferences(
                    subjects_checkboxes, custom_preference_subject
                ),
                "artists_movements": check_preferences(
                    styles_checkboxes, custom_preference_style
                ),
                "art_mediums": check_preferences(
                    art_mediums_checkboxes, custom_preference_medium
                ),
            }
            # Set diffusion steps based on dropdown selection
            diffusion_steps = 8
            if sdxl_dropdown == "SDXL Lightning [2 Step]":
                diffusion_steps = 2
            elif sdxl_dropdown == "SDXL Lightning [4 Step]":
                diffusion_steps = 4
            progress(0, desc="Starting")
            self.rec_system = ArtRecSystem(
                total_iterations=iteration_count,
                initial_preferences=initial_preferences,
                diffusion_steps=diffusion_steps,
                dummy=self.dummy,
            )

            return {
                initial_setup: gr.update(visible=False),
                GARS: gr.update(visible=True),
                output_image: gr.update(visible=True),
                # Put blank value so loading bar will come up
                output: "",
                progress_bar: gr.update(visible=False),
                restart_row: gr.update(visible=True),
                rating_row: gr.update(visible=True),
            }

        def generate_rec(
            rating,
            subject_weight,
            medium_weight,
            style_weight,
            modifiers_weight,
            locked_elements,
        ):
            """
            Generates a new recommendation based on user feedback and preferences.
            Args:
                rating (float): User rating for the previous recommendation.
                subject_weight (float): Weight for subject preference.
                medium_weight (float): Weight for medium preference.
                style_weight (float): Weight for style preference.
                modifiers_weight (float): Weight for modifiers preference.
                locked_elements (list): Elements to lock in the recommendation.
            Returns:
                dict: Updates UI elements with the generated recommendation and gallery.
            """
            # map between ui "user friendly" names for prompt components and rec system names
            rec_comp_map = {
                "Modifiers": "modifiers",
                "Medium": "art_mediums",
                "Style": "artists_movements",
                "Subject": "subjects",
            }

            lock_element_list = (
                [rec_comp_map[elem] for elem in locked_elements]
                if locked_elements
                else []
            )
            # Ensure previous rating is not used during a restart
            if self.rec_system._iteration == 0:
                rating = 0
            gen_img = self.rec_system(
                rating=rating,
                preference_weights=[
                    modifiers_weight,
                    subject_weight,
                    medium_weight,
                    style_weight,
                ],
                freeze_elements=lock_element_list,
            )
            self.output_images.append(gen_img)
            self.rec_system.diffusion_pipeline.latent_queue.put(0)
            row_visibility = not self.rec_system.is_done
            return {
                output_image: gen_img,
                output_gallery: self.output_images,
                advanced_checkbox_row: gr.update(visible=row_visibility),
                rating_row: gr.update(visible=row_visibility),
                rating_wrapper: gr.update(visible=True),
                iteration_display: gr.update(visible=True),
                gallery_row: gr.update(visible=not row_visibility),
            }

        def show_latent():
            """
            Streams latent images for testing or live preview.
            Yields:
                str: URL of the dummy image if in dummy mode, otherwise streams from the diffusion queue.
            """
            print("got to latent")
            if self.dummy:
                yield "https://fal-cdn.batuhan-941.workers.dev/files/koala/-CQBCeIxrvPqrvt4FDY5n.jpeg"
            else:
                while True:
                    try:
                        print(
                            "Current items in queue:",
                            list(self.rec_system.diffusion_pipeline.latent_queue.queue),
                        )
                        print("waiting for latent")

                        # Try to get an item without blocking
                        image = self.rec_system.diffusion_pipeline.latent_queue.get()
                        if isinstance(image, int):
                            print("None found breaking!")
                            yield self.output_images[-1]
                            break
                        yield image

                        # If item is None, break out of the loop

                        # Process the item

                        print("got latent!")

                    except queue.Empty:
                        print("Queue is empty, retrying...")

        def update_iteration() -> str:
            """
            Provides current iteration status in the GARS session.
            Returns:
                str: Formatted iteration status.
            """
            return f"## Iteration: {self.rec_system._iteration} / {self.rec_system._total_iterations}"

        def show_advanced(status: bool) -> dict:
            """
            Toggles the advanced options tab visibility.
            Args:
                status (bool): Desired visibility status of the advanced tab.
            Returns:
                dict: Updates UI visibility of the advanced tab.
            """
            return {advanced_tab: gr.update(visible=status)}

        def restart_session():
            """
            Restarts the GARS session, resetting output images and UI elements.
            Returns:
                dict: UI reset to initial setup visibility.
            """
            return {
                initial_setup: gr.update(visible=True),
                GARS: gr.update(visible=False),
                advanced_tab: gr.update(visible=False),
                advanced_checkbox_row: gr.update(visible=False),
                rating_wrapper: gr.update(visible=False),
                output_image: None,
                output_gallery: gr.update(visible=False),
                gallery_row: gr.update(visible=False),
                restart_row: gr.update(visible=False),
                iteration_display: gr.update(visible=False),
            }

        def show_progress(iteration_count):
            """
            Validates iteration count and toggles progress bar visibility accordingly.
            Args:
                iteration_count (int): Number of iterations.
            Returns:
                dict: UI updates based on validation outcome.
            """
            gars_session_validation(iteration_count)
            if not self.validation_state:
                return {
                    progress_bar: gr.update(visible=False),
                    initial_setup: gr.update(visible=True),
                }
            return {
                progress_bar: gr.update(visible=True),
                initial_setup: gr.update(visible=False),
            }

        def show_gallery():
            """
            Displays the output gallery and hides other UI elements.
            Returns:
                dict: UI updates to show gallery view.
            """
            return {
                output_gallery: gr.update(visible=True),
                gallery_row: gr.update(visible=False),
                output_image: gr.update(visible=False),
                advanced_checkbox: gr.update(visible=False),
                advanced_tab: gr.update(visible=False),
            }

        green_custom = gr.themes.utils.colors.Color(
            name="green_custom",
            c50="#e0ff00",
            c100="#c8ff00",
            c200="#b1ff00",
            c300="#99f000",
            c400="#81cb00",
            c500="#76b900",
            c600="#6aa600",
            c700="#528100",
            c800="#3b5c00",
            c900="#233700",
            c950="#0b1200",
        )
        with open(os.path.join("frontend", "ui.css"), "r") as f:
            css = f.read()
        theme = gr.themes.Base(
            primary_hue=green_custom,
            secondary_hue=green_custom,
            neutral_hue="stone",
        )
        with gr.Blocks(theme=theme, css=css) as demo:
            with gr.Row():
                with gr.Column("initial setup wrapper") as initial_setup:
                    with gr.Tab("Initial Setup", visible=True):
                        iteration_count = gr.Slider(
                            label="Iteration Count",
                            value=15,
                            minimum=10,
                            maximum=100,
                            step=1,
                        )
                        with gr.Accordion(
                            "Advanced Preferences (optional)", open=False
                        ):
                            gr.Markdown("Selection Preferences")
                            subjects_checkboxes = gr.CheckboxGroup(
                                [
                                    "Animals",
                                    "Landscapes",
                                    "Space",
                                    "Oceans",
                                    "Forests",
                                    "Mountains",
                                    "Rivers",
                                    "Deserts",
                                    "Urban Life",
                                    "Fantasy Creatures",
                                    "Mythology",
                                    "Architecture",
                                    "Cityscapes",
                                    "Flowers",
                                    "Sunsets",
                                    "Underwater Scenes",
                                    "Winter Scenes",
                                    "Autumn Forests",
                                    "Portraits",
                                    "Historical Scenes",
                                    "Abstract Concepts",
                                    "Still Life",
                                    "Vehicles",
                                    "Technology",
                                    "Sports",
                                    "Music",
                                    "Food",
                                    "Fashion",
                                    "Travel",
                                ],
                                label="Subjects",
                            )
                            custom_preference_subject = gr.Textbox(
                                show_label=False, placeholder="Custom Subject"
                            )
                            art_mediums_checkboxes = gr.CheckboxGroup(
                                [
                                    "Digital Art",
                                    "Painting",
                                    "Sculpture",
                                    "Photography",
                                    "Ceramics",
                                    "Woodworking",
                                    "Textiles",
                                    "Glass Art",
                                    "Metalwork",
                                    "Printmaking",
                                ],
                                label="Mediums",
                            )
                            custom_preference_medium = gr.Textbox(
                                show_label=False, placeholder="Custom Medium"
                            )
                            styles_checkboxes = gr.CheckboxGroup(
                                [
                                    "Impressionism",
                                    "Renaissance",
                                    "Baroque",
                                    "Modern Art",
                                    "Pop Art",
                                    "Abstract Art",
                                    "Surrealism",
                                    "Cubism",
                                    "Expressionism",
                                    "Minimalism",
                                ],
                                label="Styles",
                            )
                            custom_preference_style = gr.Textbox(
                                show_label=False, placeholder="Custom Style"
                            )
                            sdxl_dropdown = gr.Dropdown(
                                [
                                    "SDXL Lightning [2 Step]",
                                    "SDXL Lightning [4 Step]",
                                    "SDXL Lightning [8 Step]",
                                ],
                                label="Model",
                                value="SDXL Lightning [8 Step]",
                                interactive=True,
                            )
                        submit_btn = gr.Button("Submit", elem_id="submit-button")
                with gr.Column("Out", visible=False) as progress_bar:
                    with gr.Tab("Loading Model...", visible=True):
                        output = gr.Textbox(
                            label="Loading Model...",
                            placeholder="Waiting on preference",
                            visible=True,
                        )
                with gr.Column("GARS", visible=False) as GARS:
                    with gr.Tab("GARS"):
                        iteration_display = gr.Markdown("", visible=False)
                        with gr.Row(
                            visible=True, elem_id="start-over-button"
                        ) as restart_row:
                            restart_btn = gr.Button("Start Over", scale=0)
                        output_image = gr.Image(
                            streaming=not self.dummy, label="Output Image", visible=True
                        )
                        output_gallery = gr.Gallery(
                            label="Generated images",
                            show_label=False,
                            elem_id="gallery",
                            object_fit="contain",
                            height="auto",
                            visible=False,
                        )
                        with gr.Row(visible=True) as rating_row:
                            with gr.Column(visible=False) as rating_wrapper:
                                rating = gr.Slider(
                                    -1,
                                    1,
                                    value=0,
                                    label="Rating",
                                    minimum=-1,
                                    maximum=1,
                                    scale=3
                                )
                            with gr.Column(visible=True):
                                generate_btn = gr.Button(
                                    "Generate", scale=1, elem_id="generate-button"
                                )
                        with gr.Row(visible=False) as gallery_row:
                            gallery_submit = gr.Button(
                                "Show Gallery", elem_id="show-gallery-button"
                            )
                with gr.Column("Settings", visible=False) as advanced_tab:
                    with gr.Tab("Advanced Options"):
                        gr.Markdown(" Lock Elements")
                        locked_elements = gr.CheckboxGroup(
                            ["Subject", "Medium", "Style", "Modifiers"],
                            show_label=False,
                        )
                        gr.Markdown(" Adjust Element Weights")
                        with gr.Group():
                            subject_weight = gr.Slider(
                                0,
                                1,
                                value=1,
                                label="Subject",
                                minimum=0,
                                maximum=1,
                                interactive=True,
                            )
                            medium_weight = gr.Slider(
                                0,
                                1,
                                value=1,
                                label="Medium",
                                minimum=0,
                                maximum=1,
                                interactive=True,
                            )
                            style_weight = gr.Slider(
                                0,
                                1,
                                value=1,
                                label="Style",
                                minimum=0,
                                maximum=1,
                                interactive=True,
                            )
                            modifiers_weight = gr.Slider(
                                0,
                                1,
                                value=1,
                                label="Modifiers",
                                minimum=0,
                                maximum=1,
                                interactive=True,
                            )
            with gr.Row(visible=False) as advanced_checkbox_row:
                advanced_checkbox = gr.Checkbox(
                    label="Advanced", interactive=True, container=False
                )
            advanced_checkbox.change(
                fn=show_advanced, inputs=advanced_checkbox, outputs=advanced_tab
            )
            submit_btn.click(
                fn=show_progress,
                inputs=[iteration_count],
                outputs=[progress_bar, initial_setup],
            )
            submit_btn.click(
                fn=start_gars_session,
                inputs=[
                    iteration_count,
                    sdxl_dropdown,
                    subjects_checkboxes,
                    custom_preference_subject,
                    styles_checkboxes,
                    custom_preference_style,
                    art_mediums_checkboxes,
                    custom_preference_medium,
                ],
                outputs=[
                    initial_setup,
                    GARS,
                    output_image,
                    iteration_display,
                    output_image,
                    output,
                    progress_bar,
                    restart_row,
                    rating_row,
                ],
            )
            generate_btn.click(fn=show_latent, outputs=[output_image])
            generate_btn.click(
                fn=generate_rec,
                inputs=[
                    rating,
                    subject_weight,
                    medium_weight,
                    style_weight,
                    modifiers_weight,
                    locked_elements,
                ],

                outputs=[output_image, output_gallery, advanced_checkbox_row, rating_row, rating_wrapper, gallery_row, iteration_display],
            )
            restart_btn.click(
                fn=restart_session,
                outputs=[
                    initial_setup,
                    GARS,
                    advanced_checkbox,
                    advanced_tab,
                    output_image,
                    output_gallery,
                    gallery_row,
                    rating_wrapper,
                    advanced_checkbox_row,
                    iteration_display,
                    restart_row,
                ],
            )
            output_image.change(fn=update_iteration, outputs=[iteration_display])

            gallery_submit.click(
                fn=show_gallery,
                outputs=[
                    gallery_row,
                    output_gallery,
                    output_image,
                    advanced_checkbox,
                    advanced_tab,
                ],
            )
            demo.launch(server_name="0.0.0.0", server_port=8080, root_path=proxy_prefix)


proxy_prefix = os.environ.get("PROXY_PREFIX")

UI().build_ui()
