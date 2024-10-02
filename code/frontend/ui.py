import gradio as gr
import os
from PIL import Image, ImageDraw


from rec_system.minidiffusionpipeline import MiniDiffusionPipeline
from rec_system.art_rec import ArtRecSystem

def greet(name, intensity, rating):
    return f"Hello, {name}!" * int(intensity), create_image(name, rating)

# Initialize image pipeline and recommendation system globally
image_pipeline = None
rec_system = None
current_iteration = 0  # Track the current iteration
output_images = []
def check_preferences(element_checkboxes, element_preferences):
    return_list = []
    if len(element_checkboxes) > 0:
        return_list = element_checkboxes
    if len(element_preferences) > 0:
        return_list.append(element_preferences)
    return return_list

def start_gars_session(iteration_count, sdxl_dropdown, subjects_checkboxes, custom_preference_subject, styles_checkboxes, custom_preference_style, art_mediums_checkboxes, custom_preference_medium, progress=gr.Progress(track_tqdm=True)):

    global rec_system, current_iteration, output_images  # Declare as global to modify outer variables
    progress(0, desc="Starting")
    initial_preferences = {
        "subjects": check_preferences(subjects_checkboxes, custom_preference_subject),
        "artists_movements": check_preferences(styles_checkboxes, custom_preference_style),
        "art_mediums": check_preferences(art_mediums_checkboxes, custom_preference_medium)
    }

    # Initialize the ArtRecSystem with the given preferences
    diffusion_steps = 8
    if sdxl_dropdown == "SDXL Lightning [2 Step]":
        diffusion_steps = 2
    if sdxl_dropdown == "SDXL Lightning [4 Step]":
        diffusion_steps = 4

    rec_system = ArtRecSystem(total_iterations=iteration_count, initial_preferences=initial_preferences, diffusion_steps=diffusion_steps)
    current_iteration = 0  # Reset iteration counter

    gen_img = rec_system(0)
    output_images.append(gen_img)

    return {
        initial_setup: gr.update(visible=False),
        GARS: gr.update(visible=True),
        advanced_checkbox: gr.update(visible=True),
        output_image: gen_img
    }

def generate_rec(rating, subject_weight, medium_weight, style_weight, modifiers_weight, locked_elements, progress=gr.Progress(track_tqdm=True)):
    global output_images
    gen_img = None
    progress(0, desc="Starting")
    if locked_elements:
        lock_element_list = []
        if "Modifiers" in locked_elements:
            lock_element_list.append("modifiers")
        if "Subject" in locked_elements:
            lock_element_list.append("subjects")
        if "Medium" in locked_elements:
            lock_element_list.append("art_mediums")
        if "Style" in locked_elements:
            lock_element_list.append("artists_movements")
        
        gen_img = rec_system(rating=rating, preference_weights=[modifiers_weight, subject_weight, medium_weight, style_weight], freeze_elements=lock_element_list)

    else:
        gen_img = rec_system(rating=rating, preference_weights=[modifiers_weight, subject_weight, medium_weight, style_weight])


    
    output_images.append(gen_img)
    if rec_system.is_done:
        row_visibility = False
    if rec_system:
        return {
            output_image: gen_img,
            output_gallery: output_images,
            rating_row: gr.update(visible=not rec_system.is_done),
            gallery_row: gr.update(visible = rec_system.is_done),
        }
    return None
def update_iteration():
    return f"## Iteration: {rec_system._iteration} / {rec_system._total_iterations}"
def show_advanced(status):
    return {advanced_tab: gr.update(visible=status)}

theme = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="teal",
    neutral_hue="stone",
)

def show_gallery():
    return {
        output_gallery: gr.update(visible=True),
        gallery_row: gr.update(visible=False),
        output_image: gr.update(visible=False)
    }

with gr.Blocks(theme=theme) as demo:
    with gr.Row():
        with gr.Tab("Initial Setup", visible=True) as initial_setup:
            iteration_count = gr.Slider(label="Iteration Count", value=10, minimum=10, maximum=100, step=1)
            with gr.Accordion("Advanced Preferences (optional)", open=False):
                gr.Markdown("Selection Preferences")
                subjects_checkboxes = gr.CheckboxGroup(
                    [
                        "Animals", "Landscapes", "Space", "Oceans", "Forests",
                        "Mountains", "Rivers", "Deserts", "Urban Life",
                        "Fantasy Creatures", "Mythology", "Architecture",
                        "Cityscapes", "Flowers", "Sunsets", "Underwater Scenes",
                        "Winter Scenes", "Autumn Forests", "Portraits",
                        "Historical Scenes", "Abstract Concepts", "Still Life",
                        "Vehicles", "Technology", "Sports", "Music", "Food",
                        "Fashion", "Travel"
                    ],
                    label="Subjects",
                )
                custom_preference_subject = gr.Textbox(show_label=False, placeholder="Custom Subject")

                art_mediums_checkboxes = gr.CheckboxGroup(
                    [
                        "Digital Art", "Painting", "Sculpture", "Photography",
                        "Ceramics", "Woodworking", "Textiles", "Glass Art",
                        "Metalwork", "Printmaking"
                    ],
                    label="Mediums",
                )
                custom_preference_medium = gr.Textbox(show_label=False, placeholder="Custom Medium")
                styles_checkboxes = gr.CheckboxGroup(
                    [
                        "Impressionism", "Renaissance", "Baroque", "Modern Art",
                        "Pop Art", "Abstract Art", "Surrealism", "Cubism",
                        "Expressionism", "Minimalism"
                    ],
                    label="Styles",
                )
                custom_preference_style = gr.Textbox(show_label=False, placeholder="Custom Style")

                sdxl_dropdown = gr.Dropdown(
                    ["SDXL Lightning [2 Step]", "SDXL Lightning [4 Step]", "SDXL Lightning [8 Step]"], label="Model", value="SDXL Lightning [8 Step]", interactive=True)


            submit_btn = gr.Button("Submit")

        with gr.Tab("GARS", visible=False) as GARS:
            iteration_display = gr.Markdown("## Iteration: ", visible=True)
            output_image = gr.Image(label="Output Image", visible=True)
            output_gallery = gr.Gallery(
                label="Generated images", show_label=False, elem_id="gallery", object_fit="contain", height="auto", visible=False)
            with gr.Row(visible=True) as rating_row:
                rating = gr.Slider(-1, 1, value=0, label="Rating", minimum=-1,maximum=1, scale=3)
                generate_btn = gr.Button("Generate", scale=1)
            with gr.Row(visible=False) as gallery_row:
                gallery_submit = gr.Button("Show Gallery")
        with gr.Tab("Settings", visible=False) as settings:
            with gr.Row():
                text_input3 = gr.Textbox(label="Settings Configuration")
        with gr.Column("Settings", visible=False) as advanced_tab:
            with gr.Tab("Advanced Options"):
                gr.Markdown(" Lock Elements")
                locked_elements = gr.CheckboxGroup(
                    [
                        "Subject", "Medium", "Style", "Modifiers"
                    ],
                    show_label=False
                )
                gr.Markdown(" Adjust Element Weights")
                with gr.Group():
                   subject_weight =  gr.Slider(0, 1, value=1, label="Subject", minimum=0,maximum=1, interactive=True)
                   medium_weight = gr.Slider(0, 1, value=1, label="Medium", minimum=0, maximum=1, interactive=True)
                   style_weight = gr.Slider(0, 1, value=1, label="Style", minimum=0, maximum=1, interactive=True)
                   modifiers_weight = gr.Slider(0, 1, value=1, label="Modifiers", minimum=0, maximum=1, interactive=True)


    advanced_checkbox = gr.Checkbox(
        label="Advanced", visible=False, interactive=True, container=False
    )
    advanced_checkbox.change(
        show_advanced, inputs=advanced_checkbox, outputs=advanced_tab
    )

    submit_btn.click(
        fn=start_gars_session,

        inputs=[iteration_count,  sdxl_dropdown, subjects_checkboxes, custom_preference_subject, styles_checkboxes, custom_preference_style, art_mediums_checkboxes, custom_preference_medium],
        outputs=[initial_setup, GARS, advanced_checkbox, output_image, iteration_display],
    )


    generate_btn.click(fn=generate_rec, inputs=[rating, subject_weight, medium_weight, style_weight, modifiers_weight, locked_elements], outputs=[output_image, output_gallery, rating_row, gallery_row])
    output_image.change(fn=update_iteration, outputs=[iteration_display])
    gallery_submit.click(fn=show_gallery, outputs=[gallery_row, output_gallery, output_image])

proxy_prefix = os.environ.get("PROXY_PREFIX")
demo.launch(server_name="0.0.0.0", server_port=8080, root_path=proxy_prefix)
