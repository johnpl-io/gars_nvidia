# Generative Art Recommendation System (GARS)
- This project is Generative Art Recommendation System (GARS). It's main goal is to allow users to 
generate images that are catered toward their preferences or allow them to explore images of their liking. 
GARS works as a generative image diffusion model design tool. GARS takes advantage of fast 
SDXL (Stable Diffusion XL) Lightning models and CPU offloading to allow for seamless use 
on a Nvidia RTX 3070 Ti. 
- The workflow begins by allowing users to start a recommendation session by offering optional preferences and or model configuration. 
Once the user commences the recommendation session, images will appear on the screen in which the user rates from (-1 to 1). 
Users can optionally provide additional control to the recommendation by adjusting prompt element weights and freezing elements entirely. After the session is done, 
a gallery view of all the generated images is provided.

## Description
Since prompt engineering can be overwhelming, we hope to eliminate trial and error required for generative image models by capturing 
an overall sense of user preferences. User preferences are captured through a user vector and learn more over time
allowing for many different unique combinations of prompts to be crafted. Prompts were broken up into their elements using 
https://github.com/krea-ai/open-prompts and through our own input. GARS offers the possibility to generate 1.4720475e+14 combinations of prompts, a space that would be impossible to explore without a recommendation system guiding a user. 
## Getting Started with Nvdia AI Workbench
- Clone the project and add OPENAI_API_KEY as a secret. Note: the OpenAI() api was used to create embeddings for preferences and not 
for image generation. In addition, there is a gars.db container that contains word embeddings created using OpenAI's API.
This was done to not make the user load word embeddings into the database manually. 
- If anything fails with loading the database, you can manually run python3 -m db.init_db within code to take embeddings contained within a .npy file and load them into the database.
- Go to Applications and start up the gars-frotend. From there you join the Gradio frontend from a 
web browser and use GARS accordingly. 
## Hardware Used for Running 
- OS: Ubuntu 22.04.5 LTS x86_64 
- GPU: NVIDIA GeForce RTX 3070 Ti
- CPU: AMD Ryzen 7 5800X (16) @ 3.800G
- Memory: 32014MiB