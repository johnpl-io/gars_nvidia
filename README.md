# Generative Art Recommendation System (GARS)


## Table of Contents
- [Introduction](#introduction)
- [Project Description](#project-description)
- [Technical Details](#technical-details)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Hardware Used](#hardware-used)

# Introduction








GARS is a generative art recommendation system that generates personalized artwork by adapting to individual user preferences and iteratively refining each piece based on real-time feedback to create a unique and evolving art experience. Currently, the nature of the interaction between users and most generative art models is much like that between an artist and a commissioner in that the user instructs the model on what to create and the model creates it while taking some liberties. While this form of interaction can be useful, it has certain limitations. First, being able to articulate your preferences effectively to the model requires considerable expertise in prompt engineering and a great deal of familiarity with all of the quirks associated with a specific model. Moreover, we believe that it is ineffective if the user's goal is to explore and expand their interests or if more generally, they don't know what they like. GARS radically transforms this dynamic and in so doing, empowers users, especially those with minimal experience in generative models, to efficiently navigate and explore the vast space of artworks available to them.

### GARS in Action

https://github.com/user-attachments/assets/dea18c52-e8e4-42af-a464-a50dad21af9a


## Project Description

One challenge that we faced was that in most recommendation systems, the set of items being recommended to the user is finite and each item is known and easily retrievable. However, in our case, the set of images that we can recommend to the user is nearly infinite and retrieving or generating a single image can take entire seconds these images. To compensate for this, we decided to instead of having our recommendation system recommend images to the user, to have it recommend the individual prompt components that when put together in a prompt template, form a prompt that we can feed into a text-to-image model. Our database of prompt components was extracted from a large list of publicly available prompts: https://github.com/krea-ai/open-prompts as well as through our own input. Through this prompt-based representation, GARS is able to efficiently recommend images among a space of roughly 1.47e+14 images.

The workflow begins by allowing users to start a recommendation system by offering optional preferences, model configuration, and iteration count. By setting optional preferences, the user can strongly bias the system towards recommending artworks aligning with those preferences. In choosing the model used to generate the artworks, there is generally a tradeoff between time taken to generate and the quality of the generated artwork where models with fewer diffusion steps are very fast at the expense of sacrificing some quality. Finally, the iteration count determines how many recommendations the system will provide during a GARS session. Generally these sessions are designed so that the system converges to a specific image by the end of a session. In our experience, 15 to 25 iterations seem to provide the best experience to users but this isn't something that we have extensively tested.

![image](https://github.com/user-attachments/assets/49dcc442-d94f-4183-854d-f31e478d1c11)

Once the user begins the recommendation session, they will be presented with images which they can rate from -1 to 1. Users can also optionally provide additional control to the recommendation by adjusting prompt component weights and freezing elements entirely.

![Screenshot from 2024-11-15 18-59-48](https://github.com/user-attachments/assets/0acc8829-1cfb-41c7-89d3-5af4d15f91a9)



When the session is complete, a gallery view of all the generated images is provided where the user has the option to download and save the ones they liked the most. Additionally, the user also has the option to start over and begin a new session.

![Screenshot from 2024-11-15 19-03-14](https://github.com/user-attachments/assets/727b471a-ad41-4b8e-8de3-2a3cb20e07a9)



During the entire session, users can also look at the terminal for logging information where they will see things such as the prompt used to generate their image as well as the amount of time taken to recommend the image.

<img width="851" alt="image" src="https://github.com/user-attachments/assets/db6f0e82-9f55-494b-98b4-9235e426f830">


## Technical Details

To efficiently recommend images to the user, we encoded the user's preference as a matrix with each row vector representing a user's preference for one of the prompt components. In particular, the row vector was represented as a weighted moving average of the embeddings of the corresponding prompt components that the user was exposed to with the weights being the rating supplied by the user. Then to recommend a prompt component, the user's corresponding row vector was used as a query vector to find the prompt components with the highest similarity score (dot product) to the row vector. To do this efficiently, we stored the prompt component embeddings in a Milvus database which is a vector database that is optimized for vector similarity searches. This allowed us to recommend prompts in under a second.

 In many recommendation systems, there is a balancing act between exploration and exploitation, where the system must decide whether to explore new options to discover potentially better recommendations or exploit known preferences to provide the most relevant and satisfying choices to the user. For our recommendation system, we achieved this balance by instead of recommending the most similar prompt component recommended to the user, randomly sampling from the top k most similar prompt components to recommend to the user. The idea behind this is that as the session continues, k will decrease so that in the beginning the system will attempt to explore more of the space. But as time goes on and it becomes more confident in the model it has developed of the user's preferences, it will decrease k and provide more precise recommendations to the user. 

To generate the images we used ...
GARS works as a generative image diffusion model design tool. GARS takes advantage of fast 
SDXL (Stable Diffusion XL) Lightning models to allow for seamless use 
on a Nvidia RTX 3070 Ti a consumer grade GPU. Through the use of CPU offloading, we were able to allow to model to be efficiently fit in and swapped into GPU memory when needed. 
Since we are running fully locally, we are able to get additional features, such as loading and decoding latent representation of images in real time. To do so 
we created a queue that communicates across threads to pipe data to the frontend. 

We chose to use the Gradio to communicate between our backend and frontend as it is very lightweight and flexible which makes it ideal for providing rapid recommendations to the user.

Finally, for our user interface, we chose Gradio because of its seamless integration with the diffusion models we utilized. This allows for quick and efficient interactions, enabling users to easily explore and refine image generation in real-time. Gradio's flexibility made it simple to display results and manage inputs, streamlining the entire user experience.

## Getting Started

1. Clone the project and add OPENAI_API_KEY as a secret. Note: the OpenAI() api was used to create embeddings for preferences and not 
for image generation. In addition, there is a gars.db container that contains word embeddings created using OpenAI's API.
This was done to not make the user load word embeddings into the database manually.

3. Go to Applications and start up the Gars-frotend. From there you join the Gradio frontend from a 
web browser and use GARS accordingly.

![image](https://github.com/user-attachments/assets/a97387a3-2cd0-48f4-a269-16950c58edc1)
![image](https://github.com/user-attachments/assets/fa8e7039-c712-491b-8714-9e09cf8a8c8d)

4. The first run will take a while (~5-7 min) the model will be downloaded from hugging face. 
Subsequent runs with that model will be fast as the model will be cached and caching is persistent across runs.

## Hardware Used
- OS: Ubuntu 22.04.5 LTS x86_64 
- GPU: NVIDIA GeForce RTX 3070 Ti
- CPU: AMD Ryzen 7 5800X (16) @ 3.800G
- Memory: 32014MiB
- Web Browser: Google Chrome
