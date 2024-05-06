import os
import json

import google.generativeai as genai

# get the working directory
working_directory = os.path.dirname(os.path.abspath(__file__))

config_fie_path = f"{working_directory}/config.json"
config_data = json.load(open(config_fie_path))

# loading the api key
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

# configuring google.generativeai with API key
genai.configure(api_key=GOOGLE_API_KEY)


# function to load jarvis-pro model for chatbot
def load_jarvis_pro_model():
    jarvis_pro_model = genai.GenerativeModel("gemini-pro")
    return jarvis_pro_model


# function for image captioning
def jarvis_pro_vision_response(prompt, image):
    jarvis_pro_vision_model = genai.GenerativeModel("gemini-pro-vision")
    response = jarvis_pro_vision_model.generate_content([prompt, image])
    result = response.text
    return result


# function to get embedding for text
def embedding_model_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(model=embedding_model, content=input_text, task_type="retrieval_document")
    embedding_list = embedding["embedding"]
    return embedding_list


# function to get a response from jarvis-pro
def jarvis_pro_response(user_prompt):
    jarvis_pro_model = genai.GenerativeModel("gemini-pro")
    response = jarvis_pro_model.generate_content(user_prompt)
    result = response.text
    return result