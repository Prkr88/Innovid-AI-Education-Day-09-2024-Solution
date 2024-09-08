import streamlit as st
import openai
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from PIL import Image
from io import BytesIO
import os
import json
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain import LLMChain, PromptTemplate

# Load environment variables
load_dotenv(find_dotenv())

# Set up OpenAI API using credentials stored in Streamlit secrets
client = openai.OpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"))

# Set up Spotify API using client credentials from Streamlit secrets
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=st.secrets["spotify_client_id"],
    client_secret=st.secrets["spotify_client_secret"]
))


def generate_workout_plan(goals, equipment, preferences):
    """
    Generates a workout plan based on user input (goals, available equipment, and preferences)
    using Azure OpenAI.

    Args:
        goals (str): User's fitness goals.
        equipment (str): Available workout equipment.
        preferences (str): User's workout preferences or limitations.

    Returns:
        str: Generated workout plan.
    """
    llm = AzureChatOpenAI(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("CHAT_MODEL"),
        openai_api_version=os.getenv("OPENAI_API_VERSION")
    )

    # Define the template for generating a workout plan
    template = """
    Create a workout plan for someone with the following goals: {goals}. 
    Available equipment: {equipment}. Preferences: {preferences}.
    """

    # Initialize the LLM chain with the prompt
    prompt = PromptTemplate(template=template, input_variables=["goals", "equipment", "preferences"])
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain and return the workout plan
    plan = chain.run({"goals": goals, "equipment": equipment, "preferences": preferences})
    return plan


def extract_exercises_with_gpt(workout_plan):
    """
    Extracts the exercise names from a workout plan using Azure OpenAI.

    Args:
        workout_plan (str): A workout plan containing exercises and other details.

    Returns:
        list: A list of extracted exercise names.
    """
    llm = AzureChatOpenAI(
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("CHAT_MODEL"),
        openai_api_version=os.getenv("OPENAI_API_VERSION")
    )

    # Define the prompt to extract only exercise names from the workout plan
    prompt = f"""
    You are a smart assistant. Given the following workout plan, please extract only the names of the exercises:

    Workout Plan:
    {workout_plan}

    Please return the list of exercises only, without sets, reps, or any other details.
    """

    # Send the prompt to Azure OpenAI and process the response
    response = llm.predict(prompt)
    exercises = [exercise.strip() for exercise in response.split("\n") if exercise.strip()]

    return exercises


def generate_exercise_image(exercise):
    """
    Generates an image based on the exercise name using Hugging Face's inference API.

    Args:
        exercise (str): The name of the exercise to generate an image for.

    Returns:
        PIL.Image: Generated image.
    """
    url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": exercise}

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()

    img_data = response.content
    img = Image.open(BytesIO(img_data))

    # Ensure image is in RGB mode for consistent display
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    return img


def search_for_spotify_playlist(workout_type):
    """
    Searches Spotify for a workout playlist based on workout type.

    Args:
        workout_type (str): The type of workout to search for.

    Returns:
        tuple: Spotify playlist URL and name, or (None, None) if no playlist is found.
    """
    results = sp.search(q=f"{workout_type} workout", type='playlist', limit=1)
    if results['playlists']['items']:
        playlist = results['playlists']['items'][0]
        return playlist['external_urls']['spotify'], playlist['name']
    return None, None


# Streamlit UI for generating a personalized workout
st.title("Personalized Workout Generator")

# User inputs
goals = st.text_area("What are your fitness goals?")
equipment = st.text_input("What equipment do you have access to?")
preferences = st.text_area("Any workout preferences or limitations?")

if st.button("Generate Workout Plan"):
    if goals and equipment and preferences:
        # Generate workout plan
        workout_plan = generate_workout_plan(goals, equipment, preferences)
        st.subheader("Your Personalized Workout Plan")
        st.write(workout_plan)

        # Extract and visualize exercises
        exercises = extract_exercises_with_gpt(workout_plan)[:2]
        st.subheader("Exercise Visualizations")
        for exercise in exercises:
            st.write(f"Generating visualization for: {exercise}")
            image = generate_exercise_image(f"{exercise} exercise")
            st.image(image, caption=exercise, use_column_width=True)

        # Search for Spotify playlist
        st.subheader("Workout Playlist")
        playlist_url, playlist_name = search_for_spotify_playlist(goals.split()[0])
        if playlist_url:
            st.write(f"We've found a great playlist for your {goals.split()[0]} workout!")
            st.write(f"Playlist: {playlist_name}")
            st.markdown(f"[Open in Spotify]({playlist_url})")
        else:
            st.write("Sorry, we couldn't find a suitable playlist for your workout.")
    else:
        st.warning("Please fill in all the fields before generating a workout plan.")
