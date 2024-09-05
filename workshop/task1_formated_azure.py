from dotenv import find_dotenv, load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import AzureChatOpenAI
import streamlit as st
from transformers import pipeline
import os

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Function to convert an image to text using a pre-trained image captioning model
def extract_description_from_image(url):
    # Use the BLIP image captioning model to generate a text description of the image
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    # Generate the text description from the image URL
    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text



# Function to generate a story based on a given scenario using LangChain
def generate_story(scenario):
    # Initialize the OpenAI LLM with the API key loaded from the environment
    llm = AzureChatOpenAI(openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                 azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                 deployment_name=os.getenv("CHAT_MODEL"),
                 openai_api_version=os.getenv("OPENAI_API_VERSION")
                 )

    # Define the prompt template that will be used to generate the story
    template = """
    You are a storyteller. Please generate a short story based on the following scenario.
    The story should be more than 20 words but less than 50.

    CONTEXT: {scenario}
    STORY:
    """
    # Create a PromptTemplate object with the template and input variables
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    # Create the LLMChain with the prompt and the AzureOpenAI model
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain with the scenario to generate a story
    story = chain.run(scenario)


    print(story)
    return story


# Function to convert text to speech using a pre-trained model
def convert_story_to_speech(message):
    from transformers import pipeline
    from datasets import load_dataset
    import soundfile as sf
    import torch

    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    # You can replace this embedding with your own as well.

    speech = synthesiser(message, forward_params={"speaker_embeddings": speaker_embedding})

    sf.write("audio.flac", speech["audio"], samplerate=speech["sampling_rate"])


# Streamlit setup for the web app
st.set_page_config(page_title="Img 2 Audio Story", page_icon="😎")
st.header("Turn Image into Story")

# Upload an image file via the Streamlit UI
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# If a file is uploaded, process it
if uploaded_file is not None:
    # Save the uploaded file locally
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    # Display the uploaded image in the Streamlit app
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    # Convert the image to a descriptive text scenario
    scenario = extract_description_from_image(uploaded_file.name)

    # Generate a story based on the scenario using the LangChain-based function
    story = generate_story(scenario)

    # Convert the story text to speech
    convert_story_to_speech(story)

    # Display the scenario and story in expandable sections
    with st.expander("Scenario"):
        st.write(scenario)
    with st.expander("Story"):
        st.write(story)

    # Play the generated audio file in the Streamlit app
    st.audio("audio.flac")
