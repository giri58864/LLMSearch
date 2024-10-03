import streamlit as st
import requests
from dotenv import load_dotenv
import os
from PIL import Image
from io import BytesIO
import base64
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()

# Access your environment variables
api_key = "Bearer " + os.getenv('API_KEY')

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = api_key
)

invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/phi-3-vision-128k-instruct"

headers = {
    "Authorization": api_key,
    "Accept": "application/json",
}

def get_description(image_url):
    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"Describe only the clothing from image as this will be used as a product description:<img src=\"{image_url}\"/>"
            }
        ],
        "temperature": 1,
        "top_p": 0.7,
        "max_tokens": 512
    }
    try:
        session = requests.Session()
        response = session.post(invoke_url, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        # Extract and return only the relevant content
        content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        return content
    except requests.exceptions.HTTPError as errh:
        st.error(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        st.error(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        st.error(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        st.error(f"An Error Occurred: {err}")
    return {}

def upload_image(image_file):
    # Convert uploaded image to base64
    image = Image.open(image_file)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()    
    return f"data:image/png;base64,{img_str}"

def fetch_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except requests.exceptions.HTTPError as errh:
        st.error(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        st.error(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        st.error(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        st.error(f"An Error Occurred: {err}")
    return None

# st.title('Clothing Description Generator')

# option = st.selectbox('Choose input method:', ['Upload Image', 'Provide Image URL'])

# if option == 'Upload Image':
#     uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
#     if uploaded_file:
#         image_url = upload_image(uploaded_file)
# else:
#     image_url = st.text_input('Enter the image URL:')

# if st.button('Get Description'):
#     if image_url:
#         try:
#             description = get_description(image_url)
#             if description:
#                 st.write(description)
#                 refined_description = get_openai_context("Write a clothing product description from the passed text", description)
#                 st.write(refined_description)
#             else:
#                 st.error('No description returned from the API.')
#         except Exception as e:
#             st.error(f"An error occurred: {e}")
#     else:
#         st.error('Please enter a valid image URL or upload an image.')
