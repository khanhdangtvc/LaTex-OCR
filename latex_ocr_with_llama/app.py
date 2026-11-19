from huggingface_hub import login 
from dotenv import load_dotenv
import os



load_dotenv(override = True)
huggingface_token = os.getenv("HUGGING_FACE_ACCESS_TOKEN")

print(huggingface_token)