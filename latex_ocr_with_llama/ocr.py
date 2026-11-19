from huggingface_hub import login
from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForImageTextToText
from dotenv import load_dotenv
import torch
import os
from PIL import Image

PROMPT = [
    {'role':'system',
     'content':'You are a helpful assistant that turn image of latex equation into text'},
    {
    'role': 'user',
    'content': [
                {"type":"image"},
                {"type": "text",
                "text": """Understand the mathematical equation in the provided image and output the corresponding LaTeX code.
                Here are some guidelines you MUST follow or you will be penalized:
                - NEVER include any additional text or explanations.
                - DON'T add dollar signs ($) around the LaTeX code.
                - DO NOT extract simplified versions of the equations.
                - NEVER add documentclass, packages or begindocument.
                - DO NOT explain the symbols used in the equation.
                - Output only the LaTeX code corresponding to the mathematical equations in the image."""}
                ]}]

def hf_login():
    load_dotenv(override = True)
    hf_token = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
    login(hf_token, add_to_git_credential=True)



class LlamaOCR():
    def __init__(self,model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", quant:bool =True ):
        self.model_id = model_id
        if quant:
            self.quant_config = BitsAndBytesConfig(    
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4")
        else:
            self.quant_config=None

        # Model processor: transfomr input to the correct  format of the model
        self.processor = AutoProcessor.from_pretrained(model_id)
        # Load the model
        self.model = AutoModelForImageTextToText.from_pretrained(model_id,device_map="auto",quantization_config = self.quant_config )
        # Evaluate
        self.model.eval()
    def extract_latex(self,image:Image.Image,prompt:str = PROMPT) -> str:
        # Format the conversation, turn the PROMPT into the format the llama model expected
        text_input = self.processor.apply_chat_template(prompt,add_generation_prompt =True, tokenize = False)

        # Convert text and image to PyTorch Tensors in the correct format
        final_input = self.processor(images = [image], text = [text_input], add_special_tokens=False, return_tensors="pt").to('cuda')

        prompt_len = final_input['input_ids'].shape[-1]
        # Generate output
        with torch.inference_mode():
            output = self.model.generate(**final_input,max_new_tokens = 100)

        # Decode only the latex text
        latext = self.processor.decode(output[0][prompt_len:],skip_special_tokens=True)
        return latext

        



hf_login()
model = LlamaOCR()
img = Image.open("img.png")
model.extract_latex(image=img)
