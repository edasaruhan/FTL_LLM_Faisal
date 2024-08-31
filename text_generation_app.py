
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

# Load the pre-trained model and tokenizer
model_name = "bigscience/bloom-560m"  # Example model; change to your preferred model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Streamlit UI setup
st.title("Text Generation with AI")
prompt = st.text_input("Enter your prompt related to SDGs:")
generate_button = st.button("Generate Text")

if generate_button:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(generated_text)
