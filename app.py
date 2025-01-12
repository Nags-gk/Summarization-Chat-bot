import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Hugging Face Model and Tokenizer
HUGGINGFACE_MODEL_PATH = "nagsgk/summarize_model"  # Replace with your Hugging Face model path

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(HUGGINGFACE_MODEL_PATH)
    return tokenizer, model

# Load the model
tokenizer, model = load_model()

# Streamlit UI
st.title("Text Summarization App")
st.write("Enter the text you want summarized and click the 'Summarize' button.")

# User input
input_text = st.text_area("Enter text:", height=200)

# Generate summary on button click
if st.button("Summarize"):
    if input_text.strip():
        # Tokenize input text
        inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate summary
        summary_ids = model.generate(
            inputs, max_length=100, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Display summary
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize!")

# Footer
st.markdown("Built with ❤️ using [Streamlit](https://streamlit.io/) and [Hugging Face](https://huggingface.co/).")
