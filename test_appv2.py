import torch
import transformers
import streamlit as st
from io import StringIO

# Page configuration
st.set_page_config(page_title="MatSciQA RAG Demo", page_icon=":microscope:", layout="wide")

# Seed
torch.random.manual_seed(0)

# Set up the title of the application
st.title('MatSciQA RAG Demo')

# Select model
model_option = st.selectbox(
   "Select a model",
   ("microsoft/Phi-3-mini-4k-instruct", "meta-llama/Meta-Llama-3-8B-Instruct", "anthropic/claude-haiku"),
   index=0,
   placeholder="Select model",
)

# Select token length
token_length = st.select_slider(
   "Token length",
   options=[256, 512, 1024, 2048],
   value=512,  # Default value
)

# Read input
option = st.radio(
   "Input method",
   ("Upload text file", "Enter text"),
   index=0,
   help="Choose whether to upload a text file containing your question or to enter the text manually."
)

question = ''

if option == "Upload text file":
    uploaded_file = st.file_uploader("Add text file containing question", type=["txt"])
    if uploaded_file:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        st.text_area("Uploaded Question:", value=string_data, height=150)
        question = string_data
    else:
        st.warning("No file uploaded!")
elif option == "Enter text":
    question = st.text_area("Enter your question here:")

# Load model and tokenizer
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_option,
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_option)

# Prepare the pipeline
pipe = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Set generation arguments
generation_args = {
    "max_new_tokens": token_length,
    "return_full_text": True,
    "do_sample": True,
    "top_p": 0.9,
}

# Create a button that when clicked will output the LLMs generated output
if st.button('Generate output'):
    output = pipe(messages, **generation_args)
    model_output = output[0]['generated_text'][1]['content']
    # Display a generated output message below the button
    st.write(f'{model_output}')
