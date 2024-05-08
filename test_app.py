# Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import streamlit as st
from io import StringIO

# Seed
torch.random.manual_seed(0)

# Set up the title of the application
st.title('MatSciQA RAG Demo')

# Select model
model_option = st.selectbox(
   "Select a model",
   ("microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-4k-instruct"),
   index=None,
   placeholder="Select model",
)

# Read input
option = st.selectbox(
   "Input method",
   ("Upload text file", "Enter text"),
   index=None,
   placeholder="Select model",
)

if option == "upload text file":
    uploaded_file = st.file_uploader("Add text file containing question")
    if uploaded_file:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        st.write(string_data)
        question = string_data
    else:
        st.write(f"No file uploaded!")
elif option == "Enter text":
    question = st.text_input("Enter your question", "Type Here...")
else:
    question = ''

model = AutoModelForCausalLM.from_pretrained(
    model_option,
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_option)

system_prompt = "Solve the following question with highly detailed step by step explanation. Write the correct answer inside a dictionary at the end in the following format. The key 'answer' has a list which can be filled by all correct options or by a number as required while answering the question. For example for question with correct answer as option (a), return {'answer':[a]} at the end of solution. For question with multiple options'a,c' as answers, return {'answer':[a,c]}. And for question with numerical values as answer (say 1.33), return {'answer':[1.33]}"
# question = "When a zinc metal rod is immersed in dilute hydrochloric acid, it results in (A) Evolution of hydrogen (B) Evolution of chlorine (C) Evolution of oxygen (D) No evolution of any gas"

messages = [
    {"role": "user", "content": question}, # system_prompt + question},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 1024,
    "return_full_text": True,
#    "temperature": 0.0,
    "do_sample": True,
    "top_p": 0.9,
}

# Create a button that when clicked will output a greeting
if st.button('Generate output'):
    output = pipe(messages, **generation_args)
    model_output = output[0]['generated_text'][1]['content']
    # Display a greeting message below the button
    st.write(f'{model_output}')