from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
model = ChatGoogleGenerativeAI(
     model = "gemini-1.5-pro",
)

st.header('AI Research Assistant')

paper_input = st.selectbox("Select Research Paper Name", 
                           ["Select...", "Attention is All You Need", 
                            "BERT: Pre-training of Deep Bidirectional Transformers", 
                            "GPT-3:Language Models are Few-Shot Learners", 
                            "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-friendly",
                                                        "Technical",
                                                        "Academic",
                                                        "Professional"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)",
                                                          "Medium (3-5 paragraphs)",
                                                          "Long (6-10 paragraphs)"])
#template
template = load_prompt('template.json')


if st.button('Summarize'):
    chain = template | model 
    result = chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})
    st.write(result.content)
