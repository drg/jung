import streamlit as st
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import os

from dotenv import load_dotenv

load_dotenv()

@st.cache_data
def interpret_dream(dream_description):
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY"))

    prompt_template_name = PromptTemplate(
        input_variables = ['dream_description'],
        template = "You are a Jungian analyst. You are analyzing a dream. Your "
                   "analysis will include references to concepts from Jungian psychology "
                   "such as the persona, shadow, anima/animus, and self. You will note "
                   "each symbolic reference present in the dream and explain the significance "
                   "of those symbols with reference to the structure of the dream, and the "
                   "usual meaning of those symbols in Jungian psychology. "
                   "The dream is described as follows: {dream_description}."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="dream_interpretation")

    response = name_chain({'dream_description': dream_description})

    return response

@st.cache_data
def interpret_dream_symbols(dream_description):
    llm = ChatOpenAI(
        temperature=0.3,
        model_name="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY"))

    prompt_template_name = PromptTemplate(
        input_variables = ['dream_description'],
        template = "You are a Jungian analyst. You are analyzing a dream. The initial stage "
                   "of your analysis will be to identify each symbolic feature of the dream, "
                   "whether it is a person, animal, object, place, practice, event, or other "
                   "feature. You will produce a list of these symbolic features and a short "
                   "description of their role in the dream. At this stage, you will not provide "
                   "any further interpretation of their meaning. Your response should contain "
                   "only symbols and descriptions, written out like this:\n\n"
                   "First symbol: first symbol description.\n"
                   "Second symbol: second symbol description.\n"
                   "The dream is described as follows: {dream_description}."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="dream_symbols")

    response = name_chain({'dream_description': dream_description})

    return response

def retrieve_symbol_references(symbols):
    from langchain.vectorstores import FAISS
    import pickle

    store = pickle.load(file=open("man-and-his-symbols-faiss.pkl", "rb"))

    return store.similarity_search(symbols)

if __name__ == "__main__":
    print(interpret_dream("I was walking in the forest and I saw a cat. It was brown in color. I wanted to pet it but it ran away."))
