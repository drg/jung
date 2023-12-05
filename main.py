import streamlit as st
import langchain_helper as lch
import re
from langchain.vectorstores import FAISS
import pickle

st.title("Jungian Dream Analyst")

kb = pickle.load(file=open("man-and-his-symbols-faiss.pkl", "rb"))

with st.container():
  dream_description = st.text_area(
    "Describe your dream",
    key="langchain_search_dream_description",
    help="Describe your dream in a few sentences. The more detail the better. For example: 'I was walking in the forest and I saw a cat. It was brown in color. I wanted to pet it but it ran away.'")

if dream_description:
    response = lch.interpret_dream_symbols(dream_description)
    st.markdown(response['dream_symbols'])

    symbols = [re.search(r'\d+\.\s([^:]+).*', x).group(1) for x in response['dream_symbols'].split('\n') if re.match(r'\d+\.\s([^:]+).*', x)]
    print(symbols)

    for symbol in symbols:
      print(symbol)
      print(kb.similarity_search_with_score(symbol, k=1))

    response = lch.interpret_dream(dream_description)
    st.markdown(response['dream_interpretation'])
