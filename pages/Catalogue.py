import streamlit as st
import pandas as pd

st.set_page_config(page_title="Restaurant Catalogue", layout="wide")

st.title("Yelp Business Catalogue")

@st.cache_resource
def load_data():
    return pd.read_parquet("datasets/yelp_business.parquet")

df = load_data()

st.data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic"
)
