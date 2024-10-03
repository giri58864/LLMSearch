import streamlit as st
import pandas as pd
import re

def load_data():
    data = pd.read_csv("fashion.csv")
    return data

def search_products(data, query):
    # Define a case-insensitive regex pattern for the search query
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    
    # Apply the pattern to the 'name' column to find matches
    data['Match'] = data['name'].apply(lambda x: bool(pattern.search(x)))
    
    # Filter the data where there is a match
    results = data[data['Match']]
    
    # Limit to top results
    results = results.head()
    return results

def highlight_keywords(text, query):
    # Define a case-insensitive regex pattern for the search query
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    
    # Replace the matched text with highlighted HTML (green color)
    highlighted_text = pattern.sub(lambda m: f"<span style='color: red;'>{m.group(0)}</span>", text)
    return highlighted_text

st.title("Basic Keyword Search")

data = load_data()

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown("<div class='container'>", unsafe_allow_html=True)

st.markdown("<div class='search-bar'>", unsafe_allow_html=True)

search_query = st.text_input(label=" ", placeholder="Type your query here...", label_visibility="collapsed")

st.markdown("</div>", unsafe_allow_html=True)

if search_query:
    search_results = search_products(data, search_query)
    
    st.write(f"Found {len(search_results)} results for '{search_query}':")


    st.markdown("<div class='product-grid'>", unsafe_allow_html=True)

    cols = st.columns(3)

    for idx, row in enumerate(search_results.iterrows()):
        with cols[idx % 3]:
            st.markdown(f"""
                <div class='product-item'>
                    <img src='{row[1]['img']}' alt='{row[1]['name']}'>
                    <div class='product-info'>
                        <div class='product-name'>{row[1]['name']}</div>
                        <div class='product-price'>Rs. {row[1]['price']}</div>
                        <div>Colour: {row[1]['colour']}</div>
                        <div>Brand: {row[1]['brand']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)