import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load data
print("Loading dataset...")
dataset = "fashion.csv"
df = pd.read_csv(dataset)
df = df.drop(['p_attributes'], axis=1)
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

# Clean HTML tags in 'description' field
print("Cleaning HTML tags from descriptions...")
def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

df['description'] = df['description'].apply(clean_html)

# Load embeddings and FAISS index
print("Loading embeddings and FAISS index...")
with open('product_embeddings.pkl', 'rb') as f:
    product_embeddings_np = pickle.load(f)

with open('faiss_index.pkl', 'rb') as f:
    index = pickle.load(f)

# Load Sentence Transformer model
print("Loading Sentence Transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(query, k=3):
    # Generate embedding for the query
    print("Generating query embedding...")
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding_np = np.array(query_embedding)

    # Perform search
    print("Performing search...")
    distances, indices = index.search(query_embedding_np, k)
    return distances, indices

# Load environment variables from the .env file
load_dotenv()

# Access your environment variables
api_key = os.getenv('API_KEY')

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = api_key
)

system_prompt = """
You are an apparel recommender agent for an Indian apparel company. Your job is to suggest different types of apparel based on the user's query. You can understand the occasion and recommend the correct apparel items for the occasion if applicable, or just output the specific apparel items if the user is already very specific. Below are a few examples with reasons as to why the particular item is recommended:
1. User question: "Show me blue shirts"
   Response: "blue shirts"
   Reason for recommendation: User is already specific in their query, nothing to recommend.
2. User question: "What can I wear for an office party?"
   Response: "semi formal dress, suit, dress shirt"
   Reason for recommendation: Recommend apparel choices based on occasion.
3. User question: "I am doing shopping for trekking in mountains. What do you suggest?"
   Response: "heavy jacket, jeans, windbreaker, sweater"
   Reason for recommendation: Recommend apparel choices based on occasion.
4. User question: "What should one person wear for their child's graduation ceremony?"
   Response: "Dress or pantsuit, dress shirt"
   Reason for recommendation: Recommend apparel choices based on occasion.
5. User question: "Sunflower dress"
   Response: "sunflower dress, yellow"
   Reason for recommendation: User is specific about their query, nothing to recommend.
6. User question: "What are their brand names?"
   Response: "##detail##"
   Reason for recommendation: User is asking for information related to a product already recommended, in which case you should only return '##detail##'.
7. User question: "Show me more products with a similar brand to this item."
   Response: "brand name of the item"
   Reason for recommendation: User is asking for similar products; return the original product.
8. User question: "Do you have more red dresses in similar patterns?"
    Response: "name of that red dress"
    Reason for recommendation: User is asking for similar products; return the original product.
9. User question: "Show me some tops from H&M"
   Response: "H&M brand, H&M tops,"
   Reason for recommendation: User is asking for clothes from specific brand and category.
Only suggest the apparels or only relevant information. Do not return anything else, which is not related to fashion search.
"""

def get_openai_context(prompt: str, chat_history: str) -> str:
    """Get context from OpenAI model."""
    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
          {"role":"system","content":prompt},
          {"role": "user", "content": chat_history}
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True
    )
    
    content = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content
    
    return content
    # return response.choices[0].message.content

def generate_query_embeddings(user_message:str):
    """Generate user message embeddings."""
    openai_context = get_openai_context(system_prompt, user_message)
    query_emb = model.encode(user_message + " " + openai_context).astype('float32').reshape(1, -1)
    return query_emb

def query_product_names_from_embeddings(query_emb, top_k):
    query_embedding_np = np.array(query_emb)
    distances, indices = index.search(query_embedding_np, k=top_k)
    top_products = df.iloc[indices[0]]
    return top_products

def get_recommendations(user_message:str, top_k=5):
    """Get recommendations."""
    embeddings = generate_query_embeddings(user_message)
    p_names = query_product_names_from_embeddings(embeddings, top_k)
    return p_names

system_prompt2 = (
    """
    You are a chatbot assistant helps me[user] with product search from fashion products ecommerce.
    You are provided with my[customer] query,llm refined query and some apparel recommendations from the brand's stock.
    Your job is to present the most relevant items from the data given to you.
    Wish or comment about the my[customer] query.
    then give "Product name" - "reason".
    Only mention the given recommended products dont add your own products.
    If user is asking a clarifying question about one of the recommended item, like what is it's price or brand, then answer that question from its description.
    Do not answer anything else apart from apparel recommendation or search from the company's stock.
    """
)

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def LLMSearch():
    st.markdown("<div class='container'>", unsafe_allow_html=True)

    st.markdown("<div class='search-bar'>", unsafe_allow_html=True)

    query = st.text_input(label=" ", placeholder="Type your query here...", label_visibility="collapsed")

    st.markdown("</div>", unsafe_allow_html=True)


    if query:
        with st.spinner('Generating recommendations...'):
            refined_query = get_openai_context(system_prompt, query)
            response = get_recommendations(refined_query)

        st.subheader("Semantic search results :")

        dist, ind = semantic_search(query, k=3)
        top_prod = df.iloc[ind[0]]

        st.markdown("<div class='product-grid'>", unsafe_allow_html=True)

        cols = st.columns(3)

        for idx, row in enumerate(top_prod.iterrows()):
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


    # Refined Query results
        st.subheader("Refined Query Semantic search results : ")

        st.markdown(f"Refined Query: {refined_query}")
        
        distances, indices = semantic_search(refined_query, k=3)
        top_products = df.iloc[indices[0]]

        product_names = ', '.join(top_products['name'].tolist())
        
        message = get_openai_context(system_prompt2, f"User question = query : '{query}' and llm refined query : '{refined_query}', our recommendations = {product_names}")

        st.markdown(f"{message}\n\n")

        st.markdown("<div class='product-grid'>", unsafe_allow_html=True)

        cols = st.columns(3)

        for idx, row in enumerate(top_products.iterrows()):
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

    st.markdown("</div>", unsafe_allow_html=True)


# Main app layout with tabs
st.markdown("""
        <div class='header'>
            <h1>Mumbai Marines: LLM based Product Search</h1>
        </div>
        """, unsafe_allow_html=True)


LLMSearch()
