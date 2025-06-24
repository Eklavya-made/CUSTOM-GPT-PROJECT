import os
from transformers import pipeline
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load Embeddings
embeddings = download_hugging_face_embeddings()

# Connect to Pinecone Vector Store
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Load LLM Pipeline (e.g., MBZUAI/LaMini-Flan-T5-783M)
llm_pipeline = pipeline(
    "text2text-generation",
    model="MBZUAI/LaMini-Flan-T5-783M"
)

# Retrieve Relevant Context from Pinecone
def get_context_from_query(query):
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

# Construct Prompt with Context
def build_prompt(context, question):
    return f"""
You are a helpful legal assistant. Use the following context to answer the question clearly and concisely.

Context:
{context}

Question: {question}
"""

# Ask Question and Get Model Response
def ask_question(question):
    context = get_context_from_query(question)
    prompt = build_prompt(context, question)
    result = llm_pipeline(prompt, max_new_tokens=200, do_sample=False)
    return result[0]["generated_text"]
