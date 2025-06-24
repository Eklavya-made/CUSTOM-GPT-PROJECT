# from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings

# ✅ Updated imports (LangChain v0.2+)
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Extract Data From PDF Files in a Directory
def load_pdf_file(data_path):
    loader = DirectoryLoader(
        path=data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# Split Extracted Documents into Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Load Embeddings Model from Local HuggingFace Path
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="C:/PYTHON PROJECT/Models/all-MiniLM-L6-v2"
    )  # Must exist locally
    return embeddings
