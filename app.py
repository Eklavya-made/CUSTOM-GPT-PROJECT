from flask import Flask, render_template, request
import langchain
from src.helper import download_hugging_face_embeddings
from src.prompt import ask_question
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Set keys as environment variables (if not already set)
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize Flask app
app = Flask(__name__)

# Define route for UI
@app.route("/")
def index():
    return render_template("chat.html")  # Ensure you have templates/chat.html

# Define route for backend communication
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Message:", msg)
    
    # Use vault Q&A pipeline
    response = ask_question(msg)
    
    print("Bot Response:", response)
    return response

# @app.route("/get", methods=["POST"])
# def chat():
#     try:
#         msg = request.form["msg"]
#         print("📝 User Input:", msg)

#         response = langchain.invoke({"input": msg})
#         print("🤖 Response:", response["answer"])

#         return str(response["answer"])
#     except Exception as e:
#         print("❌ ERROR:", e)
#         return "Sorry, something went wrong."

# Entry point
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
