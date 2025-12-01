📚 Local RAG Research Assistant

A privacy-focused, offline document Q&A system. This tool allows you to chat with your own PDF documents using a local Large Language Model.

🌟 How It WorksLoad: 
You drop PDF research papers into the papers/ folder.
Read: The system reads and "chunks" the text.
Index: It converts text into mathematical vectors and stores them locally.
Chat: When you ask a question, it finds the relevant chunks and uses Llama 3 to generate an answer with citations.

🚀 Business Use Cases

This architecture is production-ready for scenarios such as:
Legal Analysis: Querying vast repositories of case law or contracts.
Technical Support: Instant answers from technical manuals and documentation.
HR Automation: Answering employee policy questions based on internal handbooks.

🛠️ Installation & Setup (Step-by-Step)

Follow these steps to run the project in an isolated Virtual Environment.

1. Prerequisites
Python: Ensure Python (3.10 or higher) is installed.
Ollama: Download and install from ollama.com.

2. Prepare the AI ModelsOpen your terminal (Command Prompt or Terminal) and run:
ollama pull llama3
ollama pull nomic-embed-text

3. Project Setup
Create a folder for your project and move your main.py and requirements.txt into it. Then, open your terminal in that folder.

4. Create a Virtual Environment
A virtual environment keeps this project's libraries separate from your other projects.
For Windows:
python -m venv venv
.\venv\Scripts\activate
For Mac / Linux:
python3 -m venv venv
source venv/bin/activate
(You will see (venv) appear at the start of your command line, indicating it is active.)

5. Install Dependencies
Now that the virtual environment is active, install the required libraries:
pip3 install -r requirements.txt

6. Organize Files
Create a folder named papers inside your project directory.
Paste your PDF documents (research papers) into the papers/ folder.

7. Run the Application
python3 main.py
📝 UsageThe first time you run it, it will take a moment to "vectorize" your PDFs.Once ready, type your question when prompted.To exit, type exit or q.📂 Project Structurelocal_rag/
├── venv/               # Virtual environment (auto-created)
├── papers/             # PUT YOUR PDFS HERE
├── vector_db/          # Database (auto-created)
├── main.py             # The application code
└── requirements.txt    # List of libraries
