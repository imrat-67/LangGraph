# Simple RAG App (Beginner Friendly)

This is a plain RAG (Retrieval Augmented Generation) app:
PDF -> split into chunks -> embed -> find relevant chunks -> ask the model.

No memory, no agents, no tools. Just the RAG basics.

## 1. Install Ollama and pull the model
Download Ollama from https://ollama.com, then run:
```
ollama pull qwen3:0.6b
```
Keep Ollama running in the background (it starts automatically after install).

## 2. Install Python packages
```
pip install -r requirements.txt
```

## 3. Run the app
```
streamlit run app.py
```

## How it works (in plain words)
1. You upload a PDF.
2. The app splits it into small text chunks (~1000 characters each).
3. Each chunk is turned into a vector (a list of numbers) using an embedding model
   (`all-MiniLM-L6-v2`, downloaded automatically the first time).
4. These vectors are stored in a FAISS vector store (a searchable index).
5. When you ask a question, the app also embeds your question and finds the
   4 most similar chunks (this is the "retrieval" part).
6. Those chunks + your question are put into a prompt and sent to the local
   `qwen3:0.6b` model through Ollama, which writes the final answer.
