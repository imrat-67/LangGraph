# Simple Memory Chatbot

STM (chat history) + LTM (remembered facts) — no tools, no RAG.
Model: qwen3:0.6b (via Ollama, local). DB: Postgres in Docker.

## Run it (3 steps)

1) Start Postgres (only needs sudo because your user isn't in the
   `docker` group yet — after adding yourself once, you won't need
   sudo again):

   sudo docker compose up -d

   (one-time fix so you never need sudo for docker again:
    sudo usermod -aG docker $USER   then log out and log back in)

2) Activate the virtual env (already created, packages already installed):

   source venv/bin/activate

3) Run the app:

   streamlit run app.py

Open the link it prints (usually http://localhost:8501).
