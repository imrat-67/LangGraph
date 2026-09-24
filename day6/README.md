# Day 6 - One All-in-One Chatbot

A single Streamlit chatbot. No modes, no switches - everything is built into the backend:

| Feature | What it does | Copied from |
|---|---|---|
| 🧠 Memory | Remembers facts about you forever + keeps all your chats (Postgres) | day 4 |
| 🛠️ Tools | Calls `calculate_age` / `random_number` when your question needs it | day 3 |
| 📄 RAG | If you upload a PDF in the sidebar, it uses the PDF to answer | day 5 |

Just type in the chat box. The bot decides by itself what it needs.
(If no PDF is uploaded, it simply doesn't use one.)

## The code in one picture

```
START -> remember -> chat -+-> END        (normal answer)
                     ^     |
                     |     +-> tools      (model asked for a tool)
                     +---------+
```

All of it lives in one file: `app.py`.

## 1) Central virtual environment (do this once)

```bash
cd /home/imtiaj-hossain-saikat/Documents/BJIT/langgraph/day6
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is the ONE central file - it has the packages of every day (1 to 6).

## 2) Things that must be running

- **Ollama** with the model: `ollama pull qwen3:0.6b`
- **Postgres** on port 5442:
  `cd ../day4/simple_memory_chatbot && docker compose up -d`
- **uv** at `/home/imtiaj-hossain-saikat/.local/bin/uv` (it starts the tool server from day 3)

## 3) Run

```bash
cd /home/imtiaj-hossain-saikat/Documents/BJIT/langgraph/day6
source .venv/bin/activate
streamlit run app.py
```

## Things to try

- `Hi, my name is Rahim and I am a data engineer.` (saved to long-term memory)
- `How old is someone born on 1995-03-10?` (uses a tool)
- Upload a PDF, then ask something about it (uses RAG)
- Click **New chat**, then ask `What is my name?` (memory works across chats)
