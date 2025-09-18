# SKCPCB
# SkillCapital Course Chatbot 🤖

An AI-powered course advisor chatbot for [SkillCapital.ai](https://www.skillcapital.ai).  
This bot helps students explore courses, view curriculum, and get enrollment support.  
Built with **FastAPI**, **LangChain**, **ChromaDB**, and **OpenAI embeddings**.

---

## 🚀 Features
- **RAG-powered chatbot**: Answers only from the SkillCapital knowledge base (`knowledge.txt`).
- **Course-aware responses**: Provides course lists, curriculum, pricing, free course info, and enrollment process.
- **Memory-enabled**: Remembers the current course in conversation.
- **Web frontend**: Clean chat widget served by FastAPI.
- **Vector database**: Uses **ChromaDB** for persistent embeddings.
- **API + UI**: Chatbot can be accessed via REST API or web interface.

---

## 📂 Project Structure
.
├── build_knowledge_vector_db.py # Builds vector DB from knowledge.txt
├── course_advisor_chatbot.py # Chatbot logic (LLM + retrieval + memory)
├── main.py # FastAPI app (serves API + UI)
├── knowledge.txt # Knowledge base (courses + policies)
├── requirements.txt # Python dependencies
└── .env # API keys and secrets (not committed)

yaml
Copy code

---

## ⚙️ Setup & Installation

### 1. Clone repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
2. Create virtual environment
bash
Copy code
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Add your OpenAI API key
Create a .env file in the root directory:

env
Copy code
OPENAI_API_KEY=sk-xxxxxxx
🏗️ Build Knowledge Vector DB
Run this script to convert knowledge.txt into embeddings stored in Chroma:

bash
Copy code
python build_knowledge_vector_db.py
This will create a knowledge_vector_db/ folder.

▶️ Run Locally
Start the FastAPI server:

bash
Copy code
uvicorn main:app --reload
Chat UI: http://127.0.0.1:8000

API endpoint: POST /chat

json
Copy code
{
  "message": "List all courses",
  "session_id": null
}



🐳 Deploy with Docker
1. Create Dockerfile
dockerfile
Copy code
# ---------------------------
# SkillCapital Course Chatbot
# ---------------------------
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
2. Build Docker image
bash
Copy code
docker build -t skillcapital-chatbot .
3. Run container
Make sure to pass your .env file:

bash
Copy code
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  skillcapital-chatbot
Now the chatbot is live at http://localhost:8000.

🌐 Deployment Options
Local Docker: Run on your machine/server.

Cloud VPS (e.g., AWS EC2, DigitalOcean): Push Docker image and run container.

Container services (Render, Railway, Fly.io, Azure App Service, AWS ECS, etc.): Deploy directly with Dockerfile.

📧 Support
For course or enrollment issues, contact hello@skillcapital.ai.

📜 License
This project is proprietary and intended for SkillCapital.ai use.

yaml
Copy code

---

✅ This `README.md` explains everything (setup, running, Docker).  
Would you like me to also **add a `docker-compose.yml`** (with persistence for Chroma vector DB) so deployment 
