import os
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv(dotenv_path=".env", override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("⚠️ OPENAI_API_KEY not found. Please set it in your .env file.")

# ------------------------------
# Paths & constants
# ------------------------------
KNOWLEDGE_FILE = "knowledge.json"
VECTOR_DB_DIR = "knowledge_vector_db"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-ada-002"

# ------------------------------
# Load knowledge.json
# ------------------------------
if not os.path.exists(KNOWLEDGE_FILE):
    raise FileNotFoundError(f"❌ {KNOWLEDGE_FILE} not found.")

with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    knowledge_data = json.load(f)

knowledge_text = json.dumps(knowledge_data, ensure_ascii=False, indent=2)

# ------------------------------
# Split text into chunks
# ------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

chunks = text_splitter.split_text(knowledge_text)
print(f"\n✅ Total chunks created: {len(chunks)}")

# ------------------------------
# Chunk analysis
# ------------------------------
print("\n" + "=" * 80)
print("📊 CHUNK ANALYSIS")
print("=" * 80)

course_names = [
    "Python", "DevOps", "JavaScript", "AWS Cloud", "Azure Cloud",
    "React JS", "UI/UX", "HTML & CSS", "Terraform", "Kubernetes",
    "Site Reliability Engineer", "Oops With Python", "Fundamentals of Tech"
]

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- CHUNK {i} ---")
    print(f"Length: {len(chunk)} characters")

    # Detect course list chunk
    courses_found = [course for course in course_names if course in chunk]
    if courses_found:
        print("🎯 CONTAINS COURSE LIST")
        print(f"📚 Courses found: {', '.join(courses_found)}")

    # Show preview
    preview = chunk[:200].replace("\n", " ")
    print(f"Preview: {preview}...")

print("\n" + "=" * 80)
print("✅ END CHUNK ANALYSIS")
print("=" * 80)

# ------------------------------
# Build Chroma vector DB
# ------------------------------
if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
    print(f"\n⚡ Chroma vector DB already exists at '{VECTOR_DB_DIR}'. Delete it to rebuild.")
else:
    print(f"\n🚀 Building Chroma vector DB...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    vector_db.persist()
    print(f"✅ Chroma vector DB created and saved at '{VECTOR_DB_DIR}'.")
