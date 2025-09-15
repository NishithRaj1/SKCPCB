# build_knowledge_vector_db.py
import os
import uuid
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import re

# ------------------------------
# Load env
# ------------------------------
load_dotenv(dotenv_path=".env", override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Set it in .env")

# ------------------------------
# Paths & constants
# ------------------------------
KNOWLEDGE_FILE = "knowledge.txt"
VECTOR_DB_DIR = "knowledge_vector_db"
COLLECTION_NAME = "skillcapital_knowledge"

# Token-aware chunk targets
CHUNK_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 60

# OpenAI embeddings
EMBEDDING_MODEL = "text-embedding-3-small"

# ------------------------------
# Load knowledge file
# ------------------------------
if not os.path.exists(KNOWLEDGE_FILE):
    raise FileNotFoundError(f"{KNOWLEDGE_FILE} not found.")

with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    knowledge_text = f.read()

print(f"Loaded knowledge.txt: {len(knowledge_text)} characters")

# ------------------------------
# Split by course sections
# ------------------------------
course_sections = re.split(r'(### [A-Za-z0-9 &]+)\n', knowledge_text)
docs = []

for i in range(1, len(course_sections), 2):
    heading = course_sections[i].strip(" #\n")
    content = course_sections[i + 1].strip()
    docs.append(Document(
        page_content=content,
        metadata={"source": KNOWLEDGE_FILE, "course": heading}
    ))

print(f"Total course sections found: {len(docs)}")

# ------------------------------
# Token-aware splitter
# ------------------------------
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
def token_len(s: str) -> int:
    return len(enc.encode(s))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_TOKENS,
    chunk_overlap=CHUNK_OVERLAP_TOKENS,
    length_function=token_len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

split_docs = []
for doc in docs:
    chunks = splitter.split_documents([doc])
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["course"] = doc.metadata["course"]
        split_docs.append(chunk)

print(f"Total chunks created: {len(split_docs)}")

# ------------------------------
# Build or load Chroma vector DB
# ------------------------------
if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
    print(f"\n✅ Loading existing Chroma vector DB from '{VECTOR_DB_DIR}'...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    vector_db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)
else:
    print(f"\n✅ Creating Chroma vector DB at '{VECTOR_DB_DIR}'...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME,
    )
    vector_db.persist()
    print("Chroma vector DB created and persisted.")
