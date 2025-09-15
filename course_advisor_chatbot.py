# course_advisor_chatbot.py
import os
from dotenv import load_dotenv
import uuid
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# ------------------------- Load API Key -------------------------
load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

# ------------------------- Vector DB -------------------------
VECTOR_DB_DIR = "knowledge_vector_db"
COLLECTION_NAME = "skillcapital_knowledge"

embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)
vector_store = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

# ------------------------- Retriever -------------------------
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # retrieve only top relevant chunks
)

# ------------------------- Memory -------------------------
memory_store = {}  # session_id → ConversationBufferMemory

def get_memory(session_id: str):
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return memory_store[session_id]

# ------------------------- Prompt -------------------------
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     You are the SkillCapital course assistant.
     Answer ONLY using the retrieved knowledge. Do NOT invent anything.
     If info is not found, respond exactly: "Please contact hello@skillcapital.ai".

     Rules:
     - Keep answers concise (2-4 sentences) unless more detail is explicitly requested.
     - For course list → output all course titles found in retrieved knowledge, one course per line, numbered starting from 1.
     - For curriculum/details → extract items under "Curriculum:" from the relevant course.
     - For free courses → say 'Fundamentals of Tech' is free only with another purchase.
     - For enrollment or course access:
         - Explain naturally that after payment, LMS credentials are emailed to the student.
         - If there are payment issues, ask to contact hello@skillcapital.ai.
         - If there are LMS access issues, ask to contact hello@skillcapital.ai.
         - Do NOT give step-by-step “Enroll Now” instructions.
     - Always pick the most relevant chunk(s) for the user query.
     - Provide clickable links in markdown wherever applicable.
     """
    ),
    MessagesPlaceholder("chat_history"),
    ("system", "Retrieved knowledge:\n{context}"),
    ("human", "{input}")
])


# ------------------------- LLM & RAG Chain -------------------------
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0.3,
    max_tokens=250  # concise responses
)
qa_chain = create_stuff_documents_chain(llm=llm, prompt=answer_prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# ------------------------- Chat Function -------------------------
def ask_course_bot(query: str, session_id=None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    memory = get_memory(session_id)

    try:
        # Retrieve relevant chunks first
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "Please contact hello@skillcapital.ai", session_id

        # Pass previous conversation to LLM
        chat_history = memory.load_memory_variables({})["chat_history"]
        res = rag_chain.invoke({"input": query, "chat_history": chat_history})

        # Save response to memory
        memory.save_context(
            {"input": query},
            {"output": res if isinstance(res, str) else res.get("answer", "")}
        )

        # Return response
        if isinstance(res, str):
            answer = res
        elif isinstance(res, dict):
            answer = res.get("answer") or res.get("output_text") or "Please contact hello@skillcapital.ai"
        else:
            answer = "Please contact hello@skillcapital.ai"

        return answer, session_id

    except Exception as e:
        print("Error:", e)
        return "Please contact hello@skillcapital.ai", session_id

# ------------------------- CLI -------------------------
if __name__ == "__main__":
    print("🤖 SkillCapital Course Chatbot (type 'exit' to quit)\n")
    sess = None
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break
        if not q or q.lower() in {"exit", "quit", "bye"}:
            print("👋 Goodbye!")
            break
        ans, sess = ask_course_bot(q, sess)
        print(f"Bot: {ans}\n")
