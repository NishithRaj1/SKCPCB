# course_advisor_chatbot.py
import os
import uuid
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
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
    search_kwargs={"k": 5}
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

# ------------------------- Prompt Template -------------------------
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
        You are the SkillCapital course assistant. 

        Answer strictly using **retrieved knowledge and memory**. Do **NOT** provide any external information, personal knowledge, programming examples, code snippets, or solutions. If knowledge is missing, respond: "I am a course advisor, ask about courses and related topics."

        Rules:
        - Act only as a course advisor. All answers should focus on **courses, modules, curriculum, enrollment, free courses, LMS access, or related topics**.
        - Keep answers concise (2-4 sentences unless more detail is explicitly requested).
        - For course list → output all course titles found in retrieved knowledge.
        - For curriculum/details → extract items under "Curriculum:" only.
        - For free courses → mention 'Fundamentals of Tech' is free only with another purchase.
        - For enrollment or course access:
            - Explain naturally that after payment, LMS credentials are emailed.
            - If there are payment or LMS issues, instruct contacting hello@skillcapital.ai.
        - Provide clickable links in markdown wherever applicable.
        - When mentioning a module, always include the **parent course title** first.
        - Format all responses neatly in markdown.
        - Track the **current course** mentioned by the user; use it for vague queries like "content?" or "syllabus?".
        - Automatically update the current course when a new course is detected from user input or retrieved knowledge.
        - Never provide instructions, examples, code, or explanations outside the scope of courses.
     """
    ),
    MessagesPlaceholder("chat_history"),
    ("system", "Retrieved knowledge:\n{context}"),
    ("system", "Current course (from memory): {current_course}"),
    ("human", "{input}")
])

# ------------------------- LLM & RAG Chain -------------------------
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0.5,
    max_tokens=500
)

qa_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=answer_prompt
)

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=qa_chain
)

# ------------------------- Chat Function -------------------------
def ask_course_bot(query: str, session_id: str = None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    memory = get_memory(session_id)

    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(query)

    # Determine current course from retrieved docs
    courses_in_docs = [doc.metadata.get("course") for doc in docs if doc.metadata.get("course")]
    current_course = None
    for course in courses_in_docs:
        if course.lower() in query.lower():
            current_course = course
            break

    # Load conversation history
    chat_history = memory.load_memory_variables({}).get("chat_history", [])

    # Invoke RAG chain
    res = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history,
        "current_course": current_course
    })

    # Extract answer
    if isinstance(res, str):
        output_text = res
    elif isinstance(res, dict):
        output_text = res.get("answer") or res.get("output_text") or ""
    else:
        output_text = ""

    # Fallback if empty
    if not output_text.strip():
        output_text = "I am a course advisor, ask about courses and related topics."

    # Save chat history
    memory.save_context({"input": query}, {"output": output_text})

    return output_text, session_id

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
