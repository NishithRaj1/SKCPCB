import os
from dotenv import load_dotenv

# ------------------------- Updated imports -------------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ------------------------- Load API Key -------------------------
load_dotenv(dotenv_path="./.env", override=True)   
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")

# ------------------------- Vector Database (Chroma) -------------------------
VECTOR_DB_DIR = "knowledge_vector_db"

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vector_store = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ------------------------- Memory for conversation -------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ------------------------- Official Prompt -------------------------
prompt_template = """
You are 'course_advisor', the official SkillCapital assistant.
Always represent SkillCapital positively and highlight our unique advantages.
Use ONLY the knowledge base retrieved below. If the answer is not in the knowledge, say "I don’t know".

Rules:
- If asked for a course list → provide only course titles.
- If asked about free courses → explain 'Fundamentals of Tech' is free only with another purchase.
- If user negotiates price → politely redirect to hello@skillcapital.ai.
- If asked how to enroll → explain only enrollment steps (not price, not curriculum).
- After enrollment → inform that users will receive an email with access details via LMS.
- Always answer in clear and professional English, regardless of the user’s input language.
- Only include hello@skillcapital.ai if the user explicitly asks for contact details.

Chat History:
{chat_history}

Relevant Knowledge (retrieved):
{context}

User Question:
{question}

Final Answer (to user):
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=prompt_template,
)

# ------------------------- LLM Setup -------------------------
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0.3
)

# ------------------------- Conversational RAG Chain -------------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# ------------------------- Function for FastAPI / CLI -------------------------
def course_advi_rag(user_query: str) -> str:
    result = qa_chain.invoke({"question": user_query})
    return result["answer"]  # <-- only return the final answer string

# ------------------------- CLI Testing -------------------------
if __name__ == "__main__":
    print("🤖 Course Advisor Chatbot (type 'exit' to quit)\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit","bye"]:
            print("👋 Goodbye!")
            break
        try:
            response = course_advi_rag(query)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"⚠️ Error: {str(e)}\n")
