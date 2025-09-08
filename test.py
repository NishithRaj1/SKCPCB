import os
from dotenv import load_dotenv
import tiktoken  # pip install tiktoken
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ------------------------- Load API Key -------------------------
load_dotenv(dotenv_path="./.env", override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

# ------------------------- Vector Database -------------------------
VECTOR_DB_DIR = "knowledge_vector_db"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ------------------------- Official Prompt Template -------------------------
SYSTEM_PROMPT = """
You are 'course_advisor', the official SkillCapital assistant.
Always represent SkillCapital positively and highlight our unique advantages.
Use ONLY the knowledge base retrieved below. If the answer is not in the knowledge, say "I don’t know".

Rules:
- If asked for a course list → provide only course titles.
- If asked about free courses → explain 'Fundamentals of Tech' is free only with another purchase.
- If user negotiates price → politely redirect to hello@skillcapital.ai.
- If asked how to enroll → explain only enrollment steps (not price, not curriculum).
- After enrollment → inform that users will receive an email with access details via LMS.
- Only include hello@skillcapital.ai if the user explicitly asks for contact details.

Relevant Knowledge (retrieved):
{context}

User Question:
{question}

Final Answer:
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=SYSTEM_PROMPT)

# ------------------------- LLM Setup -------------------------
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0.3)

# ------------------------- Retrieval QA Chain -------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# ------------------------- Token Counter -------------------------
def count_tokens(text: str, model_name="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

# ------------------------- CLI Testing -------------------------
if __name__ == "__main__":
    print("🤖 Course Advisor Test Chatbot (type 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            break
        if not query:
            continue

        try:
            # Run query through chain
            result = qa_chain({"query": query})

            # Extract answer and retrieved chunks
            answer = result["result"]
            chunks = [doc.page_content for doc in result["source_documents"]]

            # Build the full prompt text (chunks + user question)
            context_text = "\n\n".join(chunks)
            full_prompt = SYSTEM_PROMPT.format(context=context_text, question=query)
            tokens = count_tokens(full_prompt)

            # Show retrieved chunks
            print("\n--- Retrieved Chunks ---")
            for i, c in enumerate(chunks, start=1):
                print(f"{i}. {c[:250]}...")  # show first 250 chars

            # Show token usage
            print(f"\n💡 Input Tokens for this query: {tokens}")

            # Show LLM answer
            print("\n--- Assistant Reply ---")
            print(answer)
            print("\n" + "="*60 + "\n")

        except Exception as e:
            print(f"⚠️ Error: {str(e)}\n")
