from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from create_database import embedding_model

load_dotenv()



vector_store = Chroma(
    persist_directory="chrom_db",
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k" : 10,
        "lambda_mult" : 0.5
    }
)

llm = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant who have read many books for system design in software engineering.
            Use ONLY the provided context to answer the question.
            If the answer is not present in the context,
            say : "I could not find the answer in the document"
            """
        ),
        (
            "human",
            """
           Context : {context}



           question : {question}
            """
        )
    ]
)

print("rag system created")

print("press 0 to exit")

while True:
    query = input("You : ")
    if query == "0":
        break
    docs = retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt = prompt.invoke ({
        "context" : context,
        "question" : query
    })
    response = llm.invoke(final_prompt)

    print(f"\n AI : {response.content}")