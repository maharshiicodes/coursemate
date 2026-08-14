from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

data = TextLoader("./document-loaders/haaland.txt")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
chunks = splitter.split_documents(docs)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are an AI that is a football enthusiast and summarizes the text"),
        ("human","{data}")
    ]
)
final_prompt = prompt.format_messages(data = docs[0].page_content)

model = ChatMistralAI(model="mistral-small-2506")
response = model.invoke(final_prompt)
print(response.content)