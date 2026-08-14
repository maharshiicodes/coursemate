from langchain_community.document_loaders import TextLoader

data = TextLoader("./haaland.txt")
docs = data.load()
print(docs[0])