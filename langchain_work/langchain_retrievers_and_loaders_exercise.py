from pathlib import Path
import os
import wikipedia

from langchain_community.retrievers import ArxivRetriever, WikipediaRetriever
wikipedia.set_user_agent("TidorinoLangChainExercise/1.0 (contact: tidorino@hotmail.com)")
arxiv_retriever = ArxivRetriever(load_max_docs=2, get_full_documents=False)
wikipedia_retriever = WikipediaRetriever(top_k_results=3)

print("Fetching data from Wikipedia...")
docs = wikipedia_retriever.invoke(input="transformers")

# Look at the retrieved data
for idx, doc in enumerate(docs):
    print(f"\n--- Document {idx+1} ---")
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Snippet: {doc.page_content[:150]}...")

from langchain_core.tools import create_retriever_tool

wiki_tool = create_retriever_tool(wikipedia_retriever, name="WikipediaRetriever", description="Use this tool to search for related pages in Wikipedia.")
print(f'Tool:{wiki_tool}')


#LOADERS:
os.environ["USER_AGENT"] = "TidorinoLangChainExercise/1.0 (contact: tidorino@hotmail.com)"
from langchain_community.document_loaders import DirectoryLoader, TextLoader,WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

print("Downloading book from Project Gutenberg...")
vector_store = Chroma(collection_name="books", persist_directory="/content/chroma_db")

web_loader = WebBaseLoader("https://www.gutenberg.org/cache/epub/1342/pg1342.txt")
#directory_loader = DirectoryLoader("https://www.gutenberg.org/cache/epub/1342/pg1342.txt", glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
books = web_loader.load()
recursive_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_book_chunks = recursive_text_splitter.split_documents(books)

print(f"Originally, there were {len(books)} source files that were then split into {len(split_book_chunks)} chunks.")
print(f"Sample content from chunk 1:\n\n{split_book_chunks[12].page_content[:300]}...")
from itertools import batched

for indx, batch in enumerate(batched(split_book_chunks, 250)):
  vector_store.add_documents(batch)
  print(f"Processed batch #{indx + 1}")
  

book_retriever = vector_store.as_retriever()
info = book_retriever.invoke("book")
print(f'Tool:{info}')




