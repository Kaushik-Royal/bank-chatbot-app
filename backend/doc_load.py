import os
import ollama
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import re

load_dotenv()  # loads variables from .env into os.environ

class DocLoad:
    # RAG Concept
    # RAG, or Retrieval-Augmented Generation, is an AI framework that enhances Large Language Models (LLMs) 
    # by giving them access to external, up-to-date, and domain-specific information.
    # 
    # LangChain implements a Document abstraction, which is intended to represent a unit of text 
    # and associated metadata. 

    # It has three attributes:
    # page_content: a string representing the content;
    # metadata: a dict containing arbitrary metadata;
    # id: (optional) a string identifier for the document.
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))

        # Vector stores
        # LangChain VectorStore objects contain methods for adding text and Document objects to the store, and querying them
        # Chroma is the vector store that we uses for this example.

        self.vector_store = Chroma(
                collection_name="example_collection",
                embedding_function=self.embeddings,
                persist_directory="./bank_db1",  # Where to save data locally, remove if not necessary
        )
       
     

    def load_documents(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        #pdf documet
        #this is first line
        #this is second line
        #this is the third line
        # list [
        #      docs:  "This is first line",
        #      docs:  "This is second line",
        #      docs:  "This is third line"
        # ]

        # print(len(docs))
        # print(f"{docs[0].page_content[:200]}\n")
        # print(f"{docs[0].page_content}\n")
        # print(f"{docs[0].metadata}\n")
        # print(docs[0].metadata)


        # TextSpliiter
        # We will use a simple text splitter that partitions based on characters. We will split our documents into 
        # chunks of 1000 characters # with 200 characters of overlap between chunks. The overlap helps mitigate the 
        # possibility of separating a statement from # important context related to it. We use the 
        # RecursiveCharacterTextSplitter # We set add_start_index=True so that the character index where each split 
        # Document starts within the initial Document is preserved as metadata attribute “start_index”.



        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)

        # print(len(all_splits))

        # Embeddings
        # Vector search is a common way to store and search over unstructured data (such as unstructured text).
        # The idea is to store numeric vectors that are associated with the text. Given a query, we can embed 
        # it as a vector of the same dimension and use vector similarity metrics (such as cosine similarity) to 
        # identify related text.

        # LangChain supports embeddings from dozens of providers. 
        # These models specify how text should be converted into a numeric vector. we will use Google Gemini model


        

        # Below "add_documents" method performs the following steps for each document you provide:
        # Embeds the Document: It takes the page_content of each Document object and uses the embedding_
        # function (which you provide when initializing Chroma, e.g., GoogleGenerativeAIEmbeddings) to 
        # convert that text into a numerical vector (embedding).

        # Stores in Chroma: It then stores this newly generated embedding along with the original 
        #     ge_content and any associated metadata from the Document object into the Chroma database.
        self.vector_store.add_documents(documents=all_splits)


    # file_path = "C:/dev/tmp/nke-10k-2023.pdf"   
    # load_documents(file_path)

    def sanitize_response(self, response):
        """
        Correctly handles all cases:
        1. Response with <think> block and Answer: tag
        2. Response with <think> block but no Answer: tag
        3. Response with one-line summary and Answer: tag
        4. Raw response without any structure
        """
        # First remove all <think> blocks if they exist
        clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        # Case 1 & 3: Check for Answer: tag (handles both standalone and after summary)
        answer_match = re.search(r'Answer:\s*(.*)', clean_response, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Case 2 & 4: No Answer tag - return the full cleaned response
        if clean_response:
            # Remove any markdown formatting
            clean_response = re.sub(r'\*\*|\*|`', '', clean_response)
            # Take everything up to first line break or end
            return re.split(r'[\n]', clean_response)[0].strip()
        
        return "Information not specified."


    def getResponse(self, query):
            
        # query = "How many distribution centers does Nike have in the US?"
        results = self.vector_store.similarity_search(query)

        print(results)

        context = "\n\n".join([doc.page_content for doc in results])
                    
        # Prepare the prompt with retrieved context
        prompt = f"""Use the following context to answer the question at the end.
        If you don't know the answer, just say you don't know, don't try to make up an answer.
                    
        Context:
        {context}
                
        Question: Answer only: {query}
                
        Answer:"""

        print("prompt")
        print(prompt)   
        # Query the locally running Deepseek model via Ollama
        response = ollama.chat(
        model='deepseek-r1:1.5b',  # Make sure you've pulled deepseek model first: ollama pull deepseek
        messages=[{'role': 'user', 'content': prompt}]
        )
                
        # answer = response['message']['content']
        answer=""
        respcontent = response['message']['content']
        print(respcontent)
        answer=self.sanitize_response(respcontent)
        # match = re.search(r"Answer:\s*(.*?)(?=\n|$)", respcontent, re.DOTALL)
        # if match:
        #     answer = match.group(1)  # Returns '5' as a string
        #     # print(answer)  # Output: 5
        # else:
        #    answer="sorry dont know"
        print(answer)
        return answer
