from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorDB:
    """Class to manage document loading and vector database creation."""
    
    def __init__(self,
                 directory='docs/my_docs',
                 glob='./*.pdf',
                 data_format:str = None,
                 url='https://de.wikipedia.org/wiki/Sokrates'
                #  url='https://en.wikipedia.org/wiki/General_relativity'
                 ):
        self.directory = directory
        self.glob = glob
        self.data_format = data_format
        self.url = url

    def load_documents(self, data_format):
        if data_format.lower() == 'pdf':
            loader = DirectoryLoader(self.directory, glob=self.glob, loader_cls=PyPDFLoader)
            return loader.load()
        elif data_format.lower() == 'url':
            loader = WebBaseLoader(self.url)
            return loader.load()
        else:
            return

    def create_vector_db(self):
        documents = self.load_documents(self.data_format)
        if self.data_format.lower() == 'pdf':
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
            texts = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            return Chroma.from_documents(texts, embeddings)
        elif self.data_format.lower() == 'url':
            text_splitter = RecursiveCharacterTextSplitter()
            document_chunks = text_splitter.split_documents(documents)   
            return Chroma.from_documents(document_chunks, OpenAIEmbeddings()) 
        else:
            return 
    
class ConversationalRetrievalChain:
    """Class to manage the QA chain setup."""

    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        self.model_name = model_name
        self.temperature = temperature

    def create_chain(self):
        model = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
            )
        vector_db = VectorDB(data_format='url')
        retriever = vector_db.create_vector_db().as_retriever(search_type="similarity", search_kwargs={"k": 2})

        return RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            memory=memory,
        )

def main():
    """Main function to execute the QA system."""
    query = input('\n Q: ')
    qa_chain = ConversationalRetrievalChain().create_chain()
    result = qa_chain({"query": query})
    print(result['result'])


if __name__ == "__main__":
    main()
