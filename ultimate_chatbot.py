from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, YoutubeLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter


class VectorDB:
    """Class to manage document loading and vector database creation."""
    
    def __init__(self, 
                 data_format:str = None,
                 url:str = None,
                 glob: str ='./*.pdf',
                 ):

        self.data_format = data_format
        self.url = url
        self.glob = glob

    def create_vector_db(self):

        if self.data_format.lower() == 'pdf':
            loader = DirectoryLoader(self.url,
                                     glob=self.glob,
                                     loader_cls=PyPDFLoader
                                     )
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
            texts = text_splitter.split_documents(loader.load())
            embeddings = OpenAIEmbeddings()
            return Chroma.from_documents(texts, embeddings)
        
        elif self.data_format.lower() == 'wikipedia':
            loader = WebBaseLoader(self.url)
            text_splitter = RecursiveCharacterTextSplitter()
            document_chunks = text_splitter.split_documents(loader.load()) 
            return Chroma.from_documents(document_chunks, OpenAIEmbeddings()) 
        
        elif self.data_format.lower() == 'youtube':
            loader = YoutubeLoader.from_youtube_url(self.url)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            document_chunks = text_splitter.split_documents(loader.load())
            return Chroma.from_documents(document_chunks, OpenAIEmbeddings()) 
        
class ConversationalRetrievalChain:
    """Class to manage the QA chain setup."""

    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        self.model_name = model_name
        self.temperature = temperature
      
    def create_chain(self, data_format:str,
                     url:str,
                     ):
        
        model = ChatOpenAI(model_name=self.model_name,
                           temperature=self.temperature,
                           )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
            )
        
        vector_db = VectorDB(data_format=data_format,
                             url=url
                             )

        retriever = vector_db.create_vector_db().as_retriever(search_type="similarity",
                                                              search_kwargs={"k": 2},
                                                              )

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
