from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI


class VectorDB:
    """Class to manage document loading and vector database creation."""
    
    def __init__(self, directory='docs/my_docs', glob="./*.pdf"):
        self.directory = directory
        self.glob = glob

    def load_documents(self):
        loader = DirectoryLoader(self.directory, glob=self.glob, loader_cls=PyPDFLoader)
        return loader.load()

    def create_vector_db(self):
        documents = self.load_documents()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        return Chroma.from_documents(texts, embeddings)


class QAChain:
    """Class to manage the QA chain setup."""

    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        self.model_name = model_name
        self.temperature = temperature

    def create_chain(self):
        model = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\n
        just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise\n
        as possible. Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        vector_db = VectorDB()
        retriever = vector_db.create_vector_db().as_retriever(search_type="similarity", search_kwargs={"k": 2})

        return RetrievalQA.from_chain_type(
            llm=model,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )


def main():
    """Main function to execute the QA system."""
    query = input('\n Q: ')
    qa_chain = QAChain().create_chain()
    result = qa_chain({"query": query})
    print(result['result'])


if __name__ == "__main__":
    main()
