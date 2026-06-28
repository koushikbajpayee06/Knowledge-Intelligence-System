from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from LangChain.memory import ConversationBufferMemory
from config import Config

class LLMService:
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(
            openai_api_key=Config.OPENAI_API_KEY,
            model_name="gpt-4",
            temperature=0.7
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(),
            memory=self.memory
        )

        
    def get_response(self, question):
        try:
            response = self.chain({"question": question})
            return self.chain({"question": question})
        except Exception as e:
            print(f"Error getting LLM response:{e}")
            return "I encoutered an error processing your request."
