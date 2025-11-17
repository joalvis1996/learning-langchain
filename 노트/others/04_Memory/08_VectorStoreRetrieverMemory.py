import json
import os
from pathlib import Path
 
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import VectorStoreRetrieverMemory
from langchain_core.load import load
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
 
load_dotenv()


# LLM 설정
script_dir = Path(__file__).parent
json_path = script_dir.parent / "ch_3_Model" / "geminiLLM.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
llm = load(data)
 
# API 키를 읽어 Pinecone 서비스와 통신하는 클라이언트 객체 생성
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
 
# 벡터를 저장할 컨테이너 지정
index_name = "vector-store-retriever-memory"
 
# 지정한 이름의 index 객체 반환
pinecone_index = pc.Index(index_name)
 
# 임베딩 및 벡터 스토어
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings,
    namespace="",
)
 
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
 
memory = VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="history",
    input_key="input",
)
 
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)
