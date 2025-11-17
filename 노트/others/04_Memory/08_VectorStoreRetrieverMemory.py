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

print("=== 첫 번째 대화 ===")
response1 = conversation.predict(
    input="아스날 FC는 프리미어리그에서 가장 아름다운 축구를 하는 팀이야."
)
print("응답 1:", response1)
 
print("=== 두 번째 대화 ===")
response2 = conversation.predict(
    input="주요 선수로는 부카요 사카와 마틴 외데고르가 있어."
)
print("응답 2:", response2)
 
print("=== 세 번째 대화 ===")
response3 = conversation.predict(
    input="제주도는 바다가 정말 이쁜 것 같아."
)
print("응답 3:", response3)
 
 
print("=== 메모리에서 검색된 대화 ===")
retrieved = memory.load_memory_variables({"input": "프리미어리그에서 가장 아름다운 축구를 하는 팀은 누구라고?"})
print(retrieved["history"])