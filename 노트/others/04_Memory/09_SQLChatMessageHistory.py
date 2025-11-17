import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from langchain_core.load import load

load_dotenv()

# LLM 설정
script_dir = Path(__file__).parent
json_path = script_dir.parent / "ch_3_Model" / "geminiLLM.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
llm = load(data)

# SQLChatMessageHistory 설정
# SQLite 데이터베이스에 대화 기록 저장
db_path = script_dir / "chat_history.db"
message_history = SQLChatMessageHistory(
    connection=f"sqlite:///{db_path}",
    session_id="user_001"
)

print("=== 1번: message_history ===\n", message_history)

# ConversationBufferMemory에 SQLChatMessageHistory 연결
memory = ConversationBufferMemory(
    chat_memory=message_history,
    return_messages=True,
    memory_key="history"
)

# ConversationChain 생성
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)

print("=== 첫 번째 대화 ===")
response1 = conversation.predict(
    input="안녕, 내 이름은 홍길동이야."
)
# print("응답:", response1)

print("=== 2번: message_history ===\n", memory.load_memory_variables({}))

print("\n=== 두 번째 대화 ===")
response2 = conversation.predict(
    input="내 취미는 사진찍기랑 서핑이야."
)
# print("응답:", response2)
print("=== 3번: message_history ===\n", memory.load_memory_variables({}))