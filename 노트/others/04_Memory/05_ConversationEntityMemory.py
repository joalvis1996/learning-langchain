import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain_core.load import load

from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

load_dotenv()

script_dir = Path(__file__).parent
json_path = script_dir.parent / "ch_3_Model" / "geminiLLM.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

llm = load(data)

memory = ConversationEntityMemory(llm=llm, return_messages=True)

conversation = ConversationChain(
    llm=llm,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=memory
)

conversation.predict(input="안녕하세요, 대한민국의 수도는 어디인가요?")
conversation.predict(input="아하 그렇다면 일본의 수도는 어디인가요?")
conversation.predict(input="안녕하세요, 중국의 수도는 어디인가요?")
conversation.predict(input="아하 그렇다면 미국의 수도는 어디인가요?")

print("\n" + "=" * 70)
print("저장된 엔티티 정보 (entity_store.store):")
print("=" * 70)
# 엔티티 정보는 entity_store.store에서 확인
entities = conversation.memory.entity_store.store
if entities:
    for entity, info in entities.items():
        print(f"\n엔티티: {entity}")
        print(f"  정보: {info}")
else:
    print("저장된 엔티티가 없습니다.")
    print("(Gemini API 할당량 초과로 인해 엔티티 추출이 실패했을 수 있습니다)")

print("\n" + "=" * 70)
print("메모리 변수 (load_memory_variables):")
print("=" * 70)
memory_vars = conversation.memory.load_memory_variables({})
print(memory_vars)

print("\n" + "=" * 70)
print("참고:")
print("=" * 70)
print("• 엔티티 정보는 conversation.memory.entity_store.store에서 확인")
print("• load_memory_variables()는 {'history': ..., 'entities': ...} 형태 반환")
print("• Gemini API 할당량 초과 시 엔티티 추출이 실패할 수 있음")