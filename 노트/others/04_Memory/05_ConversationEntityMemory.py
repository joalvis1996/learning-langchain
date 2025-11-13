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
conversation.predict(
    input="창밖으로 끊임없이 비가 내리던 오후, 지하철 끝자락에 앉은 지훈은 손에 쥔 작은 봉투를 내려다보았다. 몇 시간 전, 오래된 책방에서 우연히 건네받은 이 봉투에는 ‘미래를 바꾸고 싶다면, 오늘 밤 11시에 열어보세요’라는 문장이 적혀 있었다. 장난 같으면서도 이상하게 마음을 끄는 그 말에 그는 내내 생각에 잠겼고, 지하철이 종착역에 다다를 때쯤엔 이미 결심이 서 있었다. 집으로 돌아온 지훈은 시계가 11시를 가리키자 조용히 봉투를 뜯었다. 그리고 그 안에서 나온 건 오래된 사진 한 장—아직 떠나지 못한, 그러나 다시 만날 용기도 없었던 누군가의 얼굴이었다"
)

print("저장된 엔티티 정보 (entity_store.store):")

## 실행 결과
# 엔티티: 지훈
# 정보: 지훈은 오래된 책방에서 우연히 받은 '미래를 바꾸고 싶다면 오늘 밤 11시에 열어보세요'라는 문구의 봉투를 11시에 열어, 아직 떠나지 못한 누군가의 오래된 사진 한 장을 발견했다.
 