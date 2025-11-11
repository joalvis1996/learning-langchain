import json
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

load_dotenv()

with open("../ch_3_Model/geminiLLM.json", "r", encoding="utf-8") as f:
    data = json.load(f)

memory = ConversationBufferMemory()

memory.save_context(
    inputs={
"human": "안녕, 내 이름은 홍길동이야"
    },
    outputs={
"ai": "안녕하세요! 홍길동님, 무엇을 도와드릴까요?"
    },
)

memory.save_context(
    inputs={
"human": "양재에서 강남으로 가는 방법 알려줘"
    },
    outputs={
"ai": "140번 버스를 이용하시면 10분 내외로 갈 수 있습니다."
    }
)

# 메시지 히스토리 반환 함수
print(memory.load_memory_variables({}))
## 실행 결과
# {'history': 'Human: 안녕, 내 이름은 홍길동이야\nAI: 안녕하세요! 홍길동님, 무엇을 도와드릴까요?\nHuman: 양재에서 강남으로 가는 방법 알려줘\nAI: 140번 버스를 이용하시면 10분 내외로 갈 수 있습니다.'}