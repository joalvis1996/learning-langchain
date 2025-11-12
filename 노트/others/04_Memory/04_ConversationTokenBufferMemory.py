import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.load import load

load_dotenv()

script_dir = Path(__file__).parent
json_path = script_dir.parent / "ch_3_Model" / "geminiLLM.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

llm = load(data)

# 최대 토큰 길이를 150개로 제한
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30, return_messages=True) 

memory.save_context(
    inputs={
        "human": "안녕하세요, 대한민국의 수도는 어디인가요?"
    },
    outputs={
        "ai": "안녕하세요! 대한민국의 수도는 서울입니다."
    },
)
memory.save_context(
    inputs={"human": "아하 그렇다면 일본의 수도는 어디인가요?"},
    outputs={
        "ai": "일본의 수도는 도쿄입니다!"
    },
),
memory.save_context(
    inputs={"human": "안녕하세요, 중국의 수도는 어디인가요?"},
    outputs={
        "ai": "중국의 수도는 베이징입니다."
    },
),
memory.save_context(
    inputs={"human": "아하 그렇다면 미국의 수도는 어디인가요?"},
    outputs={
        "ai": "미국의 수도는 워싱턴입니다!"
    },
)

# 토큰 제한을 설정하고 대화를 저장했을 때 어떻게 동작하는지 확인인
print(memory.load_memory_variables({}))

## 실행 결과
# {'history': [HumanMessage(content='아하 그렇다면 미국의 수도는 어디인가요?', additional_kwargs={}, response_metadata={}), AIMessage(content='미국의 수도는 워싱턴입니다!', additional_kwargs={}, response_metadata={})]}