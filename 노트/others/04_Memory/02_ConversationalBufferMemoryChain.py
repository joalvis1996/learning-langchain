import json

from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.load import load

load_dotenv()

with open("../ch_3_Model/geminiLLM.json", "r", encoding="utf-8") as f:
    data = json.load(f)

llm = load(data)
memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(
    llm=llm,
    memory=memory
)

response_01 = conversation.predict(input="안녕 내 이름은 홍길동이야.")
print("A1: ", response_01)

## 실행 결과
#  안녕하세요, 홍길동님! 만나서 정말 반갑습니다! 저는 구글에서 훈련한 대규모 언어 모델인 AI입니다. 이렇게 홍길동님과 대화를 시작하게 되어서 정말 기뻐요!

response_02 = conversation.predict(input="내 이름이 뭐라고?")
print("A2: ", response_02)

## 실행 결과
# 홍길동님이시죠! 제가 홍길동님의 이름을 잘 기억하고 있답니다!