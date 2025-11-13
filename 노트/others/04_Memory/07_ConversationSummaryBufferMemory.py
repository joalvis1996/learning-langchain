import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory.prompt import SUMMARY_PROMPT
from langchain_core.load import load

load_dotenv()

script_dir = Path(__file__).parent
json_path = script_dir.parent / "ch_3_Model" / "geminiLLM.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

llm = load(data)

# max_token_limit: 버퍼에 저장할 최대 토큰 수
memory = ConversationSummaryBufferMemory(
    llm=llm, 
    max_token_limit=500,  # 버퍼에 최대 100 토큰까지 저장
    return_messages=True,
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True  # 요약 과정을 확인하기 위해 verbose=True 설정
)

# 여러 대화를 진행하면서 어떻게 동작하는지 확인
print("=== 첫 번째 대화 ===")
response1 = conversation.predict(input="안녕, 내 이름은 홍길동이야. 나는 서울에 살고 있어.")
print("응답:", response1)

print("=== 두 번째 대화 ===")
response2 = conversation.predict(input="나는 개발자로 일하고 있고, Python을 주로 사용해.")
print("응답:", response2)

print("=== 세 번째 대화 ===")
response3 = conversation.predict(input="내 취미는 독서와 영화 감상이야.")
print("응답:", response3)

print("=== 네 번째 대화 ===")
response4 = conversation.predict(input="나는 최근에 Django로 웹 프로젝트를 시작했어.")
print("응답:", response4)

# 저장된 메모리 확인
print("=== 저장된 메모리 (요약 + 최근 대화) ===")
print(memory.load_memory_variables({}))


## 실행 결과 예시:
# {'history': [SystemMessage(content='Current summary:\nThe human introduces themselves as Hong Gildong, living in Seoul, and reveals they work as a developer primarily using Python. The AI greets Hong Gildong, noting their location, and introduces itself as a Google-trained Large Language Model, explaining its purpose is to process information, answer questions, and generate text based on its vast learned data, as it lacks a physical body, name, or home. The AI expresses enthusiasm for Hong Gildong\'s profession, highlighting Python\'s critical role in AI development (including its own training) and its own extensive understanding of the language, then asks about their specific Python development focus or favorite libraries/frameworks. Hong Gildong initially shares their hobbies of reading and watching movies, to which the AI expresses appreciation and explains its extensive knowledge of both, offering to engage in deep discussions or recommendations. Hong Gildong then reveals they recently started a web project using Django. The AI expresses excitement and praises Django as an an excellent choice, reiterating its deep understanding of Python and Django\'s status as a powerful, full-stack web framework. It explains Django\'s "Perfectionists with deadlines" and "Batteries included" philosophies, detailing how it provides built-in features like ORM, admin site, authentication, and templating to enable fast, efficient development by allowing developers to focus on core business logic. The AI confirms its extensive knowledge of Django\'s core components (Models, Views, Templates), advanced topics (Forms, Middleware, Caching, Security), and Django REST Framework (DRF). It then asks Hong Gildong for more details about the specific type of web application they are building (e.g., blog, e-commerce), any interesting or challenging aspects, or planned features, to provide more tailored assistance. Hong Gildong then asks the AI to summarize the information exchanged so far. The AI obliges, providing a detailed summary that reiterates Hong Gildong\'s identity as a Python developer named Hong Gildong residing in Seoul, who enjoys reading and movies, and recently started a Django web project. The AI also summarizes its own identity as a Google-trained LLM with deep knowledge in Python, Django, reading, and movies, capable of assisting with the project, and then asks for confirmation of its summary\'s accuracy.', additional_kwargs={}, response_metadata={})]}