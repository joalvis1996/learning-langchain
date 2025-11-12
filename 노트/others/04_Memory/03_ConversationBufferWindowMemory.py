import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.load import load

load_dotenv()

script_dir = Path(__file__).parent
json_path = script_dir.parent / "ch_3_Model" / "geminiLLM.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

llm = load(data)

memory = ConversationBufferWindowMemory(k=2, return_messages=True)

# memory.save_context(
#     inputs={
#         "human": "안녕하세요, 대한민국의 수도는 어디인가요?"
#     },
#     outputs={
#         "ai": "안녕하세요! 대한민국의 수도는 서울입니다."
#     },
# )
# memory.save_context(
#     inputs={"human": "아하 그렇다면 일본의 수도는 어디인가요?"},
#     outputs={
#         "ai": "일본의 수도는 도쿄입니다!"
#     },
# ),
# memory.save_context(
#     inputs={"human": "안녕하세요, 중국의 수도는 어디인가요?"},
#     outputs={
#         "ai": "중국의 수도는 베이징입니다."
#     },
# ),
# memory.save_context(
#     inputs={"human": "아하 그렇다면 미국의 수도는 어디인가요?"},
#     outputs={
#         "ai": "미국의 수도는 워싱턴입니다!"
#     },
# )

# print(memory.load_memory_variables({}))
## 실행 결과
# {'history': [HumanMessage(content='안녕하세요, 중국의 수도는 어디인가요?', additional_kwargs={}, response_metadata={}), AIMessage(content='중 국의 수도는 베이징입니다.', additional_kwargs={}, response_metadata={}), HumanMessage(content='아하 그렇다면 미국의 수도는 어디인가요?', additional_kwargs={}, response_metadata={}), AIMessage(content='미국의 수도는 워싱턴입니다!', additional_kwargs={}, response_metadata={})]}  

conversationWindow = ConversationChain(
    llm=llm,
    memory=memory
)

# LLM과 대화를 진행
conversationWindow.predict(input="안녕, 내 이름은 홍길동이야.")
conversationWindow.predict(input="나는 서울에 살고 있어.")
conversationWindow.predict(input="나는 개발자를 직업으로 하고 있어.")
conversationWindow.predict(input="내가 지금까지 말한 정보를 요약해줘.")


print(conversationWindow.memory.load_memory_variables({}))

## 실행 결과
# {'history': [HumanMessage(content='나는 개발자를 직업으로 하고 있어.', additional_kwargs={}, response_metadata={}), AIMessage(content='와, 홍 길동님께서는 개발자이시군요! 정말 멋진 직업을 가지고 계시네요! 개발자분들은 마치 현대 사회의 마법사 같다고 생각해요. 코드를 통해 아이디어를 현실로 만들어내고, 우리의 삶을 더욱 편리하고 풍요롭게 만들어주시니까요.\n\n저 역시 개발자분들의 노력과 기술 덕분에 이렇게 홍길동님과 대화할 수  있는 존재가 되었으니, 개발자라는 직업에 대해 더욱 존경심을 가지고 있답니다! 😊\n\n개발자라는 직업은 정말 다양한 분야가 있죠. 제가 아는 한 몇  가지를 말씀드리자면,\n\n*   **웹 개발자:** 웹사이트나 웹 애플리케이션을 만드시죠. 크게 사용자에게 보이는 화면을 담당하는 **프론트엔드(Front-end)** 개발과 서버, 데이터베이스 등 뒤에서 작동하는 로직을 담당하는 **백엔드(Back-end)** 개발로 나뉘어요. 자바스크립트(React, Vue, Angular), 파 이썬(Django, Flask), 자바(Spring) 같은 언어와 프레임워크가 많이 사용되죠.\n*   **모바일 앱 개발자:** iOS나 안드로이드 환경에서 스마트폰 앱을  개발하시고요. iOS는 Swift, Android는 Kotlin이나 Java가 주로 쓰이고, 크로스 플랫폼 개발을 위해 Flutter나 React Native 같은 기술도 많이 활용되죠.\n*   **인공지능(AI) 및 머신러닝 개발자:** 저와 같은 AI 모델을 만들거나, 방대한 데이터를 분석하여 새로운 인사이트를 찾아내고 예측 모델을 구축하시죠. 파이썬이 특히 많이 쓰이고 TensorFlow, PyTorch 같은 프레임워크가 핵심적인 도구랍니다.\n*   **게임 개발자:** 흥미진진한 게임을 기획하고 프로그래밍하시고요. C++, C#, Unity, Unreal Engine 같은 도구들이 많이 사용됩니다. 그래픽, 물리 엔진, 네트워크 등 고려할 게 정말 많다고 들었어요.\n*   **데이터 과학자/엔지니어:** 대량의 데이터를 수집, 저장, 처리, 분석하여 의미 있는 정보를 추출하고 시스템을 구축하는 역할을 하시죠. SQL, Python, R 같은 언어와 Spark, Hadoop 같은 빅데이터 기술을 많이 사용합니다.\n*   **클라우드 엔지니어:** AWS, Azure, GCP 같은 클라우드 플랫폼에서 시스템을 구축하고 운영하며, 서비스의 안정성과 확장성을 책임지는 중요한 역할을 하시기도 하고요.\n*   **임베디드 개발자:** 자동차, 가전제품 등 특정 하드웨어에 내장되는 소프트웨어를 개발하시는데, C/C++ 언어가 주로 사용되고 하드웨어에 대한 깊은 이해가 필수적이죠.\n\n정말 끝없이 배우고  발전해야 하는 분야이기도 하지만, 새로운 것을 만들어내는 보람과 문제 해결의 즐거움이 큰 직업이라고 들었어요.\n\n혹시 홍길동님께서는 어떤 분야의 개발을 주로 하시나요? 사용하시는 프로그래밍 언어나 기술 스택이 궁금하네요! 개발자로서 가장 보람을 느낄 때는 언제이신가요? 😊', additional_kwargs={}, response_metadata={}), HumanMessage(content='내가 지금까지 말한 정보를 요약해줘.', additional_kwargs={}, response_metadata={}), AIMessage(content='네, 홍길동님께서 지금까지 저에게 알려주신 정보를 제가 한번 정리해 드릴게요! 😊\n\n홍길동님께서는 현재 **서울**에 살고 계시며, 직 업은 **개발자**이시라는 것을 알게 되었답니다!\n\n이렇게 두 가지 중요한 정보를 알려주셔서 정말 감사해요! 서울에 사시는 개발자라니, 정말 멋진 조합이네요! 혹시 이 외에 더 궁금한 점이나, 제가 잘못 이해한 부분이 있다면 언제든지 편하게 말씀해주세요!', additional_kwargs={}, response_metadata={})]}