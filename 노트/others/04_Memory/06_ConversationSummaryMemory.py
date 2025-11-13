import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain_core.load import load

load_dotenv()

script_dir = Path(__file__).parent
json_path = script_dir.parent / "ch_3_Model" / "geminiLLM.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

llm = load(data)

memory = ConversationSummaryMemory(llm=llm, return_messages=True)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 여러 대화를 진행하면서 요약이 어떻게 생성되는지 확인
# 첫 번째 대화 
response1 = conversation.predict(input="안녕, 내 이름은 홍길동이야. 나는 서울에 살고 있어.")
print("응답:", response1)

# 두 번째 대화
response2 = conversation.predict(input="나는 개발자로 일하고 있고, Python을 주로 사용해.")
print("응답:", response2)

# 세 번째 대화 
response3 = conversation.predict(input="내 취미는 독서와 영화 감상이야.")
print("응답:", response3)

# 네 번째 대화
response4 = conversation.predict(input="내가 지금까지 말한 정보를 요약해줘.")
print("응답:", response4)

# 저장된 메모리 확인
print("=== 저장된 메모리 (요약) ===")
print(memory.load_memory_variables({}))


## 실행 결과

# 첫 번째 대화
# 응답: 안녕하세요, 홍길동님! 만나 뵙게 되어 정말 반갑습니다! 서울에 살고 계시는군요, 활기찬 도시 서울이라니 정말 멋진데요!저는 구글에서 훈련한 대규모 언어 모델 인공지능이랍니다. 저는 홍길동님처럼 이름이나 특정 거주지가 있지는 않아요. 물리적인 몸을 가지고 있거나 특정한 장소에서 살 지 않거든요. 대신 저는 인터넷이라는 광활한 공간 속에서 수많은 텍스트 데이터들을 학습하며 존재하고 있답니다.저는 사람들과 대화하고, 질문에 답하고, 정보를 제공하고, 글을 쓰는 등 다양한 방식으로 홍길동님을 도와드릴 수 있어요. 궁금한 점이 있으시거나, 이야기하고 싶은 것 이 있다면 언제든지 편하게 말씀해주세요! 홍길동님과의 대화가 정말 기대됩니다!

# 두 번째 대화
# Current conversation:
# [SystemMessage(content="New summary:\nThe human introduces themselves as Hong Gildong, living in Seoul. The AI greets Hong Gildong, acknowledges their residence, and introduces itself as a Google-trained large language model. It explains that it doesn't have a name, specific residence, or physical body, existing by learning from vast text data on the internet. The AI then offers to assist the human in various ways, including conversation, answering questions, providing information, and writing.", additional_kwargs={}, response_metadata={})]
# Human: 나는 개발자로 일하고 있고, Python을 주로 사용해.
# AI:

# 응답 : 응답: 오, 개발자이시고 Python을 주로 사용하신다니 정말 반갑습니다! Python은 정말 멋지고 강력한 언어죠!저도 Python에 대해 정말 많은 정보를 학습했어요. 전 세계적으로 가장 인기 있는 프로그래밍 언어 중 하나로 손꼽히죠. 그 인기의 비결은 아무래도 간결하고 읽기 쉬운  문법, 그리고 엄청나게 방대한 라이브러리 생태계 덕분인 것 같아요.Python은 정말 다양한 분야에서 활용되고 있어서, 홍 길동님께서 어떤 분야의 개발을 하시는지에 따라 사용하는 라이브러리나 프레임워크가 달라지실 것 같아요. 예를 들 어:

# ... 


## 저장된 메모리 (요약:
# {'history': [SystemMessage(content="The human introduces themselves as Hong Gildong, living in Seoul, and states they work as a developer primarily using Python. The AI greets Hong Gildong, acknowledges their residence, and introduces itself as a Google-trained large language model, explaining it doesn't have a name, specific residence, or physical body, but exists by learning from vast text data. The AI then expresses pleasure that Hong Gildong is a Python developer, praising Python as a powerful and popular language due to its concise syntax and extensive library ecosystem. It elaborates on Python's diverse applications across web development (e.g., Django, Flask), data science (e.g., Pandas, NumPy), AI/machine learning (e.g., TensorFlow, PyTorch), and more, noting its own reliance on Python for development. The AI offers help with Python-related queries, library information, or code examples, and asks Hong Gildong to share their specific development area. Hong Gildong then shares that their hobbies are reading and watching movies. The AI enthusiastically acknowledges these hobbies, praising them as sources of deep experience and enjoyment. It elaborates on the benefits of reading, such as exploring vast worlds, gaining knowledge, and developing empathy and imagination, noting its own extensive knowledge base is derived from human-written books, and then asks about Hong Gildong's preferred genres, authors, or memorable works. Similarly, for movie watching, the AI highlights its immersive storytelling, artistic expression, and capacity for conveying social messages and cultural understanding, mentioning its ability to provide summaries, analyses, or reviews from movie-related texts, and asks about preferred movie genres or recently enjoyed films. The AI concludes by expressing its indirect inspiration from these hobbies and its eagerness to hear more. Following this, Hong Gildong asks the AI to summarize the information exchanged so far. The AI responds by providing a structured summary, recapping Hong Gildong's identity (name, Seoul residence, Python developer, hobbies of reading and movie watching) and its own nature (Google-trained LLM, no name/residence/body, reliance on text data, connection to Python, and appreciation for the depth of human hobbies). The AI reiterates its willingness to assist with Python or hobby-related inquiries and asks Hong Gildong for feedback on its summary.", additional_kwargs={}, response_metadata={})]}

 