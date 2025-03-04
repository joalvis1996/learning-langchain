from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)
from langchain_openai import ChatOpenAI

# 동일한 유형의 연속된 메시지 예제
messages = [
    SystemMessage(content='당신은 친절한 어시스턴트입니다.'),
    SystemMessage(content='항상 농담으로 대답하세요.'),
    HumanMessage(
        content=[{'type': 'text', 'text': '왜 langchain이라고 부르는지 모르겠어요.'}]
    ),
    HumanMessage(content='그리고 해리슨 체이스는 누구를 쫓는 건가요'),
    AIMessage(
        content='음, 아마도 \'WordRope\'와 \'SentenceString\'은 그만큼 울림이 없다고 생각했을 겁니다!'
    ),
    AIMessage(
        content='왜냐하면 그는 아마도 사무실에 남은 마지막 커피잔을 쫓을 테니까요!'
    ),
]

# 연속된 메시지를 병합
merged = merge_message_runs(messages)
print(merged)

# 선언형 구성
model = ChatOpenAI(model='gpt-4o-mini')
merger = merge_message_runs()
chain = merger | model
