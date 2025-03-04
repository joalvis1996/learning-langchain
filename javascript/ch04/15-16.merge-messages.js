import {
  HumanMessage,
  SystemMessage,
  AIMessage,
  mergeMessageRuns,
} from '@langchain/core/messages';

const messages = [
  new SystemMessage('you\'re a good assistant.'),
  new SystemMessage('you always respond with a joke.'),
  new HumanMessage({
    content: [{ type: 'text', text: 'i wonder why it\'s called langchain' }],
  }),
  new HumanMessage('and who is harrison chasing anyways'),
  new AIMessage(
    'Well, I guess they thought \'WordRope\' and \'SentenceString\' just didn\'t have the same ring to it!'
  ),
  new AIMessage(
    'Why, he\'s probably chasing after the last cup of coffee in the office!'
  ),
];

// 연속된 메시지를 병합
const mergedMessages = mergeMessageRuns(messages);
console.log(mergedMessages);

// 선언형 구성
const model = new ChatOpenAI()
const merger = mergeMessageRuns()
const chain = merger.pipe(model)

