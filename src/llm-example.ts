import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { generateText } from 'ai';

// LM Studioプロバイダーのインスタンスを作成
const lmstudio = createOpenAICompatible({
  name: 'lmstudio',
  baseURL: 'http://localhost:1234/v1',
});

// テキスト生成を実行する関数
async function generateResponse() {
  console.log("start generateText");
  const { text } = await generateText({
    model: lmstudio('jan-v1-4b'),
    prompt: 'hello.',
  });
  console.log(text);
}

// テキスト生成を実行
await generateResponse();
