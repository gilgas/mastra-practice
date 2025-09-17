// import { openai } from '@ai-sdk/openai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { Mastra } from '@mastra/core';
import { Agent } from '@mastra/core/agent';
import { PgVector } from '@mastra/pg';
import { MDocument, createGraphRAGTool } from '@mastra/rag';
import { embedMany } from 'ai';
import { LibSQLVector } from "@mastra/libsql";

// LM Studioプロバイダーのインスタンスを作成
const lmstudio = createOpenAICompatible({
  name: 'lmstudio',
  baseURL: 'http://localhost:1234/v1',
});

const graphRagTool = createGraphRAGTool({
  vectorStoreName: 'pgVector',
  indexName: 'hidamari_local',
  // model: openai.embedding('text-embedding-3-small'),
  model: lmstudio.textEmbeddingModel('text-embedding-multilingual-e5-large-instruct'),
  graphOptions: {
    // dimension: 1536,
    dimension: 1024,
    threshold: 0.7,
  },
});

export const ragAgent = new Agent({
  name: 'GraphRAG Agent',
  instructions: `You are a helpful assistant that answers questions based on the provided context. Format your answers as follows:

1. DIRECT FACTS: List only the directly stated facts from the text relevant to the question (2-3 bullet points)
2. CONNECTIONS MADE: List the relationships you found between different parts of the text (2-3 bullet points)
3. CONCLUSION: One sentence summary that ties everything together

Keep each section brief and focus on the most important points.

Important: When asked to answer a question, please base your answer only on the context provided in the tool. 
If the context doesn't contain enough information to fully answer the question, please state that explicitly.`,
  // model: openai('gpt-4o-mini'),
  model: lmstudio('jan-v1-4b'),
  tools: {
    graphRagTool,
  },
});

const pgVector = new PgVector({ connectionString: process.env.POSTGRES_CONNECTION_STRING! });
const libsqlVector = new LibSQLVector({
  connectionUrl: ":memory:",

});

export const mastra = new Mastra({
  agents: { ragAgent },
  vectors: { pgVector },
});

const doc2 = MDocument.fromText(`
### ひだまりスケッチ：作品概要と登場人物

#### 作品概要
『ひだまりスケッチ』は、蒼樹うめによる日本の4コマ漫画、およびそれを原作とするアニメシリーズである。物語は、主人公・ゆのがやまぶき高校美術科に入学し、通称「ひだまり荘」と呼ばれるアパートで個性豊かな友人たちと共同生活を送る日常を描いている。作品の主要なテーマは、友情、成長、そして何気ない日常の中にある小さな幸せの発見である。アニメ版はシャフトによって制作され、独特の演出スタイルや、季節の移り変わりを丁寧に描いた美しい背景描写が特徴である。

#### 主要登場人物

* **ゆの**: 主人公。やまぶき高校美術科の1年生（物語開始時）。小柄で、少し内気な性格だが、感受性が豊かで何事にも真面目に取り組む。美術の腕はまだ発展途上だが、ひたむきに努力する。絵を描くことが大好きで、よくスケッチをしている。
* **宮子（みやこ）**: ゆのと同級生で、ひだまり荘の隣の部屋に住んでいる。明るく元気で、大食い。天才的なひらめきで周囲を驚かせることがあるが、時々突拍子もない行動をとる。
* **ヒロ**: ゆのや宮子の一つ上の先輩で、ひだまり荘に住んでいる。家庭的な性格で、料理が得意。皆のお姉さん的存在だが、自身の体型を気にしている一面もある。
* **沙英（さえ）**: ヒロと同級生。小説家を目指しており、ペンネームで雑誌に作品を掲載している。クールで知的だが、実は恋話には弱い。

#### アニメ版の特徴
アニメシリーズでは、原作の4コマ漫画をベースにしつつ、時間の流れや季節感を重視したエピソード構成が取られている。各話のタイトルは、季節や出来事を象徴する「〜日」という形式で統一されていることが多い。また、新房昭之が監督を務めたことで知られ、独特のフォントや映像効果、カット割りなど、実験的な演出が多く取り入れられている。

#### 関連作品と影響
原作の連載は2004年から始まり、アニメは2007年に第1期が放送されて以降、複数期にわたって制作された。劇場版や特別編も発表されており、長期にわたりファンに愛されている。この作品は、その後の日常系アニメに大きな影響を与え、登場人物たちの何気ない会話や、美術を通して表現される感情の機微を丁寧に描くスタイルは、多くの追随作品を生んだ。
`);

const chunks = await doc2.chunk({
  strategy: 'recursive',
  maxSize: 512,
  overlap: 50,
  separators: [ '\n' ],
});

/**
 * 情報追加
 */

const { embeddings } = await embedMany({
  // model: openai.embedding('text-embedding-3-small'),
  model: lmstudio.textEmbeddingModel('text-embedding-multilingual-e5-large-instruct'),
  values: chunks.map(chunk => chunk.text),
});

const vectorStore = mastra.getVector('pgVector');
// const vectorStore = mastra.getVector('libsqlVector');
await vectorStore.createIndex({
  indexName: 'hidamari_local',
  // dimension: 1536,
  dimension: 1024,
});
await vectorStore.upsert({
  indexName: 'hidamari_local',
  vectors: embeddings,
  metadata: chunks?.map((chunk: any) => ({ text: chunk.text })),
});

/**
 * 質問
 */
// const queryOne =
//   "What are the direct and indirect effects of early railway decisions on Riverdale Heights' current state?";
// const answerOne = await ragAgent.generate(queryOne);
// console.log('\nQuery:', queryOne);
// console.log('Response:', answerOne.text);

// const queryTwo =
//   'How have changes in transportation infrastructure affected different generations of local businesses and community spaces?';
// const answerTwo = await ragAgent.generate(queryTwo);
// console.log('\nQuery:', queryTwo);
// console.log('Response:', answerTwo.text);

// const queryThree =
//   'Compare how the Rossi family business and Thompson Steel Works responded to major infrastructure changes, and how their responses affected the community.';
// const answerThree = await ragAgent.generate(queryThree);
// console.log('\nQuery:', queryThree);
// console.log('Response:', answerThree.text);

// const queryFour =
//   'Trace how the transformation of the Thompson Steel Works site has influenced surrounding businesses and cultural spaces from 1932 to present.';
// const answerFour = await ragAgent.generate(queryFour);
// console.log('\nQuery:', queryFour);
// console.log('Response:', answerFour.text);

const queryFive =
  'ヒロと沙英はどんな関係？';
const answerFour = await ragAgent.generateLegacy(queryFive);
console.log('\nQuery:', queryFive);
console.log('Response:', answerFour.text);
