import { mastra } from "./mastra";
const agent = mastra.getAgent("researchAgent");
 
// 基本的な概念に関するクエリ
const query1 =
  "What problems does sequence modeling face with neural networks?";
const response1 = await agent.generate(query1);
console.log("\nQuery:", query1);
console.log("Response:", response1.text);