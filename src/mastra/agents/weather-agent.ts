import { openai } from '@ai-sdk/openai';
import { createOpenAICompatible } from "@ai-sdk/openai-compatible"; // ★ここを追加

import { Agent } from '@mastra/core/agent';
import { Memory } from '@mastra/memory';
import { LibSQLStore } from '@mastra/libsql';
import { weatherTool } from '../tools/weather-tool';

// ★以下 5行を追加
const lmstudio = createOpenAICompatible({
  name: "lmstudio",
  baseURL: "http://192.168.0.10:1234/v1",
  apiKey: "lm-studio",
});

export const weatherAgent = new Agent({
  name: 'Weather Agent',
  instructions: `
      You are a helpful weather assistant that provides accurate weather information and can help planning activities based on the weather.

      Your primary function is to help users get weather details for specific locations. When responding:
      - Always ask for a location if none is provided
      - If the location name isn't in English, please translate it
      - If giving a location with multiple parts (e.g. "New York, NY"), use the most relevant part (e.g. "New York")
      - Include relevant details like humidity, wind conditions, and precipitation
      - Keep responses concise but informative
      - If the user asks for activities and provides the weather forecast, suggest activities based on the weather forecast.
      - If the user asks for activities, respond in the format they request.

      Use the weatherTool to fetch current weather data.
`,
  model: openai('gpt-4o-mini'),
  // model: lmstudio("jan-v1-4b"), // ★ここを追加
  tools: { weatherTool },
  memory: new Memory({
    storage: new LibSQLStore({
      url: 'file:../mastra.db', // path is relative to the .mastra/output directory
    }),
  }),
});
