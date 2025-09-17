
import { Mastra } from '@mastra/core/mastra';
import { PinoLogger } from '@mastra/loggers';
import { LibSQLStore } from '@mastra/libsql';
import { weatherWorkflow } from './workflows/weather-workflow';
import { weatherAgent } from './agents/weather-agent';
import { LibSQLVector } from "@mastra/libsql";
import { researchAgent } from "./agents/research-agent";

const libSqlVector = new LibSQLVector({
  connectionUrl: "file:/home/kk/workspace/mastra-practice/vector.db",
});

export const mastra = new Mastra({
  vectors: { libSqlVector },
  workflows: { weatherWorkflow },
  agents: { weatherAgent, researchAgent },
  storage: new LibSQLStore({
    // stores telemetry, evals, ... into memory storage, if it needs to persist, change to file:../mastra.db
    url: ":memory:",
  }),
  logger: new PinoLogger({
    name: 'Mastra',
    level: 'info',
  }),
});
