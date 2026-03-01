import fs from "fs";
import http from "http";
import https from "https";
import { z } from "zod";
type Cookies = Record<string, string | undefined>;

function parseCookie(str: string): Cookies {
  const cookies: Cookies = {};
  for (const pair of str.split('; ')) {
    const eqIdx = pair.indexOf('=');
    if (eqIdx < 1) continue;
    cookies[decodeURIComponent(pair.slice(0, eqIdx))] = decodeURIComponent(pair.slice(eqIdx + 1));
  }
  return cookies;
}
import GPT3Tokenizer from "gpt3-tokenizer";
import crypto from "crypto";
import secrets from "./secrets.json" with { type: "json" };

const MAX_TOKENS = 4097;
const TOKENS_SAFETY_MARGIN = 25;
const tokenizer = new GPT3Tokenizer.default({ type: "gpt3" });

const origins = [
  process.env.FRONTEND_URL_1,
  process.env.FRONTEND_URL_2,
].flatMap((f) => (f ? [f] : []));

console.log("origins", origins);

const port = process.env.PORT ?? 3000;
const httpPort = process.env.HTTP_PORT ?? 8080;

const MODEL_TO_PROVIDER = {
  "mistralai/Mixtral-8x7B-Instruct-v0.1": "deepinfra",
  "meta-llama/Meta-Llama-3.1-405B-Instruct": "deepinfra",
  "meta-llama/Llama-3-70b-chat-hf": "together",
  "anthropic/claude-3-opus:beta": "openrouter",
  "anthropic/claude-3.5-sonnet": "openrouter",
  "mistralai/mistral-large": "openrouter",
  "deepseek/deepseek-coder": "openrouter",
  "gpt-3.5-turbo-instruct": "openai",
  "gpt-3.5-turbo": "openai",
  "gpt-4": "openai",
  "gpt-4-1106-preview": "openai",
  "gpt-4.1": "openai",
  "gpt-4.1-mini": "openai",
  "gpt-4.1-nano": "openai",
  "gpt-5": "openai",
  "gpt-5-chat-latest": "openai",
  "gpt-4o-mini": "openai",
  "gpt-4o": "openai",
  "o1-preview": "openai",
  "o1-mini": "openai",
} as const;
type Model = keyof typeof MODEL_TO_PROVIDER;
const MODELS = Object.keys(MODEL_TO_PROVIDER) as Model[];

interface TopLogProb {
  token: string;
  logprob: number;
  selected?: boolean;
}

interface LogProbContent {
  token: string;
  logprob: number;
  bytes: number[];
  top_logprobs: TopLogProb[];
}

interface Logprobs {
  content: LogProbContent[];
  refusal: string | null;
}

interface Choice {
  text: string;
  index: number;
  logprobs?: Logprobs;
  finish_reason: string;
}

const MessageSchema = z.object({
  text: z.string(),
  name: z.enum(["You", "Bot"]),
  party: z.enum(["bot", "human"]),
  id: z.string(),
});
type Message = z.infer<typeof MessageSchema>;

const MessagesSchema = z.array(MessageSchema);

const BodySchema = z.object({
  messages: MessagesSchema,
  model: z.enum(MODELS),
});

const CookiesSchema = z.object({
  "__Secure-authKey": z.string().optional(),
});

interface Data {
  id: string;
  object: string;
  created: number;
  choices: Choice[];
  model: string;
}

interface ChatData {
  id: string;
  object: string;
  created: number;
  model: string;
  choices?: ChatChoice[];
}

interface ChatChoice {
  delta: ChatDelta;
  index: number;
  finish_reason: string;
}

interface ChatDelta {
  content: string;
}

function chunkToDataArray<T = Data>(chunkString: string): T[] {
  const dataLines = chunkString.split("\n\n");
  const dataArray: T[] = [];
  for (let i = 0; i < dataLines.length; i++) {
    const dataLine = dataLines[i].trim();
    if (dataLine.startsWith("data: ")) {
      if (dataLine === "data: [DONE]") {
        return dataArray;
      }
      const dataString = dataLine.slice("data: ".length);
      const data = JSON.parse(dataString);
      dataArray.push(data);
    }
  }
  return dataArray;
}

function generatePrompt(messages: Message[]) {
  messages = messages.slice(-25); // sliding window

  let prompt = `Hello, I am a chatbot powered by GPT-3. You can ask me anything and I will try my best to answer your questions.

To format my responses with code blocks, you can use the following markdown syntax:

\`\`\`
Your code goes here
\`\`\`

To format my responses with inline code, you can use the following markdown syntax:

\`Your code goes here\`

Feel free to ask me anything and I will do my best to help.

`;

  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    const { party, text } = message;
    if (party === "human") {
      prompt += `Human: ${text.trim()}\n\n`;
    } else if (party === "bot") {
      prompt += `Bot: ${text.trim()}\n\n`;
    }
  }
  // prompt for the bot
  prompt += "Bot: ";
  return prompt;
}


class DoubleNewlineReader {
  reader: ReadableStreamDefaultReader<ArrayBuffer>;
  buffer: string;
  decoder: InstanceType<typeof TextDecoder>;

  constructor(reader: ReadableStreamDefaultReader<ArrayBuffer>) {
    this.reader = reader;
    this.buffer = '';
    this.decoder = new TextDecoder();
  }

  async readUntilDoubleNewline() {
    while (true) {
      const pos = this.buffer.indexOf('\n\n');
      if (pos !== -1) {
        const upToDoubleNewline = this.buffer.substring(0, pos + 2);
        this.buffer = this.buffer.substring(pos + 2);
        return { done: false, value: upToDoubleNewline };
      }
      const { done, value } = await this.reader.read();
      if (done) {
        if (this.buffer.length) {
          const remaining = this.buffer;
          this.buffer = '';
          return { done: true, value: remaining };  // Return what's left if the stream is done
        }
        break;
      }
      const dataString = this.decoder.decode(value).replace(/\r\n/g, "\n");
      this.buffer += dataString;  // Assuming value is a string; adjust if not
    }
    return { done: true, value: this.buffer };
  }
}

function timeSafeCompare(a: string, b: string) {
  // Brad Hill's Double HMAC pattern
  const key = crypto.randomBytes(32);
  const ah = crypto.createHmac('sha256', key).update(a).digest();
  const bh = crypto.createHmac('sha256', key).update(b).digest();
  return crypto.timingSafeEqual(ah, bh) && a === b;
}

interface ModelConfig {
  apiType: 'chat' | 'instruct'
  systemMessage: 'default' | 'custom'
  bearerToken: string | undefined
  apiUrl: string
  stop: string | undefined
  authed: boolean
}

const AUTHED_MODELS: Set<Model> = new Set([
  "gpt-4",
  "gpt-4.1",
  "gpt-4.1-mini",
  "gpt-4.1-nano",
  "gpt-5",
  "gpt-5-chat-latest",
  "o1-preview",
  "o1-mini",
]);

function getModelConfig(model: Model): ModelConfig {
  const provider = MODEL_TO_PROVIDER[model]
  const authed = AUTHED_MODELS.has(model);
  switch (provider) {
    case "deepinfra":
      return {
        apiType: 'chat',
        systemMessage: 'default',
        bearerToken: secrets.DEEPINFRA_BEARER_TOKEN,
        apiUrl: "https://api.deepinfra.com/v1/openai/chat/completions",
        stop: "END_OF_STREAM",
        authed,
      }
    case "together":
      return {
        apiType: 'chat',
        systemMessage: 'default',
        bearerToken: secrets.TOGETHER_BEARER_TOKEN,
        apiUrl: "https://api.together.xyz/v1/chat/completions",
        stop: "<|eot_id|>",
        authed,
      }
    case "openrouter":
      return {
        apiType: 'chat',
        systemMessage: 'default',
        bearerToken: secrets.OPENROUTER_BEARER_TOKEN,
        apiUrl: "https://openrouter.ai/api/v1/chat/completions",
        stop: "END_OF_STREAM",
        authed,
      }
    case "openai":
      return {
        apiType: model === "gpt-3.5-turbo-instruct" ? 'instruct' : 'chat',
        systemMessage: model === "o1-preview" || model === "o1-mini" ? 'default' : 'custom',
        bearerToken: secrets.BEARER_TOKEN,
        apiUrl: model === "gpt-3.5-turbo-instruct" ? "https://api.openai.com/v1/completions" : "https://api.openai.com/v1/chat/completions",
        stop: model === "o1-preview" || model === "o1-mini" || model === "gpt-5" ? undefined : "END_OF_STREAM",
        authed,
      }
  }
}

async function streamChatCompletion(onChunk: (content: string) => void, authKey: string | undefined, messages: Message[], model: Model, modelConfig: ModelConfig) {
  const { systemMessage, bearerToken, stop, apiUrl, authed } = modelConfig;

  if (authed && !timeSafeCompare(authKey ?? "", secrets.AUTH_KEY ?? "")) {
    throw new Error("Invalid auth key");
  }
  const chatMessages = [
    ...(systemMessage === 'custom' ? [{
      role: "system",
      content: "You are a helpful AI language model assistant.",
    }] : []),
    ...messages.map((m) => ({
      role: m.party === "human" ? "user" : "assistant",
      content: m.text,
    })),
  ];
  const options = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${bearerToken}`,
    },
    body: JSON.stringify({
      model,
      messages: chatMessages,
      stream: model !== "o1-preview" && model !== "o1-mini" && model !== "gpt-5",
      stop,
    }),
  };
  const response = await fetch(
    apiUrl,
    options
  );
  if (!response.ok) {
    const text = await response.text();
    // TODO: send 4xx/5xx status code and don't put error object in text
    throw new Error(`HTTP error! status: ${response.status}. text ${text}`);
  }
  if (!response.body) {
    throw new Error("No response body");
  }

  if (model === "o1-preview" || model === "o1-mini" || model === "gpt-5") {
    const result = await response.text()
    const data = JSON.parse(result)
    const completion = data.choices[0].message.content
    onChunk(completion);
    console.log("completion", completion);
    return
  }

  const reader = response.body.getReader();
  try {
    const doubleNewlineReader = new DoubleNewlineReader(reader);
    let completion = "";
    while (true) {
      const { done, value: dataString } = await doubleNewlineReader.readUntilDoubleNewline();
      if (done) {
        break;
      }
      const dataArray = chunkToDataArray<ChatData>(dataString);

      for (let i = 0; i < dataArray.length; i++) {
        const data = dataArray[i];
        const { choices } = data;
        if (!choices) {
          continue
        }
        const lastChoice = choices[choices.length - 1];
        const { delta } = lastChoice;
        const content = delta.content ?? "";
        onChunk(content);
        completion += content;
      }
    }
    console.log("completion", completion);
  } catch (error) {
    await reader.cancel();
    throw error;
  }
  await reader.cancel();
}

async function streamInstructCompletion(onChunk: (content: string) => void, messages: Message[], model: Model, modelConfig: ModelConfig) {
  const { bearerToken, apiUrl } = modelConfig;

  const prompt = generatePrompt(messages);
  const encoded: { bpe: number[]; text: string[] } =
    tokenizer.encode(prompt);
  const promptTokens = encoded.bpe.length;

  if (promptTokens >= MAX_TOKENS - TOKENS_SAFETY_MARGIN) {
    throw new Error("Too many tokens.");
  }

  const temperature = 0.5;
  const options = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${bearerToken}`,
    },
    body: JSON.stringify({
      model,
      prompt,
      temperature,
      max_tokens: MAX_TOKENS - promptTokens - TOKENS_SAFETY_MARGIN,
      stream: true,
      stop: "END_OF_STREAM",
    }),
  };
  const response = await fetch(
    apiUrl,
    options
  );

  if (!response.ok) {
    const text = await response.text();
    // TODO: send 4xx/5xx status code and don't put error object in text
    throw new Error(`HTTP error! status: ${response.status}. text ${text}`);
  }
  if (!response.body) {
    throw new Error("No response body");
  }

  const reader = response.body.getReader();
  try {
    const doubleNewlineReader = new DoubleNewlineReader(reader);
    let completion = "";
    while (true) {
      const { done, value: dataString } = await doubleNewlineReader.readUntilDoubleNewline();
      if (done) {
        break;
      }

      // Convert the binary data to a string
      const dataArray = chunkToDataArray(dataString);

      for (let i = 0; i < dataArray.length; i++) {
        const data = dataArray[i];
        const token = data.choices[0].text;
        onChunk(token);
        completion += token;
      }
    }
    console.log("completion", completion.trim());
  } catch (error) {
    await reader.cancel();
    throw error;
  }
  await reader.cancel();
}

async function postGenerateChatCompletionStreaming(reqCookies: Cookies, res: http.ServerResponse, reqBody: string) {
  try {
    const cookies = CookiesSchema.parse(reqCookies);
    const parsed = JSON.parse(reqBody);
    const body = BodySchema.parse(parsed);
    const authKey = cookies["__Secure-authKey"];
    const messages = body.messages;
    const model = body.model;
    const lastHumanMessage = messages.findLast((m) => m.party === "human");
    if (messages[0]) {
      if (messages[0].party !== "human") {
        throw new Error("Validation error");
      }
    }

    if (!lastHumanMessage) {
      throw new Error("Validation error: no human message found");
    }

    console.log("model", model);
    console.log("human-prompt", lastHumanMessage.text);

    const modelConfig = getModelConfig(model);

    res.setHeader("Transfer-Encoding", "chunked");
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    const onChunk = (content: string) => res.write(content);
    const { apiType } = modelConfig;
    if (apiType === 'chat') {
      await streamChatCompletion(onChunk, authKey, messages, model, modelConfig);
    } else if (apiType === 'instruct') {
      await streamInstructCompletion(onChunk, messages, model, modelConfig);
    }
    res.end();
  } catch (error) {
    console.error("error", error);
    try {
      if (!res.headersSent) {
        res.setHeader("Content-Type", "application/json");
        res.write(JSON.stringify({
          success: false,
          error: { message: (error as Error).message },
        }));
      } else if (!res.writableEnded) {
        res.end();
      }
    } catch (e) {
      console.error("e", e);
      // do nothing
    } finally {
      if (!res.writableEnded) {
        res.end();
      }
    }
  }
};

function postIsAuthed(reqCookies: Cookies, res: http.ServerResponse) {
  try {
    const cookies = CookiesSchema.parse(reqCookies);
    const authKey = cookies["__Secure-authKey"];
    res.write(JSON.stringify({
      success: true,
      isAuthed: timeSafeCompare(authKey ?? "", secrets.AUTH_KEY ?? ""),
    }));
  } catch (error) {
    console.error("error", error);
    try {
      res.setHeader("Content-Type", "application/json");
      res.write(JSON.stringify({
        success: false,
        error: { message: (error as Error).message },
      }));
    } catch (e) {
      console.error("e", e);
      // do nothing
    }
  } finally {
    res.end();
  }
}

const setCors = (req: http.IncomingMessage, res: http.ServerResponse) => {
  if (origins.includes(req.headers.origin ?? "")) {
    res.setHeader("Access-Control-Allow-Methods", "POST");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    res.setHeader("Access-Control-Allow-Origin", req.headers.origin ?? "");
    res.setHeader("Access-Control-Allow-Credentials", "true");
  }
};

const requestListener = (req: http.IncomingMessage, res: http.ServerResponse) => {
  if (req.url === "/") {
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.write("OK");
    res.end();
  }
  else if (req.method === "OPTIONS" && req.url === "/is-authed") {
    setCors(req, res);
    res.end();
  }
  else if (req.method === "OPTIONS" && req.url === "/generate-chat-completion-streaming") {
    setCors(req, res);
    res.end();
  }
  else if (req.method === "POST" && req.url === "/is-authed") {
    setCors(req, res);
    res.setHeader("Content-Type", "application/json");
    const reqCookies = req.headers.cookie ? parseCookie(req.headers.cookie) : {};
    req.on("data", () => {});
    req.on("end", () => {
      postIsAuthed(reqCookies, res);
    });
  }
  else if (req.method === "POST" && req.url === "/generate-chat-completion-streaming") {
    setCors(req, res);
    const reqCookies = req.headers.cookie ? parseCookie(req.headers.cookie) : {};
    const reqBody: Buffer[] = [];
    req.on("data", (chunk) => {
      reqBody.push(chunk);
    });
    req.on("end", async () => {
      await postGenerateChatCompletionStreaming(reqCookies, res, Buffer.concat(reqBody).toString());
    });
  }
  else {
    res.writeHead(404, { "Content-Type": "text/plain" });
    res.end("Not found");
  }
}

const httpsServer = https
  .createServer(
    {
      key: fs.readFileSync("./key.pem"),
      cert: fs.readFileSync("./cert.pem"),
    },
    requestListener
  )
  .on('error', (_err: Error, _req: http.IncomingMessage, res: http.ServerResponse) => {
    res.writeHead(500, { "Content-Type": "text/plain" });
    res.end("Internal server error");
  })
  .listen(port);
console.log(`Server running on port ${port}`);

const httpServer = http.createServer((_req, res) => {
  res.writeHead(403, { "Content-Type": "text/plain" });
  res.end("Forbidden");
}).listen(httpPort);

process.on('SIGTERM', () => {
  console.log('SIGTERM received, closing servers');
  httpsServer.close();
  httpServer.close();
});

process.on('SIGINT', () => {
  console.log('\nSIGINT received, closing servers');
  httpsServer.close();
  httpServer.close();
});
