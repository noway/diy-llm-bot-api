import express from "express";
import cors from "cors";
import fs from "fs";
import https from "https";
import GPT3Tokenizer from "gpt3-tokenizer";

require("dotenv").config();

const MAX_TOKENS = 4097;
const TOKENS_SAFETY_MARGIN = 25;
const tokenizer = new GPT3Tokenizer({ type: "codex" });

const origins = [
  process.env.FRONTEND_URL_1,
  process.env.FRONTEND_URL_2,
].flatMap((f) => (f ? [f] : []));

console.log("origins", origins);

const app = express();
app.use(express.json());
app.use(
  cors({
    origin: origins,
  })
);
app.use(express.text());

const port = process.env.PORT ?? 3000;

interface Choice {
  text: string;
  index: number;
  logprobs?: any;
  finish_reason: string;
}

interface Message {
  text: string;
  name: "You" | "Bot";
  party: "bot" | "human";
  id: number;
}

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
      if (dataLine == "data: [DONE]") {
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
    if (party == "human") {
      prompt += `Human: ${text.trim()}\n\n`;
    } else if (party == "bot") {
      prompt += `Bot: ${text.trim()}\n\n`;
    }
  }
  // prompt for the bot
  prompt += "Bot: ";
  return prompt;
}


class DoubleNewlineReader {
  reader: ReadableStreamDefaultReader<BufferSource>;
  buffer: string;

  constructor(reader: ReadableStreamDefaultReader<BufferSource>) {
    this.reader = reader;
    this.buffer = '';
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
      let dataString = new TextDecoder().decode(value);
      dataString = dataString.replace(/\r\n/g, "\n");
      this.buffer += dataString;  // Assuming value is a string; adjust if not
    }
    return { done: true, value: this.buffer };
  }
}

app.get("/", async (req, res) => {
  res.contentType("text").send("OK");
});

app.post("/generate-chat-completion-streaming", async (req, res) => {
  try {
    if (typeof req.body !== "string") {
      throw new Error("body is not a string");
    }
    const body = JSON.parse(req.body);
    const authKey = body.authKey as string;
    const messages = body.messages as Message[];
    const model = (body.model ?? "gpt-3.5-turbo") as string;
    const humanMessages = messages.filter((m) => m.party == "human");
    const lastHumanMessage = humanMessages[humanMessages.length - 1];
    console.log("model", model);
    console.log("human-prompt", lastHumanMessage.text);
    const BEARER_TOKEN = process.env.BEARER_TOKEN;

    if (model === "gpt-3.5-turbo" || model === "gpt-4" || model === "gpt-4-1106-preview" || model === "mistralai/Mixtral-8x7B-Instruct-v0.1") {
      if (model === "gpt-4" && authKey !== process.env.AUTH_KEY) {
        throw new Error("Invalid auth key");
      }
      const chatMessages = [
        {
          role: "system",
          content: "You are a helpful AI language model assistant.",
        },
        ...messages.map((m) => ({
          role: m.party == "human" ? "user" : "assistant",
          content: m.text,
        })),
      ];
      const options = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${BEARER_TOKEN}`,
        },
        body: JSON.stringify({
          model,
          messages: chatMessages,
          stream: true,
          stop: "END_OF_STREAM",
        }),
      };
      const response = await fetch(
        "https://api.deepinfra.com/v1/openai/chat/completions",
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

      // send json
      res.set({ "transfer-encoding": "chunked" });

      const reader = response.body.getReader();
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
          res.write(content);
          completion += content;
        }
      }
      res.end();
      console.log("completion", completion);
    } else {
      const prompt = generatePrompt(messages);
      const encoded: { bpe: number[]; text: string[] } =
        tokenizer.encode(prompt);
      const promptTokens = encoded.bpe.length;

      const temperature = 0.5;
      const options = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${BEARER_TOKEN}`,
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
        "https://api.openai.com/v1/completions",
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

      // send json
      res.set({ "transfer-encoding": "chunked" });

      const reader = response.body.getReader();
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
          res.write(token);
          completion += token;
        }
      }
      res.end();
      console.log("completion", completion.trim());
    }
  } catch (error) {
    console.error("error", error);
    try {
      res.json({
        success: false,
        error: { message: (error as Error).message },
      });
    } catch (e) {
      console.error("e", e);
      // do nothing
    }
  }
});

https
  .createServer(
    {
      key: fs.readFileSync("./key.pem"),
      cert: fs.readFileSync("./cert.pem"),
    },
    app
  )
  .listen(port);
