import express, { NextFunction, Request, Response } from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
import fs from "fs";
import https from "https";
import GPT3Tokenizer from "gpt3-tokenizer";
import { z } from "zod";

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
    credentials: true,
  })
);
app.use(express.text());
app.use(cookieParser());

const port = process.env.PORT ?? 3000;
const httpPort = process.env.HTTP_PORT ?? 8080;

interface Choice {
  text: string;
  index: number;
  logprobs?: any;
  finish_reason: string;
}

const MessageSchema = z.object({
  text: z.string(),
  name: z.enum(["You", "Bot"]),
  party: z.enum(["bot", "human"]),
  id: z.number(),
});
type Message = z.infer<typeof MessageSchema>;

const MessagesSchema = z.array(MessageSchema);

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
    const cookies = CookiesSchema.parse(req.cookies);
    const body = JSON.parse(req.body);
    const authKey = cookies["__Secure-authKey"];
    const messages = MessagesSchema.parse(body.messages);
    const model = z.string().parse(body.model);
    const humanMessages = messages.filter((m) => m.party == "human");
    const lastHumanMessage = humanMessages[humanMessages.length - 1];
    if (messages[0]) {
      if (messages[0].party !== "human") {
        throw new Error("Validation error");
      }
    }

    console.log("model", model);
    console.log("human-prompt", lastHumanMessage.text);
    const BEARER_TOKEN = process.env.BEARER_TOKEN;

    const isMixtral = model === "mistralai/Mixtral-8x7B-Instruct-v0.1";
    const isLlama3_70b = model === "meta-llama/Llama-3-70b-chat-hf";
    const isOpus = model === "anthropic/claude-3-opus:beta";
    const isMistralLarge = model === "mistralai/mistral-large";

    if (model === "gpt-3.5-turbo" || model === "gpt-4" || model === "gpt-4-1106-preview" || model === "gpt-4o" || isMixtral || isLlama3_70b || isOpus || isMistralLarge) {
      if (model === "gpt-4" && authKey !== process.env.AUTH_KEY) {
        throw new Error("Invalid auth key");
      }
      const chatMessages = [
        ...(!(isMixtral || isLlama3_70b || isOpus || isMistralLarge) ? [{
          role: "system",
          content: "You are a helpful AI language model assistant.",
        }] : []),
        ...messages.map((m) => ({
          role: m.party == "human" ? "user" : "assistant",
          content: m.text,
        })),
      ];
      const options = {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${isMixtral ? process.env.DEEPINFRA_BEARER_TOKEN : ((isOpus || isMistralLarge) ? process.env.OPENROUTER_BEARER_TOKEN : (isLlama3_70b ? process.env.TOGETHER_BEARER_TOKEN : BEARER_TOKEN))}`,
        },
        body: JSON.stringify({
          model,
          messages: chatMessages,
          stream: true,
          stop: isLlama3_70b ? "<|eot_id|>" : "END_OF_STREAM",
        }),
      };
      const response = await fetch(
        isMixtral ? "https://api.deepinfra.com/v1/openai/chat/completions" : ((isOpus || isMistralLarge) ? "https://openrouter.ai/api/v1/chat/completions" : (isLlama3_70b ? "https://api.together.xyz/v1/chat/completions" : "https://api.openai.com/v1/chat/completions")),
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

app.post("/is-authed", async (req, res) => {
  try {
    const cookies = CookiesSchema.parse(req.cookies);
    const authKey = cookies["__Secure-authKey"];
    res.json({
      success: true,
      isAuthed: authKey === process.env.AUTH_KEY,
    });
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
})

app.use(function (err: Error, req: Request, res: Response, next: NextFunction) {
  res.status(500).contentType("text").send("Internal server error");
});

app.use(function (req, res, next) {
  res.status(404).contentType("text").send("Not found");
})

https
  .createServer(
    {
      key: fs.readFileSync("./key.pem"),
      cert: fs.readFileSync("./cert.pem"),
    },
    app
  )
  .listen(port);

const httpApp = express();
httpApp.use(function (req, res) {
  res.status(403).contentType("text").send("Forbidden")
});
httpApp.listen(httpPort);