import express from "express";
import cors from "cors";
import fs from "fs";
import https from "https";

require("dotenv").config();

const app = express();
app.use(express.json());
app.use(
  cors({
    origin: process.env.FRONTEND_URL,
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

interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

interface OpenAICompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Choice[];
  usage: Usage;
}

interface Message {
  text: string;
  party: "bot" | "human";
  timestamp: number;
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

app.get("/", async (req, res) => {
  res.contentType("text").send("OK");
});
app.post("/generate-chat-completion", async (req, res) => {
  const messages = req.body.messages as Message[];
  const model = (req.body.model ?? "text-davinci-002") as string;
  const humanMessages = messages.filter((m) => m.party == "human");
  const lastHumanMessage = humanMessages[humanMessages.length - 1];
  console.log("model", model);
  console.log("human-prompt", lastHumanMessage.text);
  try {
    const BEARER_TOKEN = process.env.BEARER_TOKEN;
    const temperature = 0.5;
    const options = {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${BEARER_TOKEN}`,
      },
      body: JSON.stringify({
        model,
        prompt: generatePrompt(messages),
        temperature,
        max_tokens: 1024,
      }),
    };
    const response = await fetch(
      "https://api.openai.com/v1/completions",
      options
    );
    const body = (await response.json()) as OpenAICompletionResponse;
    const completion = body["choices"][0]["text"];
    console.log("completion", completion);
    // send json
    res.send(JSON.stringify({ success: true, completion }));
  } catch (error) {
    console.error(error);
    res.json({ success: false, error: { message: (error as Error).message } });
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
