import express from "express";
const app = express();
const port = 3000;

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

app.get("/", async (req, res) => {
  const BEARER_TOKEN = process.env.BEARER_TOKEN;
  const model = "text-davinci-002";
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
      max_tokens: 1024,
    }),
  };
  const response = await fetch(
    "https://api.openai.com/v1/completions",
    options
  );
  const body = (await response.json()) as OpenAICompletionResponse;
  res.send(body["choices"][0]["text"]);
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
