// Product-page assistant (demo, TypeScript). Detected heuristically.
import OpenAI from "openai";

const openai = new OpenAI();

export async function askProduct(question: string, productJson: string) {
  const completion = await openai.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: "Answer questions about this product: " + productJson },
      { role: "user", content: question },
    ],
  });
  return completion.choices[0].message.content;
}
