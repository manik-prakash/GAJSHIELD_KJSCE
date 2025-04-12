const express = require('express');
const Groq = require('groq-sdk');
const router = express.Router();
require('dotenv').config();

const fixedPrompt = `You are an AI-powered chatbot for a Web3-based land allocation system. 
  Your primary role is to assist users in understanding land ownership, smart contracts, 
  bidding, and transaction history on the blockchain.
  Provide clear, structured, and informative responses to user queries. Cover these key topics:
  1. **General Inquiry:** Basics of the land allocation system, security, and decentralization benefits.
  2. **Bidding and Auctions:** On-chain land bidding process, auctions, and smart contracts.
  3. **Ownership and Transactions:** Verifying ownership, transaction history, and land transfers.
  4. **Smart Contracts:** Managing land allocations, escrow mechanisms, and dispute resolution.
  5. **Security and Privacy:** Blockchain security, data privacy, and immutability of records.
  6. **MOST IMPORTANT:** Only answer questions related to Web3-based land allocation. Avoid financial advice. Also only give short answers not long answers.
  IMPORTANT: Only answer questions related to Web3-based land allocation. Avoid financial advice.
  Keep responses concise, engaging, and Web3-focused. Do not generate code or unnecessary formatting.`;

// Function to get Groq chat stream
async function getGroqChatStream(userMessage, apiKey) {
  if (!apiKey) {
    throw new Error("Groq API key is required.");
  }

  const groq = new Groq({ apiKey });

  return groq.chat.completions.create({
    messages: [
      { role: "system", content: fixedPrompt },
      { role: "user", content: userMessage },
    ],
    model: "llama-3.3-70b-versatile",
    temperature: 0.5,
    max_completion_tokens: 1024,
    top_p: 1,
    stream: true,
  });
}

router.post("/chat", async (req, res) => {
  try {
    const { message } = req.body;
    const groqApiKey = process.env.GROQ_API_KEY;
    if (!message) {
      return res.status(400).json({ success: false, error: "User prompt is required." });
    }
    if (!groqApiKey) {
      return res.status(400).json({ success: false, error: "Groq API key is missing." });
    }
    res.setHeader("Content-Type", "application/json");
    let responseContent = "";
    const stream = await getGroqChatStream(message, groqApiKey);
    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || "";
      responseContent += content;
    }
    res.json({ success: true, response: responseContent });
  } catch (error) {
    console.error("Error in /chat endpoint:", error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

module.exports = router;