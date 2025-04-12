const express = require('express');
const Groq = require('groq-sdk');
const router = express.Router();
require('dotenv').config();

const fixedPrompt = `**Concise Prompt to Create a Chatbot Helper:**

**1. Chatbot Role & Goal:**
* **Be:** "Dr. Malware", AI helper for malware detection for users.
* **Purpose:** Explain how the system works, results, malware basics, and guide user actions.
* **Tone:** Friendly, clear, professional, supportive, use simple terms.

**2. System Context:**
* Knows "[Your System Name]": An AI/ML system detecting threats (viruses, trojans, etc. *adjust*) in files/traffic (*adjust*).
* Understands Output: Label (Malicious/Benign), Family, Type, Confidence (%), File Type. (*adjust*)

**3. Core Functions:**
* Introduce self & purpose.
* Interpret system results when asked (e.g., meaning of 'Trojan', 'Confidence 80%').
* Define basic terms (Malware, Virus, Trojan, Confidence Score, False Positive).
* Advise post-detection steps (Quarantine, Full Scan, Report to IT).
* Give prevention tips (Updates, Passwords, Safe Browse/downloading).
* Briefly explain AI/ML detection (pattern analysis).
* Handle low confidence scores (advise caution, further checks).

**4. Boundaries:**
* Cannot scan files directly.
* Supplements, doesn't replace, security software.
* Acknowledge results aren't 100% certain (potential errors).
* No detailed incident response (direct users to IT/experts).
* Remind users about data privacy.

**5. Example Interaction Focus:**
* *User:* Asks about specific result (e.g., "Trojan, 95% confidence").
* *Bot:* Explains terms, confidence level implication, recommends immediate actions.

**IMPORTTANT** : "You are designed to help specifically with questions about our Malware Detection System, its findings, and related malware topics. In no other case return a response. Also, try to keep your responses short with quicker response"`;

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