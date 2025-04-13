const express = require("express");
const path = require("path");
const fs = require("fs");
const PDFDocument = require("pdfkit");
const fetch = (...args) =>
  import("node-fetch").then(({ default: fetch }) => fetch(...args));
const router = express.Router();

const groqApiKey = "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";

const getGroqChatCompletion = async (userMessage) => {
  const fixedPrompt = `${userMessage}
  Based on the provided input containing antivirus engine results, please generate a formal Malware Detection Report with the following structure:
  **1. Title:**  
  - "Malware Detection Report"
  **2. Executive Summary:**  
  - 3-4 lines summarizing the analysis
  - Include the number of engines detecting the file as malicious
  - Mention the file type (if available)
  - Provide an overall confidence assessment
  - List the names of detecting engines if possible

  **3. Detection Details:**  
  - **Label:** "Malicious" if any engine flags it; otherwise, "Clean"
  - **Detection Count:** Number of engines detecting as malicious
  - **Family:** Most frequently reported malware family (if available)
  - **Type:** General malware type (e.g., 'Test File', 'Potentially Malicious')
  - **Confidence:** Calculate percentage and rate as:
    - Low: < 25%
    - Moderate: 25% to 75%
    - High: > 75%
  - **File Name/Hash:** Value, if available
  - **File Type:** Value if available; otherwise infer if possible
  - **Engine Detections:**
    - [Engine Name 1]: [Result String]
    - [Engine Name 2]: [Result String]
    - (List all engines)

  **4. Technical Analysis:**
  - Summarize common behaviors or descriptions reported by the engines
  - Focus on detected traits, file type characteristics, and typical file uses
  - Avoid deep malware analysis

  **5. Risk Assessment:**
  - **Severity:** Rate as "Low/Medium/High/Informational" based on consensus and malware type
    - EICAR test files = Low/Informational
    - Multiple Trojans = High
  - **Potential Impact:** Describe potential impact based on detection type
    - Example: EICAR = "No real impact, used for testing"
    - Example: Real malware = "Data theft, system compromise," etc.

  **6. Recommendations:**
  - **High Severity:**
    "Isolate the file immediately. Conduct a full system scan. Investigate the source. Implement firewall/IDS rules to prevent further compromise."
  - **Medium Severity:**
    "Quarantine the file. Perform a full system scan. Monitor system behavior closely. Investigate potential entry points."
  - **Low/Informational Severity:**
    "No immediate action is necessary. File is likely safe. Maintain safe computing practices. Continue monitoring for unusual activity."

  If the file is completely clean (no detections), clearly state "Safe. No need to worry. The file is free from malware." in the executive summary.`;

  const finalPrompt = `${fixedPrompt}\n\nHere is the input:\n${userMessage}`;

  const response = await fetch(
    "https://api.groq.com/openai/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${groqApiKey}`,
      },
      body: JSON.stringify({
        model: "meta-llama/llama-4-scout-17b-16e-instruct",
        messages: [{ role: "user", content: finalPrompt }],
        max_tokens: 1000,
      }),
    }
  );

  const data = await response.json();
  return data;
};

router.post("/chat", async (req, res) => {
  try {
    const { prompt, filename } = req.body;

    if (!prompt) {
      return res.status(400).json({
        success: false,
        error: "All fields are required.",
      });
    }

    const completion = await getGroqChatCompletion(prompt);
    let responseContent = completion.choices?.[0]?.message?.content;

    if (!responseContent) {
      return res.status(500).json({
        success: false,
        error: "Invalid response from Groq API.",
      });
    }
    responseContent = responseContent.replace(/\*\*/g, "");
    res.json({
      success: true,
      response: responseContent,
    });
  } catch (error) {
    console.error("Server Error:", error);
    res.status(500).json({
      success: false,
      error: "An error occurred.",
    });
  }
});

module.exports = router;
