const express = require("express");
const path = require("path");
const fs = require("fs");
const PDFDocument = require("pdfkit");
const fetch = (...args) =>
  import("node-fetch").then(({ default: fetch }) => fetch(...args));
const router = express.Router();

const groqApiKey = "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";

const getGroqChatCompletion = async (userMessage) => {
  const fixedPrompt = `As a Cybersecurity Analyst, generate a formal Malware Detection Report based on the provided JSON input. Follow these guidelines:
  1. If no input data is provided or the input is empty:
    - Return a structured response titled "Insufficient Data Report"
    - Clearly state: "No analysis could be performed due to missing input data"
    - Provide standard guidance: "Please submit file samples or scan results for analysis"
  2. For valid input data containing antivirus scan results:
    Generate a comprehensive report with this structure:
    ## Malware Detection Report: [File Name/Hash if available] - Multi-Engine Analysis
    ### Executive Summary
    - 3-4 line overview including:
      * Number of engines detecting malware
      * File type (if available)
      * Overall confidence assessment
      * Brief conclusion

    ### Detection Details
    - **Label**: "Malicious" or "Clean" (determined by engine consensus)
    - **Family**: [Most reported malware family if multiple engines agree]
    - **Type**: [General classification from engine results]
    - **Confidence**: [X%] (Low/Moderate/High based on detection percentage:
      * Low: <25%
      * Moderate: 25-75%
      * High: >75%
    - **File Identification**:
      * Name/Hash: [value if available]
      * Type: [value if available or inferred]
    - **Engine Detections**:
      * [Tabular format preferred]
      * Include all engines with their specific results

    ### Technical Analysis
    - Describe common patterns in engine detections
    - Highlight any notable signatures or behaviors reported
    - Note any inconsistencies between engines

    ### Risk Assessment
    - **Severity**: [Low/Medium/High/Informational]
      * Base on both detection rate and malware type
    - **Potential Impact**: [Concise description of possible effects]

    ### Recommendations
    Provide action items based on risk level:
    - **High Risk**: Immediate isolation, full system scan, investigation
    - **Medium Risk**: Quarantine, system scan, monitoring
    - **Low/Informational**: Basic vigilance, no urgent action

  3. Formatting Requirements:
    - Use Markdown with proper headings and sections
    - Include horizontal rules between major sections
    - Use bullet points for lists and bold for key terms
    - Maintain consistent spacing and indentation

  4. Tone and Style:
    - Professional and objective
    - Technical but accessible
    - Avoid speculation beyond the data
    - Use clear, actionable language

  Example empty input response:
  ## Insufficient Data Report

  **No analysis could be performed**  
  No input data was provided for analysis. 

  **Recommended Action**:  
  Please submit either:
  - The suspicious file sample, or
  - Complete antivirus scan results

  For comprehensive analysis, include:
  - File hashes (MD5, SHA1, SHA256)
  - Multiple engine scan results
  - Any available behavioral analysis
  `;

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
        model: "llama-3.3-70b-versatile",
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
