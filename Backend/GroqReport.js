const express = require("express");
const path = require("path");
const fs = require("fs");
const PDFDocument = require("pdfkit");
const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args)); // Fix for fetch
const router = express.Router();

const groqApiKey = "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";

const getGroqChatCompletion = async (userMessage) => {
  const fixedPrompt = `As a Cybersecurity Analyst, generate a formal Malware Detection Report based on the provided JSON input. Follow this structure:
1. Title: "Malware Detection Report: [Family] [Type]"
2. Executive Summary: 3-4 lines summarizing status, family, type, and confidence
3. Detection Details:
- Label: True/False
- Family: [if known]
- Type: [Trojan/Ransomware/etc.]
- Confidence: [X%] (Low/Moderate/High)
- File Type: [extension]
4. Technical Analysis: Describe behavior and risks
5. Risk Assessment: Severity and potential impact
6. Recommendations: Immediate actions and preventive measures
7. Appendix: Original JSON input`;

  const finalPrompt = `${fixedPrompt}\n\nHere is the input:\n${userMessage}`;

  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${groqApiKey}`,
    },
    body: JSON.stringify({
      model: "llama-3.3-70b-versatile",
      messages: [
        { role: "user", content: finalPrompt }
      ],
      max_tokens: 1000,
    }),
  });

  const data = await response.json();
  return data;
};

router.post("/chat", async (req, res) => {
  try {
    const { label, family, type, confidence, filetype, filename } = req.body;

    if (!label || !family || !type || !confidence || !filetype || !filename) {
      return res.status(400).json({
        success: false,
        error: "All fields are required.",
      });
    }

    const userMessage = JSON.stringify({
      label: label,
      family: family,
      type: type,
      confidence: confidence,
      filetype: filetype,
    });

    const completion = await getGroqChatCompletion(userMessage);
    console.log("Completion from Groq API:", completion);
    const responseContent = completion.choices?.[0]?.message?.content;
    console.log("Response from Groq API:", responseContent);
    if (!responseContent) {
      return res.status(500).json({
        success: false,
        error: "Invalid response from Groq API.",
      });
    }

    const username = req.cookies?.username || "Anonymous";
    const fileName = `${username}_${filename}-${Date.now()}.pdf`;
    const pdfPath = path.join(__dirname, "../uploads", fileName);

    if (!fs.existsSync(path.join(__dirname, "../uploads"))) {
      fs.mkdirSync(path.join(__dirname, "../uploads"));
    }

    const doc = new PDFDocument();
    const writeStream = fs.createWriteStream(pdfPath);
    doc.pipe(writeStream);
    doc.fontSize(12).text(responseContent, { align: "left" });
    doc.end();

    writeStream.on("finish", () => {
      res.json({
        success: true,
        filePath: `/uploads/${fileName}`,
      });
    });

    writeStream.on("error", (err) => {
      res.status(500).json({
        success: false,
        error: "Failed to generate PDF file.",
      });
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message || "An unexpected error occurred",
    });
  }
});

module.exports = router;
