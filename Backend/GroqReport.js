const express = require("express");
const path = require("path");
const fs = require("fs");
const PDFDocument = require("pdfkit");
const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args));
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
    const { prompt, filename } = req.body;

    if (!prompt) {
      return res.status(400).json({
        success: false,
        error: "All fields are required.",
      });
    }

    const completion = await getGroqChatCompletion(prompt);
    console.log("Completion from Groq API:", completion);

    let responseContent = completion.choices?.[0]?.message?.content;
    console.log("Response from Groq API (before clean):", responseContent);

    if (!responseContent) {
      return res.status(500).json({
        success: false,
        error: "Invalid response from Groq API.",
      });
    }

    responseContent = responseContent.replace(/\*\*/g, '');
    console.log("Response from Groq API (after clean):", responseContent);

    const username = req.cookies?.username || "Anonymous";
    const fileName = `${username}_${filename}-${Date.now()}.pdf`;

    const uploadDir = path.join(__dirname, "../uploads");
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir);
    }

    const pdfPath = path.join(uploadDir, fileName);
    const doc = new PDFDocument({ margin: 50 });
    const writeStream = fs.createWriteStream(pdfPath);

    doc.pipe(writeStream);

    const sections = responseContent.split(/\n(?=[A-Z])/);

    sections.forEach((section) => {
      const lines = section.trim().split("\n");

      if (lines.length > 0) {
        doc.font("Helvetica-Bold").fontSize(16).text(lines[0], {
          align: "left",
          underline: false,
        });
        doc.moveDown(0.5);

        if (lines.length > 1) {
          doc.font("Helvetica").fontSize(12);
          for (let i = 1; i < lines.length; i++) {
            doc.text(lines[i], {
              align: "left",
              paragraphGap: 5,
            });
          }
        }

        doc.moveDown(1);
      }
    });

    doc.end();

    writeStream.on("finish", () => {
      res.json({
        success: true,
        filePath: `/uploads/${fileName}`,
      });
    });

    writeStream.on("error", (err) => {
      console.error("PDF Write Error:", err);
      res.status(500).json({
        success: false,
        error: "Failed to generate PDF.",
      });
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