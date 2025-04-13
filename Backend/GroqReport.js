const express = require("express");
const path = require("path");
const fs = require("fs");
const PDFDocument = require("pdfkit");
const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args));
const router = express.Router();

const groqApiKey = "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";

const getGroqChatCompletion = async (userMessage) => {
  const fixedPrompt = `As a Cybersecurity Analyst, generate a formal Malware Detection Report based on the provided JSON input. The input contains results from multiple antivirus engines. Follow this structure:
  1. Title: "Malware Detection Report: [File Name/Hash, if available] - Multi-Engine Analysis"
  2. Executive Summary: 3-4 lines summarizing the analysis. Include the number of engines detecting the file as malicious, the file type (if available), and an overall confidence assessment based on the engine consensus.
  3. Detection Details:
  -   Label:  "Malicious" or "Clean" (Determine based on engine results)
  -   Family:  "[Most frequently reported malware family, if any]" (If engines provide family names)
  -   Type: "[General malware type, e.g., 'Test File', 'Potentially Malicious']" (Based on engine descriptions)
  -   Confidence: "[X%] (Low/Moderate/High)" (Based on the percentage of engines flagging the file as malicious. Consider these ranges:
      -   Low: < 25%
      -   Moderate: 25-75%
      -   High: > 75%
  -   File Name/Hash: "[Value, if available in input data]"
  -   File Type: "[Value, if available. If not, infer from engine results if possible]"
  -   Engine Detections:
      -   [Engine Name 1]: [Result String]
      -   [Engine Name 2]: [Result String]
      -   ... (List all engines and their specific results)
  4. Technical Analysis: Describe the common descriptions or behaviors reported by the engines. Focus on what the engines *detected*, not on a deep dive of the malware itself. For example, "Multiple engines detected EICAR test file signature, indicating a test file and not actual malware."
  5. Risk Assessment:
  -   Severity: "[Low/Medium/High/Informational]" (Base this on the engine consensus and the general type of detection. EICAR test file is Low/Informational. Many engines reporting a Trojan would be High)
  -   Potential Impact: "[Describe the potential impact based on the detection. For EICAR, it's 'No real impact, used for testing.' For other malware, describe typical impacts like data theft, system damage, etc., if the engine results provide clues.]"
  6. Recommendations: Provide actions based on the risk assessment.
  -   If High: "Isolate the file, run a full system scan, investigate the source of the file, and implement any relevant firewall/IDS rules."
  -   If Medium: "Quarantine the file, run a system scan, and monitor for any suspicious activity."
  -   If Low/Informational: "No immediate action is required. The file is likely safe, but continue to practice safe computing habits."
  
  Provide the report in Markdown format.`;

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
    let responseContent = completion.choices?.[0]?.message?.content;

    if (!responseContent) {
      return res.status(500).json({
        success: false,
        error: "Invalid response from Groq API.",
      });
    }
    responseContent = responseContent.replace(/\*\*/g, '');

    const username = req.cookies?.username || "Anonymous";
    const fileName = `${username}-${Date.now()}.pdf`;

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
      // Send file path in response to trigger download in frontend
      res.json({
        success: true,
        filePath: `/uploads/${fileName}`,
      });
      const fileStream = fs.createReadStream(pdfPath);
      fileStream.pipe(res);
      fileStream.on('error', (err) => {
        console.error("File Stream Error:", err);
        res.status(500).json({
          success: false,
          error: "Failed to send the file.",
        });
      });

      fileStream.on('end', () => {
        console.log("File sent successfully.");
      });

      req.on('aborted', () => {
        console.log("Client aborted the request.");
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