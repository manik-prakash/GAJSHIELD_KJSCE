import { useLocation } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { jsPDF } from "jspdf";
import Navbar from "./Navbar";

export default function ResultPage() {
  const location = useLocation();
  const result = location.state?.result;
  const rawResponse = result?.response || "No response available";

  const formatResponseForWeb = (text: string) => {
    const lines = text.split("\n").filter((line) => line.trim() !== "");
    return lines.map((line, index) => {
      if (line.startsWith("### ")) {
        return (
          <h3 key={index} className="text-xl font-bold mt-6 mb-3">
            {line.replace(/^###\s+/, "")}
          </h3>
        );
      } else if (line.startsWith("## ")) {
        return (
          <h2
            key={index}
            className="text-2xl font-bold mt-8 mb-4 border-b pb-2"
          >
            {line.replace(/^##\s+/, "")}
          </h2>
        );
      } else if (line.startsWith("# ")) {
        return (
          <h1
            key={index}
            className="text-3xl font-bold mt-10 mb-6 border-b pb-2"
          >
            {line.replace(/^#\s+/, "")}
          </h1>
        );
      } else {
        // Process bold and other markdown
        const parts = line.split(/(\*\*.*?\*\*)/g);
        return (
          <p key={index} className="mb-4">
            {parts.map((part, i) => {
              if (part.startsWith("**") && part.endsWith("**")) {
                return (
                  <strong key={i} className="font-semibold">
                    {part.slice(2, -2)}
                  </strong>
                );
              }
              return part;
            })}
          </p>
        );
      }
    });
  };

  const handleDownloadPDF = () => {
    const doc = new jsPDF();
    let yPosition = 20;
    const lineHeight = 7;
    const pageHeight = doc.internal.pageSize.height - 20;
    const margin = 15;
    const maxWidth = doc.internal.pageSize.width - 2 * margin;

    // Add title
    doc.setFontSize(18);
    doc.text("Malware Detection Report", margin, yPosition);
    yPosition += 20;

    doc.setFontSize(12);
    const lines = rawResponse
      .split("\n")
      .filter((line: string) => line.trim() !== "");

    lines.forEach((line: string) => {
      if (yPosition > pageHeight) {
        doc.addPage();
        yPosition = 20;
      }

      if (line.startsWith("### ")) {
        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.text(line.replace(/^###\s+/, ""), margin, yPosition);
        doc.setFont("helvetica", "normal");
        yPosition += lineHeight + 2;
      } else if (line.startsWith("## ")) {
        doc.setFontSize(16);
        doc.setFont("", "bold");
        doc.text(line.replace(/^##\s+/, ""), margin, yPosition);
        doc.setFont("", "normal");
        yPosition += lineHeight + 4;
      } else if (line.startsWith("# ")) {
        doc.setFontSize(18);
        doc.setFont("", "bold");
        doc.text(line.replace(/^#\s+/, ""), margin, yPosition);
        doc.setFont("", "normal");
        yPosition += lineHeight + 6;
      } else {
        // Process bold text
        const parts = line.split(/(\*\*.*?\*\*)/g);
        let xPosition = margin;

        parts.forEach((part: string) => {
          if (part.startsWith("**") && part.endsWith("**")) {
            doc.setFont("", "bold");
            const boldText = part.slice(2, -2);
            const textWidth = doc.getTextWidth(boldText);

            if (xPosition + textWidth > maxWidth) {
              yPosition += lineHeight;
              xPosition = margin;
            }

            doc.text(boldText, xPosition, yPosition);
            xPosition += textWidth;
            doc.setFont("", "normal");
          } else if (part.trim() !== "") {
            const regularText = part;
            const textLines = doc.splitTextToSize(
              regularText,
              maxWidth - (xPosition - margin)
            );

            textLines.forEach((textLine: string, i: number) => {
              if (i > 0) {
                yPosition += lineHeight;
                xPosition = margin;
              }
              doc.text(textLine, xPosition, yPosition);
              xPosition += doc.getTextWidth(textLine);
            });
          }
        });

        yPosition += lineHeight;
      }
    });

    doc.save("Malware_Detection_Report.pdf");
  };

  return (
    <>
      <Navbar />
      <div className="flex flex-col items-center max-h-screen bg-white p-4 pt-4">
        <Card className="w-full max-w-4xl shadow-lg mb-8">
          <CardContent className="p-6 space-y-6">
            <h2 className="text-3xl font-bold text-center mb-6">
              Malware Detection Report
            </h2>
            <div className="bg-gray-50 p-6 rounded-lg overflow-y-auto max-h-[50vh] w-full text-gray-800">
              {formatResponseForWeb(rawResponse)}
            </div>
            <div className="flex justify-center mt-4">
              <Button
                onClick={handleDownloadPDF}
                className="bg-black text-white hover:bg-gray-800 px-6 py-3" /* Added padding */
              >
                Download PDF
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
}
