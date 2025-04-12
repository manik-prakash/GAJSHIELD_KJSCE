import React, { useState, FormEvent, useRef, useEffect } from "react";
import axios from "axios";
import Navbar from "./Navbar";

interface Message {
  type: "user" | "bot";
  text: string;
}

const formatBotMessage = (text: string): string => {
  const boldFormatted = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  return boldFormatted;
};


const ChatApp: React.FC = () => {
  const [message, setMessage] = useState<string>("");
  const [chatHistory, setChatHistory] = useState<Message[]>([
    {
      type: "bot",
      text: "Hello! Gaffer, I am here to assist you on Malicious Detection System",
    },
  ]);
  const [loading, setLoading] = useState<boolean>(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory]);


  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const trimmedMessage = message.trim();
    if (!trimmedMessage) return;

    const userMessage: Message = { type: "user", text: trimmedMessage };
    setChatHistory((prev) => [...prev, userMessage]);
    setMessage("");
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:8080/assistance/chat", {
        message: trimmedMessage,
      });

      if (res.data && typeof res.data.response === 'string') {
        const botResponse: Message = { type: "bot", text: res.data.response };
        setChatHistory((prev) => [...prev, botResponse]);
      } else {
        console.error("Invalid response structure:", res.data);
        const errorMessage: Message = {
          type: "bot",
          text: `Error: Received invalid response from server. ${res.data?.message || ''}`,
        };
        setChatHistory((prev) => [...prev, errorMessage]);
      }
    } catch (error) {
       console.error("Chat API error:", error);
       let errorText = "Error: Something went wrong while contacting the server.";
       if (axios.isAxiosError(error)) {
           if (error.response) {
               errorText = `Error: Server responded with status ${error.response.status}. ${error.response.data?.message || ''}`;
           } else if (error.request) {
               errorText = "Error: No response received from the server. Is it running?";
           } else {
               errorText = `Error: Could not send request. ${error.message}`;
           }
       }
      const errorMessage: Message = {
        type: "bot",
        text: errorText,
      };
      setChatHistory((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Navbar />
      <div className="flex items-center justify-center max-h-screen bg-white text-black p-4 pt-20">
        <div className="bg-white border border-gray-300 shadow-lg p-6 rounded-lg w-[70vw] h-[75vh] flex flex-col max-w-4xl">
          <h1 className="text-2xl font-bold text-center text-gray-800 mb-4">
            Malicious Detection Assistant
          </h1>

          <div
            ref={chatContainerRef}
            className="flex flex-col flex-grow overflow-y-auto space-y-3 mb-4 pr-2 border-t border-b border-gray-200 py-4"
          >
            {chatHistory.map((msg, index) => {
              const isUser = msg.type === "user";
              const messageHtml = !isUser ? formatBotMessage(msg.text) : undefined;

              return (
                <div
                  key={index}
                  className={`p-3 max-w-[75%] rounded-lg shadow-sm ${
                    isUser
                      ? "bg-black text-white self-end rounded-br-none"
                      : "bg-gray-200 text-gray-800 self-start rounded-bl-none"
                  }`}
                >
                  {isUser ? (
                    <span>{msg.text}</span>
                  ) : (
                    <span dangerouslySetInnerHTML={{ __html: messageHtml || '' }} />
                  )}
                </div>
              );
            })}
            {loading && (
                 <div className="self-start">
                     <div className="bg-gray-200 text-gray-500 p-3 rounded-lg rounded-bl-none inline-flex items-center space-x-2">
                         <span className="typing-dot"></span>
                         <span className="typing-dot animation-delay-200"></span>
                         <span className="typing-dot animation-delay-400"></span>
                     </div>
                 </div>
             )}
          </div>

          <form onSubmit={handleSubmit} className="flex space-x-3 items-center">
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Ask your question here..."
              className="flex-grow p-3 bg-gray-100 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent text-black placeholder-gray-500"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !message.trim()}
              className={`px-5 py-3 rounded-lg font-semibold transition-colors duration-200 ease-in-out ${
                loading || !message.trim()
                  ? "bg-gray-400 text-gray-700 cursor-not-allowed"
                  : "bg-black text-white"
              }`}
            >
              {loading ? "Sending..." : "Send"}
            </button>
          </form>
        </div>
        <style>{`
            .typing-dot {
                display: inline-block;
                width: 8px;
                height: 8px;
                background-color: currentColor;
                border-radius: 50%;
                animation: typing-bounce 1s infinite ease-in-out;
            }
            .animation-delay-200 { animation-delay: 0.2s; }
            .animation-delay-400 { animation-delay: 0.4s; }
            @keyframes typing-bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-4px); }
            }
        `}</style>
      </div>
    </>
  );
};

export default ChatApp;