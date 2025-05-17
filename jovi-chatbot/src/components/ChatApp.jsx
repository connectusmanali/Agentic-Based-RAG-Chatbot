import React, { useState, useEffect, useRef } from 'react';
import './ChatApp.css';
import { marked } from 'marked';

const ChatApp = () => {
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem('jovi_chat_history');
    return saved ? JSON.parse(saved) : [];
  });

  const [input, setInput] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const messagesRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunks = useRef([]);

  useEffect(() => {
    localStorage.setItem('jovi_chat_history', JSON.stringify(messages));
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesRef.current?.scrollTo(0, messagesRef.current.scrollHeight);
  };

  const appendMessage = (sender, content, isHtml = false) => {
    const msg = { sender, content, isHtml };
    setMessages((prev) => [...prev, msg]);
  };

  const updateLastBotMessage = (content, isHtml = false) => {
    setMessages((prev) => {
      const updated = [...prev];
      updated[updated.length - 1] = { sender: 'bot', content, isHtml };
      return updated;
    });
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = input;
    appendMessage('user', userMsg);
    appendMessage('bot', '<div class="typing-indicator"><span></span><span></span><span></span></div>', true);
    setInput('');

    const formData = new FormData();
    formData.append('message', userMsg);

    const res = await fetch('http://localhost:8000/api/query', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    updateLastBotMessage(marked.parse(data.answer), true);
  };

  const handleVoiceInput = async () => {
    if (isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    mediaRecorderRef.current = recorder;
    audioChunks.current = [];

    recorder.ondataavailable = (e) => audioChunks.current.push(e.data);

    recorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' });
      const formData = new FormData();
      const file = new File([audioBlob], 'voice.webm', { type: 'audio/webm' });
      formData.append('file', file);

      appendMessage('user', '🎤 [Voice Input]');
      appendMessage('bot', '<div class="typing-indicator"><span></span><span></span><span></span></div>', true);

      const res = await fetch('http://localhost:8000/api/transcribe', { method: 'POST', body: formData });
      const data = await res.json();

      appendMessage('user', data.query);
      const textForm = new FormData();
      textForm.append('message', data.query);

      const botRes = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        body: textForm,
      });

      const botData = await botRes.json();
      updateLastBotMessage(marked.parse(botData.answer), true);
    };

    recorder.start();
    setIsRecording(true);
  };

  return (
    <div className="chat-tab">
      <div className="chatbox">
        <div className="header">
          <div className="bot-info">
            <img src="/static/images/Avatar.png" className="avatar" alt="Bot" />
            <div>
              <div className="bot-name">JoviBot</div>
              <small className="bot-sub">jovirealty.com</small>
            </div>
          </div>
        </div>

        <div className="messages" ref={messagesRef}>
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`message ${msg.sender}`}
              dangerouslySetInnerHTML={msg.isHtml ? { __html: msg.content } : undefined}
            >
              {!msg.isHtml && msg.content}
            </div>
          ))}
        </div>

        <form className="chat-form" onSubmit={sendMessage}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            required
          />
          <button type="submit">➤</button>
          <button type="button" onClick={handleVoiceInput}>
            {isRecording ? '⏹️' : '🎤'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatApp;