import React, { useState } from 'react';
import ReactHtmlParser from 'react-html-parser';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [answers, setAnswers] = useState([]);

  const askQuestion = async (e) => {
    e.preventDefault();
    if (!question) return;

    // Call both endpoints simultaneously
    const helpmateResponsePromise = fetch('http://localhost:8000/ask/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });

    const openAIResponsePromise = fetch('http://localhost:8000/ask_with_openai/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });

    // Wait for both promises to resolve
    const [helpmateResponse, openAIResponse] = await Promise.all([
      helpmateResponsePromise,
      openAIResponsePromise,
    ]);

    const helpmateData = await helpmateResponse.json();
    const openAIData = await openAIResponse.json();

    // Format both answers
    const helpmateFormattedAnswer = helpmateData.answer.replace(/\.\s/g, '.<br /><br />');
    const openAIFormattedAnswer = openAIData.answer.replace(/\.\s/g, '.<br /><br />');

    // Update the answers state with the new answers
    setAnswers([...answers, { question: question, helpmateAnswer: helpmateFormattedAnswer, openAIAnswer: openAIFormattedAnswer }]);
    setQuestion(''); // Clear input after submit
  };

  return (
    <div className="App">
      <h1>Mr.HelpMate AI Chatbox</h1>
      <form onSubmit={askQuestion}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question..."
        />
        <button type="submit">Ask</button>
      </form>
      <div className="chatbox">
        {answers.map((entry, index) => (
          <div key={index} className="chat-entry">
            <div className="question">Q: {entry.question}</div>
            <div className="answer">HelpMate AI: {ReactHtmlParser(entry.helpmateAnswer)}</div>
            <div className="answer">OpenAI: {ReactHtmlParser(entry.openAIAnswer)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;

