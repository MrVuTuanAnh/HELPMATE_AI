import React, { useState } from 'react';
import ReactHtmlParser from 'react-html-parser';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [answers, setAnswers] = useState([]);
  const [useOpenAI, setUseOpenAI] = useState(false); // State to toggle between HelpMate AI and OpenAI

  const askQuestion = async (e) => {
    e.preventDefault();
    if (!question) return;

    // Use the appropriate endpoint based on the `useOpenAI` state
    const endpoint = useOpenAI ? 'http://localhost:8000/ask_with_openai/' : 'http://localhost:8000/ask/';

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();

    // Format the answer
    const formattedAnswer = data.answer.replace(/\.\s/g, '.<br /><br />');

    // Update the answers state with the new answer
    setAnswers([...answers, { question: question, answer: formattedAnswer }]);
    setQuestion(''); // Clear the input after submission
  };

  // Handler to toggle between HelpMate AI and OpenAI
  const toggleOpenAI = () => {
    setUseOpenAI(!useOpenAI);
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
      {/* Button to toggle OpenAI integration */}
      <button onClick={toggleOpenAI}>
        {useOpenAI ? 'Use HelpMate AI' : 'Integrate OpenAI'}
      </button>
      <div className="chatbox">
        {answers.map((entry, index) => (
          <div key={index} className="chat-entry">
            <div className="question">Q: {entry.question}</div>
            <div className="answer">A: {ReactHtmlParser(entry.answer)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
