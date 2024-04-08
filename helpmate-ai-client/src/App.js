import React, { useState } from 'react';
import ReactHtmlParser from 'react-html-parser'; // Import thư viện này
import './App.css';



function App() {

  const [question, setQuestion] = useState('');

  const [answers, setAnswers] = useState([]);

  const [useOpenAI, setUseOpenAI] = useState(false); // State to toggle between HelpMate AI and OpenAI



  const askQuestion = async (e) => {

    e.preventDefault();

    if (!question) return;

    const response = await fetch('http://localhost:8000/ask/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();

    // Cập nhật câu trả lời để mỗi câu mới xuống hàng
    const formattedAnswer = data.answer.replace(/\.\s/g, '.<br /><br />');

    setAnswers([...answers, { question: question, answer: formattedAnswer }]);
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

      {/* Button to toggle OpenAI integration */}

      <button onClick={toggleOpenAI}>

        {useOpenAI ? 'Use HelpMate AI' : 'Integrate OpenAI'}

      </button>

      <div className="chatbox">

        {answers.map((entry, index) => (

          <div key={index} className="chat-entry">

            <div className="question">Q: {entry.question}</div>
            {/* Sử dụng ReactHtmlParser để render câu trả lời đã được format */}
            <div className="answer">A: {ReactHtmlParser(entry.answer)}</div>
          </div>

        ))}

      </div>

    </div>

  );

}



export default App;