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



    const endpoint = useOpenAI ? 'http://localhost:8000/ask_with_openai/' : 'http://localhost:8000/ask/';



    try {

      const response = await fetch(endpoint, {

        method: 'POST',

        headers: {

          'Content-Type': 'application/json',

        },

        body: JSON.stringify({ question }),

      });

      const data = await response.json();



      // Check if the 'answer' property exists in the response

      if (data && data.answer) {

        // If yes, format the answer and update the state

        const formattedAnswer = data.answer.replace(/\\.\\s/g, '.<br /><br />');

        setAnswers([...answers, { question: question, answer: formattedAnswer }]);

      } else {

        // If no 'answer' property, handle it appropriately (e.g., log error, show message)

        console.error('Answer property not found in response:', data);

        // Optionally, add a placeholder or error message in the answers state

        setAnswers([...answers, { question: question, answer: "Sorry, no answer found." }]);

      }

    } catch (error) {

      // Handle fetch errors

      console.error('Fetch error:', error);

    }



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