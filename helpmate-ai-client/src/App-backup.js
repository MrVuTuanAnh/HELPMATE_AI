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



    // Check if 'answer' exists in the responses and format them, or use a default message

    const helpmateFormattedAnswer = helpmateData?.answer ? helpmateData.answer.replace(/\\.\\s/g, '.<br /><br />') : "No answer provided by HelpMate AI.";

    const openAIFormattedAnswer = openAIData?.answer ? openAIData.answer.replace(/\\.\\s/g, '.<br /><br />') : "No answer provided by OpenAI.";



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

            <div className="answer helpmate-answer">

              HelpMate AI: {ReactHtmlParser(entry.helpmateAnswer)}

            </div>

            <div className="answer openai-answer">

              OpenAI: {ReactHtmlParser(entry.openAIAnswer)}

            </div>

          </div>

        ))}

      </div>

    </div>

  );

}



export default App;
