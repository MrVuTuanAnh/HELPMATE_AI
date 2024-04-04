import React, { useState } from 'react';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [helpMateAnswer, setHelpMateAnswer] = useState('');
  const [openAIAnswer, setOpenAIAnswer] = useState('');

  const askHelpMate = async () => {
    // ... existing logic to call /ask/ endpoint
  };

  const askWithOpenAI = async () => {
    try {
      const response = await fetch('http://localhost:8000/ask_with_openai/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setHelpMateAnswer(data.answer_from_helpmate); // Update with actual value
      setOpenAIAnswer(data.answer_from_openai);
    } catch (e) {
      console.error("Error when calling askWithOpenAI:", e);
      setOpenAIAnswer("Sorry, we're having trouble getting a response from OpenAI right now.");
    }
  };

  return (
    <div className="App">
      {/* ... rest of your JSX code */}
      <button onClick={askWithOpenAI}>Integrate OpenAI</button>
      {/* ... update response display to include both answers */}
    </div>
  );
}

export default App;
