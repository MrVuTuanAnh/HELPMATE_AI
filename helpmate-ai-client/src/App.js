import React, { useState } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);

  const handleQueryChange = (event) => {
    setQuery(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await fetch('http://localhost:8000/query/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      const data = await response.json();
      if (data && data.top_3_results) {
        setResults(data.top_3_results);
      } else {
        setResults([]);
        alert('Không tìm thấy kết quả phù hợp.');
      }
    } catch (error) {
      console.error('Lỗi khi truy vấn:', error);
      alert('Có lỗi xảy ra khi thực hiện truy vấn.');
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Truy vấn về Chính Sách Bảo Hiểm</h1>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={query}
            onChange={handleQueryChange}
            placeholder="Nhập truy vấn của bạn..."
          />
          <button type="submit">Tìm kiếm</button>
        </form>
        <div>
          {results.length > 0 && (
            <div>
              <h2>Kết quả:</h2>
              <ul>
                {results.map((result, index) => (
                  <li key={index}>{result}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </header>
    </div>
  );
}

export default App;
