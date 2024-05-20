import React from 'react';
import './App.css';
import EmailClassifier from './EmailClassifier';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Email Spam Classifier</h1>
        <EmailClassifier />
      </header>
    </div>
  );
}

export default App;
