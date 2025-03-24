import React from 'react';
import ReactDOM from 'react-dom'; // Correct import for React 17
import App from './App';

// Render the app using ReactDOM.render
ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);