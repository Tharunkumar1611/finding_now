import React from 'react';
import { useNavigate } from 'react-router-dom';
import './FirstPage.css';

function FirstPage() {
  const navigate = useNavigate();

  const handleRoleSelection = (role) => {
    if (role === 'user') {
      navigate('/user-login');
    } else if (role === 'official') {
      navigate('/official-login');
    }
  };

  return (
    <div className="first-page">
      <h1>Welcome to FindThemNow</h1>
      <div className="role-selection">
        <button onClick={() => handleRoleSelection('user')}>User</button>
        <button onClick={() => handleRoleSelection('official')}>Official</button>
      </div>
    </div>
  );
}

export default FirstPage;