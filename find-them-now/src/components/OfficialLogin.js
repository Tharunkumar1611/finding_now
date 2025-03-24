import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './OfficialLogin.css';

function OfficialLogin() {
  const [official, setOfficial] = useState({
    name: '',
    email: '',
    password: '',
    phoneNumber: ''
  });

  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [registerError, setRegisterError] = useState('');
  const [showRegister, setShowRegister] = useState(false);

  const navigate = useNavigate();

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setOfficial({
      ...official,
      [name]: value
    });
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8080/officials/register', official);
      if (response.data.success) {
        alert('Official registered successfully!');
        setShowRegister(false);
      } else {
        setRegisterError(response.data.message || 'Failed to register official.');
      }
    } catch (error) {
      console.error('There was an error!', error);
      setRegisterError('Failed to register official. Please check the console for details.');
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8080/officials/login', {
        email: loginEmail,
        password: loginPassword
      });

      console.log('Login Response:', response.data); // Debugging

      if (response.data.success) {
        alert('Official logged in successfully!');
        navigate('/official-dashboard', { state: { name: response.data.official.name } });
      } else {
        setLoginError(response.data.message || 'Invalid email or password');
      }
    } catch (error) {
      console.error('There was an error!', error);
      setLoginError('Failed to log in. Please try again.');
    }
  };

  return (
    <div className="official-login">
      <h2>Official Login</h2>
      <form onSubmit={handleLogin}>
        <input
          type="email"
          placeholder="Email"
          value={loginEmail}
          onChange={(e) => setLoginEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={loginPassword}
          onChange={(e) => setLoginPassword(e.target.value)}
          required
        />
        <button type="submit">Login</button>
      </form>

      {loginError && <p className="error-message">{loginError}</p>}

      {!showRegister && (
        <p className="register-link" onClick={() => setShowRegister(true)}>
          New Official? <span>Register</span>
        </p>
      )}

      {showRegister && (
        <>
          <h2>Official Registration</h2>
          <form onSubmit={handleRegister}>
            <input
              type="text"
              name="name"
              value={official.name}
              onChange={handleInputChange}
              placeholder="Name"
              required
            />
            <input
              type="email"
              name="email"
              value={official.email}
              onChange={handleInputChange}
              placeholder="Email"
              required
            />
            <input
              type="password"
              name="password"
              value={official.password}
              onChange={handleInputChange}
              placeholder="Password"
              required
            />
            <input
              type="text"
              name="phoneNumber"
              value={official.phoneNumber}
              onChange={handleInputChange}
              placeholder="Phone Number"
              required
            />
            <button type="submit">Register</button>
          </form>

          <p className="login-link" onClick={() => setShowRegister(false)}>
            Already have an account? <span>Login</span>
          </p>
        </>
      )}

      {registerError && <p className="error-message">{registerError}</p>}
    </div>
  );
}

export default OfficialLogin;