import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './UserLogin.css';

function UserLogin() {
  const [user, setUser] = useState({
    username: '',
    email: '',
    password: '',
    phone_number: '', // Match backend field name
    address: ''
  });

  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [registerError, setRegisterError] = useState('');
  const [showRegister, setShowRegister] = useState(false);

  const navigate = useNavigate();

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setUser({
      ...user,
      [name]: value
    });
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    console.log('Register Payload:', user); // Debugging
    try {
      const response = await axios.post('http://localhost:8080/users/register', user);
      if (response.data) {
        alert('User registered successfully!');
        setShowRegister(false);
      } else {
        setRegisterError('Failed to register user.');
      }
    } catch (error) {
      console.error('There was an error!', error);
      setRegisterError('Failed to register user. Please check the console for details.');
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8080/users/login', {
        email: loginEmail,
        password: loginPassword
      });

      console.log('Login Response:', response.data); // Debugging

      if (response.data) {
        alert('User logged in successfully!');
        navigate('/user-dashboard', { state: { username: response.data.username } });
      } else {
        setLoginError('Invalid email or password');
      }
    } catch (error) {
      console.error('There was an error!', error);
      setLoginError('Failed to log in. Please try again.');
    }
  };

  return (
    <div className="user-login">
      <h2>User Login</h2>
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
          New User? <span>Register</span>
        </p>
      )}

      {showRegister && (
        <>
          <h2>User Registration</h2>
          <form onSubmit={handleRegister}>
            <input
              type="text"
              name="username"
              value={user.username}
              onChange={handleInputChange}
              placeholder="Username"
              required
            />
            <input
              type="email"
              name="email"
              value={user.email}
              onChange={handleInputChange}
              placeholder="Email"
              required
            />
            <input
              type="password"
              name="password"
              value={user.password}
              onChange={handleInputChange}
              placeholder="Password"
              required
            />
            <input
              type="text"
              name="phone_number"
              value={user.phone_number}
              onChange={handleInputChange}
              placeholder="Phone Number"
              required
            />
            <input
              type="text"
              name="address"
              value={user.address}
              onChange={handleInputChange}
              placeholder="Address"
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

export default UserLogin;