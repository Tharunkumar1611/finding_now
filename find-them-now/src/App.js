import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import FirstPage from './components/FirstPage';
import UserLogin from './components/UserLogin';
import OfficialLogin from './components/OfficialLogin';
import UserDashboard from './components/UserDashboard';
import OfficialDashboard from './components/OfficialDashboard';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<FirstPage />} />
        <Route path="/user-login" element={<UserLogin />} />
        <Route path="/official-login" element={<OfficialLogin />} />
        <Route path="/user-dashboard" element={<UserDashboard />} />
        <Route path="/official-dashboard" element={<OfficialDashboard />} />
      </Routes>
    </Router>
  );
}

export default App;