import React, { useState, useEffect } from "react";
import axios from "axios";
import { useLocation } from "react-router-dom";
import "./OfficialDashboard.css";

const OfficialDashboard = () => {
  const location = useLocation();
  const { name } = location.state || { name: "Official" };

  // State for ML model functionality
  const [video, setVideo] = useState(null);
  const [image, setImage] = useState(null);
  const [result, setResult] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // State for missing person details
  const [missingPersonData, setMissingPersonData] = useState({
    name: "",
    age: "",
    gender: "",
    lastSeenLocation: "",
  });

  // State for user reports (comments)
  const [reports, setReports] = useState([]);

  // Fetch user reports (comments) from the backend
  useEffect(() => {
    axios.get("http://localhost:8080/reports")
      .then(response => {
        setReports(response.data);
      })
      .catch(error => console.error("Error fetching reports:", error));
  }, []);

  // ML model handlers
  const handleVideoChange = (e) => {
    const file = e.target.files[0];
    if (file) setVideo(file);
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) setImage(file);
  };

  const handleMLSubmit = async (e) => {
    e.preventDefault();
    if (!video || !image) {
      setResult("Please upload both video and image.");
      return;
    }

    const formData = new FormData();
    formData.append("video", video);
    formData.append("photo", image);

    try {
      setIsLoading(true);
      const response = await axios.post("http://127.0.0.1:5000/check-face", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const data = response.data;
      if (data.result === "Face Found") {
        setResult(`✅ Match Found in Video at ${data.timestamp}`);
      } else {
        setResult("❌ Match Not Found in Video.");
      }
    } catch (error) {
      console.error("Error:", error);
      setResult("Error processing the files. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Missing person details handlers
  const handleMissingPersonChange = (e) => {
    const { name, value } = e.target;
    setMissingPersonData({ ...missingPersonData, [name]: value });
  };

  const handleMissingPersonSubmit = async (e) => {
    e.preventDefault();
    const data = {
      name: missingPersonData.name,
      age: missingPersonData.age,
      gender: missingPersonData.gender,
      lastSeenLocation: missingPersonData.lastSeenLocation,
      reportedBy: 15, // Replace with actual official ID
    };

    try {
      await axios.post("http://localhost:8080/missingpersons/add", data, {
        headers: { "Content-Type": "application/json" },
      });
      alert("Missing person details added successfully!");
    } catch (error) {
      console.error("Error:", error);
      alert("Failed to add missing person details.");
    }
  };

  return (
    <div className="official-dashboard">
      <div className="greeting">
        <h1><span className="greeting-text">Hello, {name}!</span></h1>
      </div>

      {/* ML Model Section */}
      <div className="container">
        <h2>Face Recognition</h2>
        <form onSubmit={handleMLSubmit}>
          <div>
            <label htmlFor="video-upload">Upload Video:</label>
            <input
              id="video-upload"
              type="file"
              accept="video/mp4,video/quicktime,video/x-msvideo"
              onChange={handleVideoChange}
              required
            />
          </div>
          <div>
            <label htmlFor="image-upload">Upload Image:</label>
            <input
              id="image-upload"
              type="file"
              accept="image/jpeg,image/jpg,image/png"
              onChange={handleImageChange}
              required
            />
          </div>
          <button type="submit" disabled={isLoading}>
            {isLoading ? "Uploading..." : "Upload"}
          </button>
        </form>
        {result && (
          <p className={result.includes("Match Found") ? "success" : "error"}>{result}</p>
        )}
      </div>

      {/* Missing Person Section */}
      <div className="container">
        <h2>Add Missing Person</h2>
        <form onSubmit={handleMissingPersonSubmit}>
          <div>
            <label>Name:</label>
            <input type="text" name="name" value={missingPersonData.name} onChange={handleMissingPersonChange} required />
          </div>
          <div>
            <label>Age:</label>
            <input type="number" name="age" value={missingPersonData.age} onChange={handleMissingPersonChange} required />
          </div>
          <div>
            <label>Gender:</label>
            <input type="text" name="gender" value={missingPersonData.gender} onChange={handleMissingPersonChange} required />
          </div>
          <div>
            <label>Last Seen Location:</label>
            <input type="text" name="lastSeenLocation" value={missingPersonData.lastSeenLocation} onChange={handleMissingPersonChange} required />
          </div>
          <button type="submit">Submit</button>
        </form>
      </div>

      {/* User Reports Section */}
      <div className="container">
        <h2>User Reports (Comments)</h2>
        {reports.map(report => (
          <div key={report.reportId} className="report-box">
            <h3>Report ID: {report.reportId}</h3>
            <p>Location: {report.location}</p>
            <p>Timestamp: {report.timestamp}</p>
            {report.description && (
              <div className="comment-section">
                <h4>Comment:</h4>
                <p style={{color:'white',backgroundColor:'#3a3a3a'}}>{report.description}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default OfficialDashboard;
