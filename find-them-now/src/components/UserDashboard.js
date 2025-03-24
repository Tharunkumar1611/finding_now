import React, { useEffect, useState, useMemo } from 'react';
import axios from 'axios';
import { useLocation } from 'react-router-dom';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import './UserDashboard.css';
import badge1 from './badge1.png'; // Import the badge image

// Fix for default marker icons in Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

function UserDashboard() {
  const location = useLocation();
  const { username } = location.state || { username: 'User' };
  const [missingPersons, setMissingPersons] = useState([]);
  const [comments, setComments] = useState({});
  const [personImages, setPersonImages] = useState({});
  const [searchQuery, setSearchQuery] = useState('');
  const [route, setRoute] = useState([]); // State for the route coordinates
  const [routeDescription, setRouteDescription] = useState(''); // State for the route description
  const [source, setSource] = useState(''); // State for source input
  const [destination, setDestination] = useState(''); // State for destination input

  // Memoize the randomImages array to avoid recreating it on each render
  const randomImages = useMemo(() => [
    require('./child1.jpg'),
    require('./child2.jpg'),
    require('./child3.jpg'),
    require('./child4.jpg'),
    require('./child5.jpg')
  ], []);

  // Predefined routes
  const predefinedRoutes = [
    {
      source: 'Gandhipuram',
      destination: 'Ukkadam',
      via: 'RS Puram',
      coordinates: [
        [11.0168, 76.9558], // Gandhipuram
        [11.0180, 76.9570], // RS Puram
        [11.0200, 76.9600], // Ukkadam
      ],
    },
    {
      source: 'Peelamedu',
      destination: 'Avinashi Road',
      via: 'Gandhipuram',
      coordinates: [
        [11.0300, 76.9600], // Peelamedu
        [11.0168, 76.9558], // Gandhipuram
        [11.0100, 76.9500], // Avinashi Road
      ],
    },
    {
      source: 'RS Puram',
      destination: 'Peelamedu',
      via: 'Gandhipuram',
      coordinates: [
        [11.0180, 76.9570], // RS Puram
        [11.0168, 76.9558], // Gandhipuram
        [11.0300, 76.9600], // Peelamedu
      ],
    },
    {
      source: 'Ukkadam',
      destination: 'Avinashi Road',
      via: 'Racecourse',
      coordinates: [
        [11.0200, 76.9600], // Ukkadam
        [11.0168, 76.9558], // Gandhipuram
        [11.0100, 76.9500], // Avinashi Road
      ],
    },
    {
      source: 'Gandhipuram',
      destination: 'Peelamedu',
      via: 'Hope college',
      coordinates: [
        [11.0168, 76.9558], // Gandhipuram
        [11.0180, 76.9570], // RS Puram
        [11.0300, 76.9600], // Peelamedu
      ],
    },
    {
      source: 'Avinashi Road',
      destination: 'Ukkadam',
      via: 'Lakshmi Mills',
      coordinates: [
        [11.0100, 76.9500], // Avinashi Road
        [11.0168, 76.9558], // Gandhipuram
        [11.0200, 76.9600], // Ukkadam
      ],
    },
    {
      source: 'Peelamedu',
      destination: 'Ukkadam',
      via: 'Lakshmi Mills',
      coordinates: [
        [11.0300, 76.9600], // Peelamedu
        [11.0180, 76.9570], // RS Puram
        [11.0200, 76.9600], // Ukkadam
      ],
    },
    {
      source: 'RS Puram',
      destination: 'Avinashi Road',
      via: 'Gandhipuram',
      coordinates: [
        [11.0180, 76.9570], // RS Puram
        [11.0168, 76.9558], // Gandhipuram
        [11.0100, 76.9500], // Avinashi Road
      ],
    },
    {
      source: 'Ukkadam',
      destination: 'R S Puram',
      via: 'Selvapuram',
      coordinates: [
        [11.0200, 76.9600], // Ukkadam
        [11.0180, 76.9570], // RS Puram
        [11.0300, 76.9600], // Peelamedu
      ],
    },
    {
      source: 'Avinashi Road',
      destination: 'RS Puram',
      via: 'Gandhipuram',
      coordinates: [
        [11.0100, 76.9500], // Avinashi Road
        [11.0168, 76.9558], // Gandhipuram
        [11.0180, 76.9570], // RS Puram
      ],
    },
  ];

  // Fetch all missing persons and assign images to them
  useEffect(() => {
    axios.get("http://localhost:8080/missingpersons")
      .then(response => {
        const persons = response.data;
        setMissingPersons(persons);

        // Assign a unique image to each person ID
        const images = {};
        persons.forEach((person, index) => {
          images[person.id] = randomImages[index % randomImages.length];
        });
        setPersonImages(images);
      })
      .catch(error => console.error("Error fetching data:", error));
  }, [randomImages]);

  // Handle comment submission
  const handleCommentSubmit = async (personId) => {
    try {
      await axios.post("http://localhost:8080/reports/add", {
        userId: 1,
        description: comments[personId] || "",
        location: "Coimbatore",
        timestamp: new Date().toISOString(),
      });
      alert("Comment added successfully!");
      setComments(prevComments => ({ ...prevComments, [personId]: '' }));
    } catch (error) {
      console.error("Error adding comment:", error);
      alert("Failed to add comment.");
    }
  };

  // Handle comment change
  const handleCommentChange = (personId, value) => {
    setComments(prevComments => ({ ...prevComments, [personId]: value }));
  };

  // Generate a route based on source and destination
  const generateRoute = () => {
    if (!source || !destination) {
      alert('Please enter both source and destination.');
      return;
    }

    // Find the predefined route
    const route = predefinedRoutes.find(
      (r) =>
        r.source.toLowerCase() === source.toLowerCase() &&
        r.destination.toLowerCase() === destination.toLowerCase()
    );

    if (route) {
      setRoute(route.coordinates);
      setRouteDescription(`Go via ${route.via} to reach ${destination}.`);
    } else {
      alert('No predefined route found for the given source and destination.');
    }
  };

  // Filter missing persons based on search query
  const filteredPersons = missingPersons.filter(person =>
    person.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="user-dashboard">
      {/* Top Bar with Greeting, Search Bar, and Badges */}
      <div className="top-bar">
        <div className="greeting">
          <h1>Hello, {username}!</h1>
          <p>Welcome to your dashboard.</p>
        </div>
        <div className="search-bar">
          <input
            type="text"
            placeholder="Search missing persons..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <div className="badges">
          <img src={badge1} alt="Badge" className="badge-image" />
          <span className="badge">Helper</span>
          <span className="badge">Verified</span>
        </div>
      </div>

      {/* Awareness Section for Do's and Don'ts */}
      <div className="awareness-section">
        <h2>Do's and Don'ts</h2>
        <ul>
          <li>✅ Do share missing person alerts on social media.</li>
          <li>✅ Do report any sightings to the authorities.</li>
          <li>✅ Do keep recent photos of your loved ones.</li>
          <li>✅ Do teach children about stranger danger and safe practices.</li>
          <li>✅ Do verify information before sharing it.</li>
          <li>❌ Don't share unverified information.</li>
          <li>❌ Don't approach suspicious individuals alone.</li>
          <li>❌ Don't post sensitive personal details publicly.</li>
          <li>❌ Don't ignore missing person alerts in your area.</li>
        </ul>
      </div>

      {/* Map Section */}
      <div className="map-container">
        <h2>Route Map</h2>
        <div className="route-inputs">
          <input
            type="text"
            placeholder="Source"
            value={source}
            onChange={(e) => setSource(e.target.value)}
          />
          <input
            type="text"
            placeholder="Destination"
            value={destination}
            onChange={(e) => setDestination(e.target.value)}
          />
          <button onClick={generateRoute}>Get Route</button>
        </div>
        <MapContainer
          center={[11.0168, 76.9558]} // Coimbatore coordinates
          zoom={13} // Adjust zoom level as needed
          style={{ height: '400px', width: '100%', borderRadius: '12px' }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          {route.length > 0 && (
            <>
              <Polyline positions={route} color="blue" />
              <Marker position={route[0]}>
                <Popup>Start: {source}</Popup>
              </Marker>
              <Marker position={route[route.length - 1]}>
                <Popup>End: {destination}</Popup>
              </Marker>
            </>
          )}
        </MapContainer>
        {routeDescription && (
          <div className="route-description">
            <p>{routeDescription}</p>
          </div>
        )}
      </div>

      {/* Missing Persons Container */}
      <div className="missing-persons-container">
        {filteredPersons.map(person => (
          <div key={person.id} className="missing-person-box">
            <h3>{person.name}</h3>
            <p>Age: {person.age}</p>
            <p>Gender: {person.gender}</p>
            <p>Last Seen: {person.lastSeenLocation}</p>
            
            {person.image && (
              <img
                className="missing-person-image"
                src={`data:image/jpeg;base64,${person.image}`}
                alt={person.name}
              />
            )}
            
            {personImages[person.id] && (
              <img
                className="child-image"
                src={personImages[person.id]}
                alt="Child"
              />
            )}
            
            <div className="comment-section">
              <textarea
                placeholder="Add a comment..."
                value={comments[person.id] || ''}
                onChange={(e) => handleCommentChange(person.id, e.target.value)}
              />
              <button onClick={() => handleCommentSubmit(person.id)}>Submit Comment</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default UserDashboard;