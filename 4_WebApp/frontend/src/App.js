import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [currentSign, setCurrentSign] = useState("...");
  const [sentence, setSentence] = useState("");
  const [isSpeaking, setIsSpeaking] = useState(false);
  
  // "Memory" variables (Ref prevents re-rendering)
  const lastSign = useRef("...");
  const holdTimer = useRef(null);

  // --- 1. VOICE FUNCTION ---
  const speakText = (text) => {
    if (!text) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    window.speechSynthesis.speak(utterance);
    setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
  };

  // --- 2. FETCH PREDICTION & LOGIC ---
  useEffect(() => {
    // Helper function: Adds char to sentence safely
    const addToSentence = (char) => {
      setSentence(prev => prev + char);
    };

    // Helper function: Checks if we should add the sign
    const processPrediction = (prediction) => {
      // If the sign changed, reset the timer
      if (prediction !== lastSign.current) {
        lastSign.current = prediction;
        if (holdTimer.current) clearTimeout(holdTimer.current);

        // Start a new timer: If hold for 1.2 seconds, add to sentence
        holdTimer.current = setTimeout(() => {
          if (prediction !== "..." && prediction !== "Waiting...") {
            addToSentence(prediction);
          }
        }, 1200);
      }
    };

    const interval = setInterval(() => {
      fetch('http://localhost:5000/get_prediction')
        .then(response => response.json())
        .then(data => {
          if (data.prediction) {
            setCurrentSign(data.prediction);
            processPrediction(data.prediction); // Call the internal function
          }
        })
        .catch(err => console.error("Backend Error:", err));
    }, 200);

    return () => {
      clearInterval(interval);
      if (holdTimer.current) clearTimeout(holdTimer.current);
    };
  }, []); // Empty dependency array is now correct because functions are inside

  return (
    <div className="app-container">
      <div className="gradient-bg"></div>
      <header className="header">
        <h1>Mudra<span className="highlight">Vani</span> 🖐️</h1>
        <p>Sign-to-Sentence Converter</p>
      </header>

      <div className="main-content">
        {/* CAMERA SECTION */}
        <div className="glass-panel camera-box">
          <div className="live-badge">🔴 LIVE</div>
          <img src="http://localhost:5000/video_feed" alt="feed" className="video-stream"/>
          
          {/* Overlay the Current Letter */}
          <div className="overlay-letter">{currentSign}</div>
        </div>

        {/* SENTENCE OUTPUT SECTION */}
        <div className="glass-panel output-box">
          <h3>Constructed Sentence</h3>
          
          <textarea 
            className="sentence-display"
            value={sentence}
            readOnly
            placeholder="Start signing to build a sentence..."
          />

          <div className="controls">
            <button className="action-btn space" onClick={() => setSentence(s => s + " ")}>
              Space ␣
            </button>
            <button className="action-btn backspace" onClick={() => setSentence(s => s.slice(0, -1))}>
              ⌫ Back
            </button>
            <button className="action-btn clear" onClick={() => setSentence("")}>
              Clear
            </button>
          </div>

          <button 
            className={`speak-btn ${isSpeaking ? 'speaking' : ''}`}
            onClick={() => speakText(sentence)}
          >
            🔊 Speak Full Sentence
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;