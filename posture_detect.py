import numpy as np
import time
import os
import sys
import pickle
import joblib
import subprocess
import re
from collections import deque
import threading
import curses

class PostureDetector:
    def __init__(self):
        self.posture_labels = ["sitting", "standing", "walking"]
        self.current_posture = "unknown"
        self.confidence = 0.0
        self.buffer_size = 10
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.model_dir = "models"
        self.running = False
        self.history = []  # Store posture history for visualization
        self.lock = threading.Lock()  # For thread safety
        
        # Load the model and preprocessing function
        self.load_model()
    
    def load_model(self):
        """Load the latest trained model"""
        model_path = os.path.join(self.model_dir, "latest_model.joblib")
        preprocess_path = os.path.join(self.model_dir, "preprocess_func.pkl")
        
        try:
            if not os.path.exists(model_path):
                print("Error: Model file not found. Please train a model first.")
                print("Run: python posture_train.py")
                sys.exit(1)
                
            if not os.path.exists(preprocess_path):
                print("Error: Preprocessing function not found.")
                print("Run: python posture_train.py")
                sys.exit(1)
            
            self.classifier = joblib.load(model_path)
            with open(preprocess_path, 'rb') as f:
                self.preprocess_csi = pickle.load(f)
            
            print(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def extract_wifi_csi(self):
        """
        Extract Channel State Information (CSI) from Wi-Fi signal
        This is a simplified version that uses signal strength as a proxy
        """
        try:
            # Try to get actual signal strength data if possible
            # This works on Linux systems
            try:
                output = subprocess.check_output("iwconfig 2>/dev/null | grep -i signal", shell=True).decode()
                signal_level = re.search(r"Signal level[=:](-\d+) dBm", output)
                if signal_level:
                    base_signal = float(signal_level.group(1))
                else:
                    base_signal = -65  # Default value if we can't extract
            except:
                base_signal = -65  # Default value if command fails
            
            # For real CSI we would have multiple subcarrier values
            # Here we simulate them by adding variations around the base signal
            variations = np.random.normal(0, 2, 30)
            csi_values = base_signal + variations
            
            # Modify the variations based on the current posture for the demo
            # In real usage, we wouldn't do this since we'd have real CSI data
            if self.current_posture == "sitting":
                # More stable signal
                csi_values = base_signal + np.random.normal(0, 2, 30)
            elif self.current_posture == "standing":
                # Different baseline with moderate variations
                csi_values = base_signal - 5 + np.random.normal(0, 3, 30)
            elif self.current_posture == "walking":
                # More dynamic with periodic components
                walking_pattern = 3 * np.sin(np.linspace(time.time() % (2*np.pi), 
                                                        (time.time() % (2*np.pi)) + 2*np.pi, 30))
                csi_values = base_signal + 3 + np.random.normal(0, 5, 30) + walking_pattern
            
            return csi_values
            
        except Exception as e:
            print(f"Error extracting CSI data: {e}")
            return np.ones(30) * -75  # Return default value if extraction fails
    
    def detect_posture(self):
        """Detect current posture using the trained model"""
        csi_values = self.extract_wifi_csi()
        features = self.preprocess_csi(csi_values)
        
        # Add to buffer for smoothing predictions
        with self.lock:
            self.data_buffer.append(features)
        
        # Make predictions only if we have enough data
        if len(self.data_buffer) >= 3:
            recent_features = list(self.data_buffer)[-5:]
            predictions = self.classifier.predict(recent_features)
            
            # Get prediction probabilities
            proba = self.classifier.predict_proba(recent_features)
            
            # Simple majority voting
            unique, counts = np.unique(predictions, return_counts=True)
            prediction_counts = dict(zip(unique, counts))
            detected_posture = max(prediction_counts, key=prediction_counts.get)
            
            # Calculate confidence as the percentage of the majority vote
            confidence = prediction_counts[detected_posture] / len(predictions)
            
            with self.lock:
                self.current_posture = detected_posture
                self.confidence = confidence
                
                # Save to history (limited to 500 entries)
                self.history.append((detected_posture, confidence))
                if len(self.history) > 500:
                    self.history.pop(0)
            
            return detected_posture, confidence
        
        return "unknown", 0.0
    
    def start_realtime_detection(self):
        """Start real-time posture detection in a separate thread"""
        self.running = True
        
        # Start detection in a background thread
        detection_thread = threading.Thread(target=self._detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
    
    def _detection_loop(self):
        """Background thread for continuous detection"""
        while self.running:
            self.detect_posture()
            time.sleep(0.1)  # 10 times per second
    
    def stop_detection(self):
        """Stop the real-time detection"""
        self.running = False

def draw_ui(stdscr, detector):
    """Draw the user interface using curses"""
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_RED, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)
    curses.init_pair(5, curses.COLOR_MAGENTA, -1)
    
    # Hide cursor
    curses.curs_set(0)
    
    # Get terminal size
    height, width = stdscr.getmaxyx()
    
    # Main loop
    while detector.running:
        try:
            stdscr.clear()
            
            # Title
            title = "Wi-Fi CSI Posture Detection"
            stdscr.addstr(1, max(0, (width - len(title)) // 2), title, curses.A_BOLD)
            
            # Status
            with detector.lock:
                posture = detector.current_posture
                confidence = detector.confidence
            
            # Choose color based on confidence
            color = curses.color_pair(1) if confidence > 0.8 else \
                   curses.color_pair(2) if confidence > 0.6 else \
                   curses.color_pair(3)
            
            # Display current posture
            posture_text = f"Current posture: {posture.upper()}"
            confidence_text = f"Confidence: {confidence:.2f}"
            stdscr.addstr(3, 2, posture_text, curses.A_BOLD | color)
            stdscr.addstr(4, 2, confidence_text)
            
            # Draw a simple histogram of recent postures
            if detector.history:
                stdscr.addstr(6, 2, "Recent posture history:", curses.A_BOLD)
                
                # Count occurrences of each posture in the last 100 readings
                recent_history = detector.history[-100:] if len(detector.history) >= 100 else detector.history
                posture_counts = {posture: 0 for posture in detector.posture_labels}
                for p, _ in recent_history:
                    if p in posture_counts:
                        posture_counts[p] += 1
                
                # Draw small bar chart
                for i, posture in enumerate(detector.posture_labels):
                    bar_color = curses.color_pair(i+1) if i < 5 else curses.color_pair(5)
                    count = posture_counts[posture]
                    percentage = count / len(recent_history) if recent_history else 0
                    bar_length = int(percentage * 40)
                    
                    if height > 8 + i and width > 50:
                        stdscr.addstr(8 + i, 2, f"{posture.ljust(10)}: ", curses.A_BOLD)
                        stdscr.addstr(8 + i, 14, "█" * bar_length, bar_color)
                        stdscr.addstr(8 + i, 14 + 40 + 2, f"{percentage:.1%}")
            
            # Instructions
            if height > 15:
                stdscr.addstr(height - 3, 2, "Press 's' to simulate sitting", curses.color_pair(4))
                stdscr.addstr(height - 2, 2, "Press 't' to simulate standing", curses.color_pair(4))
                stdscr.addstr(height - 1, 2, "Press 'w' to simulate walking", curses.color_pair(4))
                stdscr.addstr(height - 1, width - 18, "Press 'q' to quit", curses.color_pair(4))
            
            stdscr.refresh()
            
            # Check for user input (non-blocking)
            stdscr.timeout(100)
            key = stdscr.getch()
            
            if key == ord('q'):
                detector.stop_detection()
                break
            elif key == ord('s'):
                detector.current_posture = "sitting"
            elif key == ord('t'):
                detector.current_posture = "standing"
            elif key == ord('w'):
                detector.current_posture = "walking"
                
        except Exception as e:
            # In case of error, stop properly
            detector.stop_detection()
            curses.endwin()
            print(f"Error in UI: {e}")
            break

def main():
    print("Initializing posture detector...")
    detector = PostureDetector()
    
    print("Starting real-time detection...")
    detector.start_realtime_detection()
    
    print("Launching UI...")
    try:
        # Initialize curses
        curses.wrapper(draw_ui, detector)
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure to stop detection thread
        detector.stop_detection()
        print("Posture detection stopped.")

if __name__ == "__main__":
    main()