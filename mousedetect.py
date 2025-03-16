import pyautogui
import time
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import os
from pynput import mouse
import threading

class MouseDetector:
    def __init__(self, model_file="mouse_model.pkl", window_size=20, threshold=0.6):
        self.model_file = model_file
        self.window_size = window_size  # Number of events to consider for detection
        self.threshold = threshold  # Threshold for suspicious classification
        self.current_sequence = []
        self.running = False
        self.detection_thread = None
        
        # Load the model
        if os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {self.model_file}")
        else:
            print(f"Model file {self.model_file} not found. Please train the model first.")
            self.model = None
    
    def start(self):
        """Start monitoring mouse activity"""
        if self.model is None:
            print("Cannot start detection without a trained model")
            return
        
        print("Mouse Activity Detector started")
        print("Monitoring for suspicious mouse behavior...")
        
        # Start the listener
        self.running = True
        self.current_sequence = []
        
        listener = mouse.Listener(on_move=self.on_move, on_click=self.on_click)
        listener.start()
        
        # Start the detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        try:
            # Wait for user to stop
            while True:
                cmd = input("Type 'quit' to exit: ").strip().lower()
                if cmd == "quit":
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            listener.stop()
            if self.detection_thread:
                self.detection_thread.join(timeout=1)
            print("Mouse Activity Detector stopped")
    
    def on_move(self, x, y):
        """Callback for mouse movement"""
        if not self.running:
            return
        
        timestamp = datetime.now().timestamp()
        event = {
            'timestamp': timestamp,
            'x': x,
            'y': y,
            'event_type': 'move'
        }
        self.current_sequence.append(event)
    
    def on_click(self, x, y, button, pressed):
        """Callback for mouse clicks"""
        if not self.running or not pressed:
            return
        
        timestamp = datetime.now().timestamp()
        event = {
            'timestamp': timestamp,
            'x': x,
            'y': y,
            'event_type': 'click',
            'button': str(button)
        }
        self.current_sequence.append(event)
    
    def detection_loop(self):
        """Periodically analyze the current sequence for suspicious behavior"""
        while self.running:
            if len(self.current_sequence) >= self.window_size:
                self.analyze_current_sequence()
            time.sleep(1)  # Check every second
    
    def analyze_current_sequence(self):
        """Process the current sequence and detect suspicious behavior"""
        # Keep only the most recent events
        if len(self.current_sequence) > self.window_size:
            self.current_sequence = self.current_sequence[-self.window_size:]
        
        if len(self.current_sequence) < 2:
            return
        
        # Process the data
        processed_data = []
        
        for i in range(1, len(self.current_sequence)):
            prev = self.current_sequence[i-1]
            curr = self.current_sequence[i]
            
            # Calculate features
            duration = curr['timestamp'] - prev['timestamp']
            distance = np.sqrt((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)
            speed = distance / duration if duration > 0 else 0
            
            # Calculate direction change
            direction_change = 0
            if i >= 2:
                prev_prev = self.current_sequence[i-2]
                
                v1_x = prev['x'] - prev_prev['x']
                v1_y = prev['y'] - prev_prev['y']
                
                v2_x = curr['x'] - prev['x']
                v2_y = curr['y'] - prev['y']
                
                dot_product = v1_x * v2_x + v1_y * v2_y
                mag_v1 = np.sqrt(v1_x**2 + v1_y**2)
                mag_v2 = np.sqrt(v2_x**2 + v2_y**2)
                
                if mag_v1 > 0 and mag_v2 > 0:
                    cos_angle = dot_product / (mag_v1 * mag_v2)
                    cos_angle = max(-1.0, min(1.0, cos_angle))
                    angle = np.arccos(cos_angle)
                    direction_change = angle
            
            processed_data.append([duration, distance, speed, direction_change])
        
        # Make predictions
        if processed_data:
            X = np.array(processed_data)
            predictions = self.model.predict(X)
            suspicious_count = sum(1 for p in predictions if p == 'suspicious')
            suspicious_ratio = suspicious_count / len(predictions)
            
            # Check if the ratio exceeds the threshold
            if suspicious_ratio >= self.threshold:
                print("\n[ALERT] Suspicious!")
                print(f"Suspicious movements: {suspicious_count}/{len(predictions)} ({suspicious_ratio:.2f})")
                print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
            else:
                print("Normal")

def main():
    detector = MouseDetector()
    detector.start()

if __name__ == "__main__":
    main()