import pyautogui
import time
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from pynput import mouse

class MouseTracker:
    def __init__(self, data_file="mouse_data.csv", model_file="mouse_model.pkl"):
        self.data_file = data_file
        self.model_file = model_file
        self.movements = []
        self.current_sequence = []
        self.recording = False
        self.label = None
        
        # Create or load the dataframe
        if os.path.exists(self.data_file):
            self.df = pd.read_csv(self.data_file)
        else:
            self.df = pd.DataFrame(columns=['timestamp', 'x', 'y', 'event_type', 'duration', 
                                           'distance', 'speed', 'direction_change', 'label'])
    
    def start(self):
        """Start tracking mouse movements and listen for commands"""
        print("Mouse Tracker started")
        print("Type 'sus' to label movements as suspicious")
        print("Type 'norm' to label movements as normal")
        print("Type 'quit' to exit")
        
        # Start the listener in a non-blocking way
        listener = mouse.Listener(on_move=self.on_move, on_click=self.on_click)
        listener.start()
        
        self.recording = True
        self.current_sequence = []
        
        try:
            while True:
                cmd = input("> ").strip().lower()
                if cmd == "sus":
                    self.save_sequence("suspicious")
                    print("Movements labeled as suspicious")
                elif cmd == "norm":
                    self.save_sequence("normal")
                    print("Movements labeled as normal")
                elif cmd == "quit":
                    break
                else:
                    print("Unknown command")
        except KeyboardInterrupt:
            pass
        finally:
            self.recording = False
            listener.stop()
            print("Mouse Tracker stopped")
    
    def on_move(self, x, y):
        """Callback for mouse movement"""
        if not self.recording:
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
        if not self.recording:
            return
            
        if pressed:  # Only record press events, not releases
            timestamp = datetime.now().timestamp()
            event = {
                'timestamp': timestamp,
                'x': x,
                'y': y,
                'event_type': 'click',
                'button': str(button)
            }
            self.current_sequence.append(event)
    
    def save_sequence(self, label):
        """Process the current sequence and save it with the given label"""
        if len(self.current_sequence) < 2:
            print("Not enough data to save")
            return
        
        processed_data = self.process_sequence(self.current_sequence, label)
        self.df = pd.concat([self.df, pd.DataFrame(processed_data)], ignore_index=True)
        self.df.to_csv(self.data_file, index=False)
        self.current_sequence = []  # Reset current sequence
    
    def process_sequence(self, sequence, label):
        """Extract features from raw movement data"""
        processed_data = []
        
        for i in range(1, len(sequence)):
            prev = sequence[i-1]
            curr = sequence[i]
            
            # Calculate time delta
            duration = curr['timestamp'] - prev['timestamp']
            
            # Calculate distance
            distance = np.sqrt((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)
            
            # Calculate speed
            speed = distance / duration if duration > 0 else 0
            
            # Calculate direction change if we have at least 3 points
            direction_change = 0
            if i >= 2:
                prev_prev = sequence[i-2]
                
                # Vector from prev_prev to prev
                v1_x = prev['x'] - prev_prev['x']
                v1_y = prev['y'] - prev_prev['y']
                
                # Vector from prev to curr
                v2_x = curr['x'] - prev['x']
                v2_y = curr['y'] - prev['y']
                
                # Calculate the dot product and magnitudes
                dot_product = v1_x * v2_x + v1_y * v2_y
                mag_v1 = np.sqrt(v1_x**2 + v1_y**2)
                mag_v2 = np.sqrt(v2_x**2 + v2_y**2)
                
                # Calculate the angle between the vectors
                if mag_v1 > 0 and mag_v2 > 0:
                    cos_angle = dot_product / (mag_v1 * mag_v2)
                    # Clamp to handle floating point errors
                    cos_angle = max(-1.0, min(1.0, cos_angle))
                    angle = np.arccos(cos_angle)
                    direction_change = angle
            
            processed_data.append({
                'timestamp': curr['timestamp'],
                'x': curr['x'],
                'y': curr['y'],
                'event_type': curr['event_type'],
                'duration': duration,
                'distance': distance,
                'speed': speed,
                'direction_change': direction_change,
                'label': label
            })
        
        return processed_data
    
    def train_model(self):
        """Train a Random Forest model on the collected data"""
        if len(self.df) == 0:
            print("No data available for training")
            return None
        
        # Ensure we have both classes
        labels = self.df['label'].unique()
        if len(labels) < 2:
            print(f"Need both 'suspicious' and 'normal' data. Currently only have: {labels}")
            return None
        
        # Prepare features and labels
        features = self.df[['duration', 'distance', 'speed', 'direction_change']].values
        labels = self.df['label'].values
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, labels)
        
        # Save the model
        with open(self.model_file, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model trained and saved to {self.model_file}")
        return model

def main():
    tracker = MouseTracker()
    tracker.start()
    tracker.train_model()

if __name__ == "__main__":
    main()