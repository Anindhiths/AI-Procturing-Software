import os
import time
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
from pynput import keyboard, mouse
import threading

class KeyboardTracker:
    def __init__(self):
        self.key_counts = {}
        self.key_combinations = {}
        self.last_keys = []
        self.max_combo_length = 3  # Track up to 3-key combinations
        self.activity_type = "normal"  # Default activity type
        self.recording = False
        self.data = []
        self.session_start_time = None
        self.model_file = "keyboard_model.pkl"
        self.scaler_file = "keyboard_scaler.pkl"
        
        # Features we'll track for our model
        self.features = [
            'cmd_c_count', 'cmd_v_count', 'cmd_x_count', 'cmd_z_count',
            'cmd_a_count', 'ctrl_count', 'alt_tab_count', 'arrows_count',
            'special_keys_ratio', 'avg_key_interval', 'key_variety',
            'repeated_sequences'
        ]
        
    def on_press(self, key):
        if not self.recording:
            return
        
        # Record the key press time for interval calculation
        current_time = time.time()
        
        try:
            # Convert the key to a string representation
            key_str = key.char if hasattr(key, 'char') else str(key)
            
            # Count individual keys
            if key_str in self.key_counts:
                self.key_counts[key_str] += 1
            else:
                self.key_counts[key_str] = 1
            
            # Track key combinations (up to max_combo_length)
            self.last_keys.append(key_str)
            if len(self.last_keys) > self.max_combo_length:
                self.last_keys.pop(0)
            
            # Count the current combination
            if len(self.last_keys) > 1:
                combo = '+'.join(self.last_keys)
                if combo in self.key_combinations:
                    self.key_combinations[combo] += 1
                else:
                    self.key_combinations[combo] = 1
                    
        except AttributeError:
            # Special key
            key_str = str(key)
            if key_str in self.key_counts:
                self.key_counts[key_str] += 1
            else:
                self.key_counts[key_str] = 1
            
            # Add to combinations as well
            self.last_keys.append(key_str)
            if len(self.last_keys) > self.max_combo_length:
                self.last_keys.pop(0)
            
            if len(self.last_keys) > 1:
                combo = '+'.join(self.last_keys)
                if combo in self.key_combinations:
                    self.key_combinations[combo] += 1
                else:
                    self.key_combinations[combo] = 1
    
    def extract_features(self):
        """Extract features from the recorded key data"""
        # Calculate various metrics
        total_keys = sum(self.key_counts.values())
        if total_keys == 0:
            return None  # Not enough data
        
        # Initialize features dictionary
        features_dict = {feature: 0 for feature in self.features}
        
        # Count specific combinations
        cmd_key = "Key.cmd" if "Key.cmd" in self.key_counts else "Key.cmd_l" if "Key.cmd_l" in self.key_counts else "Key.cmd_r"
        ctrl_key = "Key.ctrl" if "Key.ctrl" in self.key_counts else "Key.ctrl_l" if "Key.ctrl_l" in self.key_counts else "Key.ctrl_r"
        alt_key = "Key.alt" if "Key.alt" in self.key_counts else "Key.alt_l" if "Key.alt_l" in self.key_counts else "Key.alt_r"
        
        # Common macOS shortcuts
        for combo in self.key_combinations:
            if f"{cmd_key}+c" in combo:
                features_dict['cmd_c_count'] += self.key_combinations[combo]
            if f"{cmd_key}+v" in combo:
                features_dict['cmd_v_count'] += self.key_combinations[combo]
            if f"{cmd_key}+x" in combo:
                features_dict['cmd_x_count'] += self.key_combinations[combo]
            if f"{cmd_key}+z" in combo:
                features_dict['cmd_z_count'] += self.key_combinations[combo]
            if f"{cmd_key}+a" in combo:
                features_dict['cmd_a_count'] += self.key_combinations[combo]
            if f"{alt_key}+Key.tab" in combo:
                features_dict['alt_tab_count'] += self.key_combinations[combo]
        
        # Count ctrl key presses
        features_dict['ctrl_count'] = self.key_counts.get(ctrl_key, 0)
        
        # Count arrow keys
        arrow_keys = ["Key.up", "Key.down", "Key.left", "Key.right"]
        features_dict['arrows_count'] = sum(self.key_counts.get(k, 0) for k in arrow_keys)
        
        # Calculate special keys ratio
        special_keys = sum(self.key_counts.get(k, 0) for k in self.key_counts if k.startswith("Key."))
        features_dict['special_keys_ratio'] = special_keys / total_keys if total_keys > 0 else 0
        
        # Calculate key variety (number of unique keys divided by total keys pressed)
        features_dict['key_variety'] = len(self.key_counts) / total_keys if total_keys > 0 else 0
        
        # Placeholder for average key interval (would need timestamps)
        features_dict['avg_key_interval'] = 0.5  # Placeholder value
        
        # Count repeated sequences (same key pressed 3+ times in a row)
        features_dict['repeated_sequences'] = sum(1 for combo in self.key_combinations 
                                               if '+'.join([combo.split('+')[0]]*3) == combo)
        
        return features_dict
    
    def start_recording(self, activity_type="normal"):
        """Start recording keyboard activity"""
        self.activity_type = activity_type
        self.recording = True
        self.key_counts = {}
        self.key_combinations = {}
        self.last_keys = []
        self.session_start_time = time.time()
        print(f"Recording started. Activity type: {activity_type}")
    
    def stop_recording(self):
        """Stop recording and save the data"""
        if not self.recording:
            return
        
        self.recording = False
        features = self.extract_features()
        
        if features is not None:
            # Add the label (activity type)
            features['activity_type'] = self.activity_type
            self.data.append(features)
            print(f"Recording stopped. {len(self.key_counts)} keys tracked.")
        else:
            print("Not enough data collected. Recording discarded.")
    
    def save_data(self, filename="keyboard_data.csv"):
        """Save the collected data to a CSV file"""
        if not self.data:
            print("No data to save.")
            return
        
        df = pd.DataFrame(self.data)
        
        # Check if file exists to decide if we need to append or create new
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    
    def train_model(self, data_file="keyboard_data.csv"):
        """Train an SVM model on the collected data"""
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found.")
            return
        
        print("Training model...")
        df = pd.read_csv(data_file)
        
        if len(df) < 2:
            print("Not enough data to train model.")
            return
        
        # Prepare features and target
        X = df[self.features]
        y = df['activity_type']
        
        # Create and train the pipeline with SVM
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True))
        ])
        
        pipeline.fit(X, y)
        
        # Save the model
        with open(self.model_file, 'wb') as f:
            pickle.dump(pipeline, f)
        
        print(f"Model trained and saved to {self.model_file}")

def collect_data():
    """Function to collect keyboard data interactively"""
    tracker = KeyboardTracker()
    
    # Start listening to keyboard
    keyboard_listener = keyboard.Listener(on_press=tracker.on_press)
    keyboard_listener.start()
    
    try:
        while True:
            cmd = input("\nEnter command (train/sus/norm/stop/quit): ").strip().lower()
            
            if cmd == "sus":
                # Record suspicious activity
                tracker.start_recording("suspicious")
                input("Press Enter to stop recording suspicious activity...")
                tracker.stop_recording()
                
            elif cmd == "norm":
                # Record normal activity
                tracker.start_recording("normal")
                input("Press Enter to stop recording normal activity...")
                tracker.stop_recording()
                
            elif cmd == "stop":
                # Stop current recording
                tracker.stop_recording()
                
            elif cmd == "train":
                # Save data and train model
                tracker.save_data()
                tracker.train_model()
                
            elif cmd == "quit":
                # Stop recording, save data, and exit
                tracker.stop_recording()
                tracker.save_data()
                break
                
            else:
                print("Unknown command. Available commands: sus, norm, stop, train, quit")
    
    except KeyboardInterrupt:
        tracker.stop_recording()
        tracker.save_data()
    
    finally:
        # Stop the listener
        keyboard_listener.stop()
        print("Keyboard tracking stopped.")

if __name__ == "__main__":
    print("Keyboard Activity Tracker and Trainer")
    print("-------------------------------------")
    print("Use 'sus' to record suspicious keyboard activity")
    print("Use 'norm' to record normal keyboard activity")
    print("Use 'train' to save data and train the model")
    print("Use 'quit' to exit")
    
    collect_data()