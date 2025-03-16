import os
import time
import pandas as pd
import numpy as np
import pickle
from pynput import keyboard
import threading

class KeyboardMonitor:
    def __init__(self):
        self.key_counts = {}
        self.key_combinations = {}
        self.last_keys = []
        self.max_combo_length = 3
        self.monitoring = False
        self.session_start_time = None
        self.model_file = "keyboard_model.pkl"
        self.check_interval = 5  # Check activity every 5 seconds
        
        # Features we track (must match the training program)
        self.features = [
            'cmd_c_count', 'cmd_v_count', 'cmd_x_count', 'cmd_z_count',
            'cmd_a_count', 'ctrl_count', 'alt_tab_count', 'arrows_count',
            'special_keys_ratio', 'avg_key_interval', 'key_variety',
            'repeated_sequences'
        ]
        
        # Load the model
        if os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                self.pipeline = pickle.load(f)
            print(f"Model loaded from {self.model_file}")
        else:
            self.pipeline = None
            print(f"Warning: Model file {self.model_file} not found. Run training first.")
        
    def on_press(self, key):
        if not self.monitoring:
            return
        
        current_time = time.time()
        
        try:
            # Convert the key to a string representation
            key_str = key.char if hasattr(key, 'char') else str(key)
            
            # Count individual keys
            if key_str in self.key_counts:
                self.key_counts[key_str] += 1
            else:
                self.key_counts[key_str] = 1
            
            # Track key combinations
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
        """Extract features from the recorded key data - same as in the trainer"""
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
    
    def classify_activity(self):
        """Classify the current keyboard activity using the trained model"""
        if self.pipeline is None:
            print("No model loaded. Cannot classify activity.")
            return None
        
        features = self.extract_features()
        if features is None:
            return None
        
        # Convert features to DataFrame (required format for the pipeline)
        features_df = pd.DataFrame([features])
        
        # Use only the features the model was trained on
        X = features_df[self.features]
        
        # Get the prediction and probability
        prediction = self.pipeline.predict(X)[0]
        probabilities = self.pipeline.predict_proba(X)[0]
        
        # Get the index of the predicted class
        classes = self.pipeline.classes_
        pred_idx = list(classes).index(prediction)
        confidence = probabilities[pred_idx]
        
        return {
            'activity_type': prediction,
            'confidence': confidence
        }
    
    def reset_data(self):
        """Reset the recorded key data"""
        self.key_counts = {}
        self.key_combinations = {}
        self.last_keys = []
        self.session_start_time = time.time()
    
    def start_monitoring(self):
        """Start monitoring keyboard activity"""
        self.monitoring = True
        self.reset_data()
        
        # Start periodic checking in a separate thread
        self.check_thread = threading.Thread(target=self._periodic_check)
        self.check_thread.daemon = True
        self.check_thread.start()
        
        print("Keyboard monitoring started. Press Ctrl+C to stop.")
    
    def stop_monitoring(self):
        """Stop monitoring keyboard activity"""
        self.monitoring = False
        print("Keyboard monitoring stopped.")
    
    def _periodic_check(self):
        """Periodically check and classify the activity"""
        while self.monitoring:
            time.sleep(self.check_interval)
            
            # Only classify if we have enough data
            if sum(self.key_counts.values()) > 5:
                classification = self.classify_activity()
                
                if classification:
                    activity_type = classification['activity_type']
                    confidence = classification['confidence'] * 100
                    print(f"\nActivity classified as: {activity_type} (confidence: {confidence:.1f}%)")
                    
                    if activity_type == "suspicious" and confidence > 70:
                        print("⚠️ ALERT: Suspicious keyboard activity detected! ⚠️")
                    
                    # Reset data for the next interval
                    self.reset_data()

def monitor_activity():
    """Function to monitor keyboard activity in real-time"""
    monitor = KeyboardMonitor()
    
    # Start listening to keyboard
    keyboard_listener = keyboard.Listener(on_press=monitor.on_press)
    keyboard_listener.start()
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Keep the main thread running
        while True:
            cmd = input("\nEnter command (stop/start/quit): ").strip().lower()
            
            if cmd == "stop":
                monitor.stop_monitoring()
                
            elif cmd == "start":
                monitor.start_monitoring()
                
            elif cmd == "quit":
                monitor.stop_monitoring()
                break
                
            else:
                print("Unknown command. Available commands: stop, start, quit")
    
    except KeyboardInterrupt:
        monitor.stop_monitoring()
    
    finally:
        # Stop the listener
        keyboard_listener.stop()
        print("Keyboard monitoring terminated.")

if __name__ == "__main__":
    print("Keyboard Activity Monitor")
    print("------------------------")
    print("This program monitors keyboard activity and classifies it based on the trained model.")
    print("Make sure you've trained a model first using the keyboard_tracker.py program.")
    
    monitor_activity()