import numpy as np
import time
import subprocess
import re
import pickle
import os
from datetime import datetime

class CSIDataCollector:
    def __init__(self):
        self.posture_labels = ["sitting", "standing", "walking"]
        self.raw_data = {label: [] for label in self.posture_labels}
        self.data_dir = "posture_data"
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            try:
                os.makedirs(self.data_dir)
                print(f"Created data directory: {self.data_dir}")
            except Exception as e:
                print(f"Warning: Could not create directory {self.data_dir}: {e}")
                # Fallback to current directory
                self.data_dir = "."
                print(f"Will save data to current directory instead.")
    
    def extract_wifi_csi(self):
        """
        Extract Channel State Information (CSI) from Wi-Fi signal
        This is a simplified version that uses signal strength as a proxy
        """
        try:
            # Try to get actual signal strength data if possible
            # This works on Linux systems
            try:
                # Handle windows differently from Linux
                if os.name == 'nt':  # Windows
                    try:
                        output = subprocess.check_output("netsh wlan show interfaces", shell=True).decode()
                        signal_match = re.search(r"Signal\s+:\s+(\d+)%", output)
                        if signal_match:
                            # Convert percentage to dBm (rough approximation)
                            signal_percent = int(signal_match.group(1))
                            # Approximate conversion from % to dBm
                            # 100% ≈ -30dBm, 0% ≈ -100dBm
                            base_signal = -100 + (signal_percent * 0.7)
                        else:
                            base_signal = -65  # Default value
                    except:
                        base_signal = -65  # Default value
                else:  # Linux/Mac
                    output = subprocess.check_output("iwconfig 2>/dev/null | grep -i signal", shell=True).decode()
                    signal_level = re.search(r"Signal level[=:](-\d+) dBm", output)
                    if signal_level:
                        base_signal = float(signal_level.group(1))
                    else:
                        base_signal = -65  # Default value
            except:
                base_signal = -65  # Default value if command fails
            
            # For real CSI we would have multiple subcarrier values
            # Here we simulate them by adding variations around the base signal
            variations = np.random.normal(0, 2, 30)
            csi_values = base_signal + variations
            
            return csi_values
            
        except Exception as e:
            print(f"Error extracting CSI data: {e}")
            return np.ones(30) * -75  # Return default value if extraction fails
    
    def collect_training_data(self, posture, duration=10):
        """Collect training data for a specific posture"""
        if posture not in self.posture_labels:
            print(f"Unknown posture: {posture}")
            return
            
        print(f"Please {posture} now. Collecting data for {duration} seconds...")
        print("3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("Start!")
        
        start_time = time.time()
        samples = []
        
        try:
            while time.time() - start_time < duration:
                csi_values = self.extract_wifi_csi()
                samples.append(csi_values)
                
                # Print progress indicator
                elapsed = time.time() - start_time
                print(f"\rProgress: {elapsed:.1f}/{duration} seconds [{len(samples)} samples]", end="")
                
                time.sleep(0.1)  # Collect approximately 10 samples per second
                
            print(f"\nCollected {len(samples)} samples for {posture}")
            self.raw_data[posture].extend(samples)
            
        except KeyboardInterrupt:
            print("\nData collection interrupted!")
    
    def save_data(self):
        """Save the collected data to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Make sure directory exists
        try:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
        except:
            print(f"Warning: Could not create or access directory {self.data_dir}")
            self.data_dir = "."  # Fallback to current directory
        
        filename = os.path.join(self.data_dir, f"csi_data_{timestamp}.pkl")
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.raw_data, f)
                
            print(f"Data saved to {filename}")
            
            # Save a summary text file
            summary_file = os.path.join(self.data_dir, f"csi_data_{timestamp}_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"Data Collection Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 50 + "\n")
                for posture in self.posture_labels:
                    f.write(f"{posture}: {len(self.raw_data[posture])} samples\n")
                    
            return filename
        except Exception as e:
            print(f"Error saving data: {e}")
            
            # Try saving to current directory as fallback
            try:
                fallback_file = f"csi_data_{timestamp}.pkl"
                with open(fallback_file, 'wb') as f:
                    pickle.dump(self.raw_data, f)
                print(f"Data saved to current directory instead: {fallback_file}")
                return fallback_file
            except Exception as e2:
                print(f"Failed to save data even to current directory: {e2}")
                return None

def main():
    collector = CSIDataCollector()
    
    print("=" * 50)
    print("Wi-Fi CSI Posture Data Collection")
    print("=" * 50)
    print("This script will collect Wi-Fi signal data for different postures.")
    print("Make sure you're in a relatively stable environment for best results.")
    print("=" * 50)
    
    for posture in collector.posture_labels:
        while True:
            try:
                duration = int(input(f"\nHow many seconds of '{posture}' data do you want to collect? (recommended: 30) "))
                if duration > 0:
                    break
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
        
        input(f"Press Enter when you're ready to collect data for '{posture}' posture...")
        collector.collect_training_data(posture, duration)
    
    print("\nData collection completed!")
    data_file = collector.save_data()
    if data_file:
        print(f"\nAll data saved to {data_file}")
        print("You can now use this data file with the train_model.py script.")
    else:
        print("\nFailed to save data. Please check permissions and disk space.")

if __name__ == "__main__":
    main()