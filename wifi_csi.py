import numpy as np
import pandas as pd
import time
import os
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import threading
import socket
import json
import pickle

class WiFiPositioningSystem:
    def __init__(self, data_dir='./data', model_dir='./model'):
        """
        Initialize WiFi Positioning System
        
        Args:
            data_dir: Directory to store collected data
            model_dir: Directory to store trained models
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Create directories if they don't exist
        for directory in [data_dir, model_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def scan_wifi_networks(self):
        """
        Scan available WiFi networks and return signal strengths
        
        Returns:
            dict: Dictionary of WiFi networks and their signal strengths (RSSI)
        """
        try:
            # For Linux systems using iwlist
            if os.name == 'posix':
                try:
                    import subprocess
                    output = subprocess.check_output(['iwlist', 'scan']).decode('utf-8')
                    networks = {}
                    current_mac = None
                    
                    for line in output.split('\n'):
                        if 'Address:' in line:
                            current_mac = line.split('Address:')[1].strip()
                            networks[current_mac] = {'ssid': '', 'signal': -100}
                        elif 'ESSID:' in line and current_mac:
                            networks[current_mac]['ssid'] = line.split('ESSID:')[1].strip('"')
                        elif 'Signal level=' in line and current_mac:
                            signal_part = line.split('Signal level=')[1]
                            if 'dBm' in signal_part:
                                networks[current_mac]['signal'] = int(signal_part.split('dBm')[0].strip())
                            else:
                                # Handle percentage format
                                value = int(signal_part.split('/')[0].strip())
                                # Convert to dBm (approximate)
                                networks[current_mac]['signal'] = value - 100
                    
                    return networks
                except Exception as e:
                    print(f"Linux WiFi scanning error: {e}")
            
            # For Windows systems using netsh
            elif os.name == 'nt':
                try:
                    import subprocess
                    output = subprocess.check_output(['netsh', 'wlan', 'show', 'networks', 'mode=Bssid']).decode('utf-8', errors='ignore')
                    networks = {}
                    current_ssid = None
                    current_mac = None
                    
                    for line in output.split('\n'):
                        line = line.strip()
                        if line.startswith('SSID'):
                            current_ssid = line.split(':')[1].strip()
                        elif 'BSSID' in line:
                            current_mac = line.split(':')[1].strip()
                            networks[current_mac] = {'ssid': current_ssid, 'signal': -100}
                        elif 'Signal' in line and current_mac:
                            signal_str = line.split(':')[1].strip().replace('%', '')
                            signal_percent = int(signal_str)
                            # Convert percent to dBm (approximate)
                            signal_dbm = (signal_percent / 2) - 100
                            networks[current_mac]['signal'] = int(signal_dbm)
                    
                    return networks
                except Exception as e:
                    print(f"Windows WiFi scanning error: {e}")
            
            # For macOS systems using airport
            elif os.name == 'darwin':
                try:
                    import subprocess
                    # macOS requires the airport command from a specific location
                    output = subprocess.check_output(['/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport', '-s']).decode('utf-8')
                    lines = output.split('\n')[1:]  # Skip header
                    networks = {}
                    
                    for line in lines:
                        if not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            ssid = parts[0]
                            # MAC address is typically in parts[1]
                            mac = parts[1] if ':' in parts[1] else parts[2]
                            # RSSI is usually the first number
                            for part in parts:
                                if part.lstrip('-').isdigit():
                                    rssi = int(part)
                                    break
                            else:
                                rssi = -100
                            
                            networks[mac] = {'ssid': ssid, 'signal': rssi}
                    
                    return networks
                except Exception as e:
                    print(f"macOS WiFi scanning error: {e}")
            
            # Fallback to simulated data for testing or unsupported platforms
            print("Using simulated WiFi scan data (platform not supported)")
            return self.simulate_wifi_scan()
            
        except Exception as e:
            print(f"Error scanning WiFi networks: {e}")
            print("Using simulated WiFi scan data")
            return self.simulate_wifi_scan()
    
    def simulate_wifi_scan(self):
        """
        Simulate WiFi scan data for testing or when actual scanning isn't available
        
        Returns:
            dict: Dictionary of simulated WiFi networks
        """
        # Generate some dummy access points with realistic MACs and signal strengths
        networks = {}
        
        # Create consistent access points at fixed virtual locations
        base_macs = [
            "00:11:22:33:44:55",
            "AA:BB:CC:DD:EE:FF",
            "01:23:45:67:89:AB",
            "CD:EF:01:23:45:67",
            "89:AB:CD:EF:01:23"
        ]
        
        # Add some randomization to simulate real-world signal fluctuations
        for i, mac in enumerate(base_macs):
            # Base signal level decreases with AP index to simulate different distances
            base_signal = -50 - (i * 8)
            
            # Add small random fluctuation to signal
            signal = base_signal + np.random.randint(-3, 4)
            
            networks[mac] = {
                'ssid': f"WiFi_AP_{i+1}",
                'signal': signal
            }
        
        return networks
    
    def collect_training_data(self, location_label, duration=10, interval=0.5):
        """
        Collect WiFi signal data for training at a specific location
        
        Args:
            location_label: Label or coordinates for the current location
            duration: Duration to collect data in seconds
            interval: Time between scans in seconds
            
        Returns:
            str: Path to the saved data file
        """
        print(f"Collecting training data for location: {location_label}")
        print(f"Please stay at this position for {duration} seconds...")
        
        samples = []
        start_time = time.time()
        end_time = start_time + duration
        
        # Format location_label for filename
        if isinstance(location_label, (list, tuple)):
            # If coordinates are provided, format as x_y_z
            loc_str = "_".join(str(coord) for coord in location_label)
        else:
            # Otherwise use the label as is
            loc_str = str(location_label)
        
        # For progress display
        num_samples = int(duration / interval)
        collected = 0
        
        try:
            while time.time() < end_time:
                networks = self.scan_wifi_networks()
                
                # Create a sample with timestamp, networks data, and location
                sample = {
                    'timestamp': time.time(),
                    'networks': networks,
                    'location': location_label
                }
                
                samples.append(sample)
                collected += 1
                
                # Display progress
                progress = int((collected / num_samples) * 20)
                print(f"\rProgress: [{'#' * progress}{' ' * (20-progress)}] {collected}/{num_samples}", end="")
                
                # Wait for the next interval
                next_scan = time.time() + interval
                while time.time() < next_scan and time.time() < end_time:
                    time.sleep(0.1)
            
            print("\nData collection complete!")
            
            # Save the collected data
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_dir}/wifi_data_{loc_str}_{timestamp}.pkl"
            
            with open(filename, 'wb') as f:
                pickle.dump(samples, f)
            
            print(f"Data saved to {filename}")
            return filename
            
        except KeyboardInterrupt:
            print("\nData collection interrupted!")
            if samples:
                # Save what we have so far
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{self.data_dir}/wifi_data_{loc_str}_{timestamp}_partial.pkl"
                
                with open(filename, 'wb') as f:
                    pickle.dump(samples, f)
                
                print(f"Partial data saved to {filename}")
                return filename
            return None
    
    def process_training_data(self):
        """
        Process all collected training data into a format suitable for model training
        
        Returns:
            tuple: (X, y, mac_to_index) - Feature matrix, target values, and MAC mapping
        """
        print("Processing training data...")
        
        # Get all data files
        data_files = [f for f in os.listdir(self.data_dir) if f.startswith('wifi_data_') and f.endswith('.pkl')]
        
        if not data_files:
            raise ValueError(f"No training data found in {self.data_dir}")
        
        all_access_points = set()
        all_samples = []
        
        # First pass: identify all unique access points
        for data_file in data_files:
            with open(os.path.join(self.data_dir, data_file), 'rb') as f:
                samples = pickle.load(f)
                
                for sample in samples:
                    networks = sample['networks']
                    all_access_points.update(networks.keys())
        
        # Sort APs for consistent ordering
        all_access_points = sorted(list(all_access_points))
        
        # Convert MAC addresses to indices for the feature vector
        mac_to_index = {mac: i for i, mac in enumerate(all_access_points)}
        
        # Second pass: create feature matrix
        for data_file in data_files:
            with open(os.path.join(self.data_dir, data_file), 'rb') as f:
                samples = pickle.load(f)
                
                for sample in samples:
                    # Create a feature vector with rssi values for each AP
                    # Default to -100 dBm (very weak) for APs not detected
                    features = np.full(len(all_access_points), -100.0)
                    
                    for mac, data in sample['networks'].items():
                        if mac in mac_to_index:
                            features[mac_to_index[mac]] = data['signal']
                    
                    all_samples.append({
                        'features': features,
                        'location': sample['location']
                    })
        
        # Convert to numpy arrays
        X = np.array([s['features'] for s in all_samples])
        
        # Handle location format - could be coordinates or labels
        locations = [s['location'] for s in all_samples]
        
        # Check if locations are numerical coordinates or labels
        if isinstance(locations[0], (list, tuple)) and all(isinstance(item, (int, float)) for loc in locations for item in loc):
            # Numerical coordinates - convert to numpy array
            y = np.array(locations)
        else:
            # Labels - keep as is for classification
            y = np.array(locations)
        
        print(f"Processed {len(all_samples)} samples with {len(all_access_points)} access points")
        
        # Save the MAC address mapping for inference
        mapping_file = os.path.join(self.model_dir, 'mac_mapping.pkl')
        with open(mapping_file, 'wb') as f:
            pickle.dump((mac_to_index, all_access_points), f)
        
        return X, y, mac_to_index
    
    def train_positioning_model(self, regression=True):
        """
        Train a positioning model
        
        Args:
            regression: If True, train regression model for coordinates, otherwise classification for labeled locations
            
        Returns:
            tuple: (model, accuracy) - Trained model and its accuracy/error
        """
        X, y, mac_mapping = self.process_training_data()
        
        # Check if we're doing regression or classification
        if regression and isinstance(y[0], (list, tuple, np.ndarray)):
            # We're predicting coordinates (regression)
            print("Training regression model for coordinate prediction...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train separate models for each coordinate dimension
            models = []
            errors = []
            
            for dim in range(y.shape[1]):
                model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=15,
                    min_samples_split=2,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Train on this dimension
                model.fit(X_train, y_train[:, dim])
                
                # Evaluate
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test[:, dim], y_pred)
                rmse = np.sqrt(mse)
                
                print(f"Dimension {dim} RMSE: {rmse:.4f}")
                
                models.append(model)
                errors.append(rmse)
            
            # Save models
            for dim, model in enumerate(models):
                model_file = os.path.join(self.model_dir, f'position_model_dim{dim}.pkl')
                joblib.dump(model, model_file)
            
            # Save model type
            with open(os.path.join(self.model_dir, 'model_type.txt'), 'w') as f:
                f.write('regression')
            
            # Calculate average error
            avg_error = np.mean(errors)
            print(f"Average RMSE across all dimensions: {avg_error:.4f}")
            
            return models, avg_error
            
        else:
            # We're predicting labeled locations (classification)
            print("Training classification model for labeled locations...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train a random forest classifier
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Classification accuracy: {accuracy:.4f}")
            
            # Save model
            model_file = os.path.join(self.model_dir, 'position_model_class.pkl')
            joblib.dump(model, model_file)
            
            # Save model type
            with open(os.path.join(self.model_dir, 'model_type.txt'), 'w') as f:
                f.write('classification')
            
            return model, accuracy
    
    def predict_position(self, wifi_data=None):
        """
        Predict position using the trained model
        
        Args:
            wifi_data: WiFi scan data, if None will perform a new scan
            
        Returns:
            Position prediction (coordinates or label)
        """
        # Get WiFi data if not provided
        if wifi_data is None:
            wifi_data = self.scan_wifi_networks()
        
        # Load MAC mapping
        mapping_file = os.path.join(self.model_dir, 'mac_mapping.pkl')
        if not os.path.exists(mapping_file):
            raise ValueError("No MAC mapping found. Please train the model first.")
            
        with open(mapping_file, 'rb') as f:
            mac_to_index, all_access_points = pickle.load(f)
        
        # Create feature vector
        features = np.full(len(all_access_points), -100.0)
        for mac, data in wifi_data.items():
            if mac in mac_to_index:
                features[mac_to_index[mac]] = data['signal']
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Determine model type
        model_type_file = os.path.join(self.model_dir, 'model_type.txt')
        if not os.path.exists(model_type_file):
            raise ValueError("Model type information not found. Please train the model first.")
            
        with open(model_type_file, 'r') as f:
            model_type = f.read().strip()
        
        # Make prediction based on model type
        if model_type == 'regression':
            # Load regression models for each dimension
            models = []
            dim = 0
            
            while True:
                model_file = os.path.join(self.model_dir, f'position_model_dim{dim}.pkl')
                if not os.path.exists(model_file):
                    break
                    
                models.append(joblib.load(model_file))
                dim += 1
            
            if not models:
                raise ValueError("No regression models found. Please train the model first.")
            
            # Predict each dimension
            position = np.array([model.predict(features)[0] for model in models])
            
            return position
            
        else:  # classification
            # Load classification model
            model_file = os.path.join(self.model_dir, 'position_model_class.pkl')
            if not os.path.exists(model_file):
                raise ValueError("No classification model found. Please train the model first.")
                
            model = joblib.load(model_file)
            
            # Predict location
            location = model.predict(features)[0]
            
            return location
    
    def visualize_position(self, position, history=None, is_2d=True):
        """
        Visualize the current position and optionally position history
        
        Args:
            position: Current position (coordinates)
            history: List of previous positions
            is_2d: If True, visualize in 2D (x,y), otherwise 3D (x,y,z)
        """
        if history is None:
            history = []
            
        # Convert to numpy arrays
        position = np.array(position)
        if history:
            history = np.array(history)
        
        # Create the plot
        if is_2d:
            plt.figure(figsize=(10, 8))
            
            # Plot history if available
            if len(history) > 0:
                plt.plot(history[:, 0], history[:, 1], 'b-', alpha=0.5, label='Path')
                plt.scatter(history[:, 0], history[:, 1], c='blue', alpha=0.5, s=30)
            
            # Plot current position
            plt.scatter(position[0], position[1], c='red', s=100, marker='o', label='Current Position')
            
            plt.title('WiFi Positioning - 2D Visualization')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.grid(True)
            plt.legend()
            
        else:
            # 3D visualization
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot history if available
            if len(history) > 0:
                ax.plot(history[:, 0], history[:, 1], history[:, 2], 'b-', alpha=0.5, label='Path')
                ax.scatter(history[:, 0], history[:, 1], history[:, 2], c='blue', alpha=0.5, s=30)
            
            # Plot current position
            ax.scatter(position[0], position[1], position[2], c='red', s=100, marker='o', label='Current Position')
            
            ax.set_title('WiFi Positioning - 3D Visualization')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_zlabel('Z coordinate')
            ax.legend()
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{self.data_dir}/position_viz_{timestamp}.png")
        
        # Show the plot
        plt.show()

    def smooth_positions(self, positions, window_size=5):
        """
        Apply median filter to smooth position estimates
        
        Args:
            positions: List of position estimates
            window_size: Size of the median filter window
            
        Returns:
            Smoothed positions
        """
        if len(positions) < window_size:
            return positions  # Not enough data to smooth
            
        positions = np.array(positions)
        
        # Apply median filter to each dimension
        smoothed = np.zeros_like(positions)
        for dim in range(positions.shape[1]):
            smoothed[:, dim] = medfilt(positions[:, dim], window_size)
            
        return smoothed
    
    def start_positioning_server(self, host='0.0.0.0', port=8000):
        """
        Start a server that provides positioning services over the network
        
        Args:
            host: Host to bind the server to
            port: Port to listen on
        """
        # Create a socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind((host, port))
            server_socket.listen(5)
            print(f"Positioning server started on {host}:{port}")
            
            while True:
                client_socket, address = server_socket.accept()
                print(f"Connection from {address}")
                
                # Start a new thread for each client
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,)
                )
                client_thread.daemon = True
                client_thread.start()
                
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            server_socket.close()
    
    def _handle_client(self, client_socket):
        """
        Handle client connection for the positioning server
        
        Args:
            client_socket: Socket for the client connection
        """
        try:
            # Receive data from client
            data = b''
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                
                # Check if we have a complete message
                if b'\n' in data:
                    break
            
            if not data:
                return
                
            # Parse the request
            request = json.loads(data.decode('utf-8'))
            
            response = {"status": "error", "message": "Invalid request"}
            
            # Handle different request types
            if 'type' in request:
                if request['type'] == 'scan':
                    # Scan WiFi networks
                    networks = self.scan_wifi_networks()
                    response = {
                        "status": "success",
                        "data": networks
                    }
                    
                elif request['type'] == 'position':
                    # Get position
                    
                    # Use client's WiFi data if provided
                    wifi_data = request.get('wifi_data', None)
                    
                    # Predict position
                    try:
                        position = self.predict_position(wifi_data)
                        response = {
                            "status": "success",
                            "position": position.tolist() if isinstance(position, np.ndarray) else position
                        }
                    except Exception as e:
                        response = {
                            "status": "error",
                            "message": f"Error predicting position: {str(e)}"
                        }
            
            # Send response
            response_data = json.dumps(response).encode('utf-8') + b'\n'
            client_socket.sendall(response_data)
            
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()
    
    def real_time_positioning(self, duration=60, interval=1.0, smoothing_window=3, visualize=True):
        """
        Perform real-time positioning for a specified duration
        
        Args:
            duration: Duration to track position in seconds
            interval: Time between position updates in seconds
            smoothing_window: Window size for position smoothing
            visualize: Whether to visualize the results
        """
        print(f"Starting real-time positioning for {duration} seconds...")
        
        position_history = []
        start_time = time.time()
        end_time = start_time + duration
        
        try:
            while time.time() < end_time:
                # Get current position
                current_position = self.predict_position()
                
                # If classification, skip visualization
                if not isinstance(current_position, np.ndarray):
                    print(f"Current location: {current_position}")
                    time.sleep(interval)
                    continue
                
                # Add to history
                position_history.append(current_position)
                
                # Apply smoothing if enough data points
                if len(position_history) >= smoothing_window:
                    smooth_history = self.smooth_positions(position_history, smoothing_window)
                    current_smooth = smooth_history[-1]
                    
                    # Show current position
                    dimensions = len(current_position)
                    if dimensions == 2:
                        print(f"Current position: X={current_smooth[0]:.2f}, Y={current_smooth[1]:.2f}")
                    else:
                        print(f"Current position: X={current_smooth[0]:.2f}, Y={current_smooth[1]:.2f}, Z={current_smooth[2]:.2f}")
                    
                    # Optional visualization
                    if visualize and len(position_history) % 10 == 0:  # Update every 10 points
                        is_2d = (dimensions == 2)
                        self.visualize_position(current_smooth, smooth_history, is_2d=is_2d)
                else:
                    # Show current position without smoothing
                    dimensions = len(current_position)
                    if dimensions == 2:
                        print(f"Current position: X={current_position[0]:.2f}, Y={current_position[1]:.2f}")
                    else:
                        print(f"Current position: X={current_position[0]:.2f}, Y={current_position[1]:.2f}, Z={current_position[2]:.2f}")
                
                # Wait for next interval
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nPositioning stopped by user.")
        
        # Final visualization
        if visualize and position_history and isinstance(position_history[0], np.ndarray):
            smooth_history = self.smooth_positions(position_history, smoothing_window)
            final_position = smooth_history[-1]
            
            is_2d = (len(final_position) == 2)
            self.visualize_position(final_position, smooth_history, is_2d=is_2d)
            
        return position_history

def main():
    """
    Main function to run WiFi positioning system from command line
    """
    parser = argparse.ArgumentParser(description='WiFi Positioning System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect training data')
    collect_parser.add_argument('--location', required=True, help='Location label or coordinates (x,y,z)')
    collect_parser.add_argument('--duration', type=int, default=30, help='Duration to collect data (seconds)')
    collect_parser.add_argument('--interval', type=float, default=0.5, help='Time between scans (seconds)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train positioning model')
    train_parser.add_argument('--type', choices=['regression', 'classification'], default='regression',
                             help='Type of model to train')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict position')
    predict_parser.add_argument('--visualize', action='store_true', help='Visualize the prediction')
    
    # Real-time command
    realtime_parser = subparsers.add_parser('realtime', help='Real-time positioning')
    realtime_parser.add_argument('--duration', type=int, default=60, help='Duration (seconds)')
    realtime_parser.add_argument('--interval', type=float, default=1.0, help='Time between updates (seconds)')
    realtime_parser.add_argument('--smoothing', type=int, default=3, help='Smoothing window size')
    realtime_parser.add_argument('--visualize', action='store_true', help='Visualize the tracking')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start positioning server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the system
    wps = WiFiPositioningSystem()
    
    # Execute the requested command
    if args.command == 'collect':
        # Parse location
        location = args.location
        try:
            # Check if location is coordinates (x,y,z)
            if ',' in location:
                location = [float(x) for x in location.split(',')]
        except ValueError:
            print("Using location as label")
        
        wps.collect_training_data(location, args.duration, args.interval)
        
    elif args.command == 'train':
        regression = (args.type == 'regression')
        wps.train_positioning_model(regression=regression)
        
    elif args.command == 'predict':
        position = wps.predict_position()
        
        if isinstance(position, np.ndarray):
            if len(position) == 2:
                print(f"Predicted position: X={position[0]:.2f}, Y={position[1]:.2f}")
            else:
                print(f"Predicted position: X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}")
                
            if args.visualize:
                wps.visualize_position(position, is_2d=(len(position) == 2))
        else:
            print(f"Predicted location: {position}")
        
    elif args.command == 'realtime':
        wps.real_time_positioning(
            duration=args.duration,
            interval=args.interval,
            smoothing_window=args.smoothing,
            visualize=args.visualize
        )
        
    elif args.command == 'server':
        wps.start_positioning_server(host=args.host, port=args.port)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()