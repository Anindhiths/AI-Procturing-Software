import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import time
import sys

class PostureModelTrainer:
    def __init__(self):
        self.posture_labels = ["sitting", "standing", "walking"]
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.data_dir = "posture_data"
        self.model_dir = "models"
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.model_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def load_real_data(self, filename=None):
        """Load real data collected with collect_data.py"""
        if filename is None:
            # Find the most recent data file
            data_files = [f for f in os.listdir(self.data_dir) if f.startswith("csi_data_") and f.endswith(".pkl")]
            if not data_files:
                print("No data files found in the data directory.")
                return None
                
            # Sort by modification time
            data_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.data_dir, x)), reverse=True)
            filename = os.path.join(self.data_dir, data_files[0])
        
        try:
            with open(filename, 'rb') as f:
                raw_data = pickle.load(f)
            
            print(f"Loaded data from {filename}")
            return raw_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def generate_synthetic_data(self, base_data=None, samples_per_posture=1000):
        """Generate synthetic CSI data for different postures"""
        synthetic_data = {label: [] for label in self.posture_labels}
        
        if base_data is None:
            # No real data available, generate completely synthetic data
            print("Generating purely synthetic data...")
            
            # Sitting: relatively stable signal with small variations
            for _ in range(samples_per_posture):
                base = -65 + np.random.normal(0, 3)
                variation = np.random.normal(0, 2, 30)
                csi_values = base + variation
                synthetic_data["sitting"].append(csi_values)
            
            # Standing: different baseline with moderate variations
            for _ in range(samples_per_posture):
                base = -60 + np.random.normal(0, 3)
                variation = np.random.normal(0, 3, 30)
                csi_values = base + variation
                synthetic_data["standing"].append(csi_values)
            
            # Walking: more dynamic signal with larger variations and periodic component
            for _ in range(samples_per_posture):
                base = -68 + np.random.normal(0, 3)
                variation = np.random.normal(0, 6, 30)
                # Add periodic component to simulate walking motion
                t = np.random.random() * 2 * np.pi  # Random phase
                walking_pattern = 3 * np.sin(np.linspace(t, t + 2*np.pi, 30))
                csi_values = base + variation + walking_pattern
                synthetic_data["walking"].append(csi_values)
        
        else:
            # We have real data, augment it to create more synthetic samples
            print("Augmenting real data with synthetic samples...")
            
            for posture in self.posture_labels:
                if not base_data[posture]:
                    print(f"No real data for {posture}, using purely synthetic data")
                    # Use purely synthetic data for this posture
                    if posture == "sitting":
                        for _ in range(samples_per_posture):
                            base = -65 + np.random.normal(0, 3)
                            variation = np.random.normal(0, 2, 30)
                            csi_values = base + variation
                            synthetic_data[posture].append(csi_values)
                    elif posture == "standing":
                        for _ in range(samples_per_posture):
                            base = -60 + np.random.normal(0, 3)
                            variation = np.random.normal(0, 3, 30)
                            csi_values = base + variation
                            synthetic_data[posture].append(csi_values)
                    elif posture == "walking":
                        for _ in range(samples_per_posture):
                            base = -68 + np.random.normal(0, 3)
                            variation = np.random.normal(0, 6, 30)
                            t = np.random.random() * 2 * np.pi
                            walking_pattern = 3 * np.sin(np.linspace(t, t + 2*np.pi, 30))
                            csi_values = base + variation + walking_pattern
                            synthetic_data[posture].append(csi_values)
                else:
                    # Use real data as a base for generating synthetic samples
                    real_samples = base_data[posture]
                    synthetic_data[posture].extend(real_samples)  # Include original samples
                    
                    # Calculate how many more samples we need
                    additional_needed = max(0, samples_per_posture - len(real_samples))
                    
                    for _ in range(additional_needed):
                        # Randomly select a real sample as base
                        base_sample = real_samples[np.random.randint(0, len(real_samples))]
                        
                        # Add random noise and small variations
                        noise_level = 2 if posture == "sitting" else 3 if posture == "standing" else 5
                        variation = np.random.normal(0, noise_level, len(base_sample))
                        
                        # For walking, add some periodic variations
                        if posture == "walking":
                            t = np.random.random() * 2 * np.pi
                            walking_pattern = 2 * np.sin(np.linspace(t, t + 2*np.pi, len(base_sample)))
                            variation += walking_pattern
                        
                        synthetic_sample = base_sample + variation
                        synthetic_data[posture].append(synthetic_sample)
        
        print("Synthetic data generation complete.")
        for posture in self.posture_labels:
            print(f"  {posture}: {len(synthetic_data[posture])} samples")
            
        return synthetic_data
    
    def preprocess_csi(self, csi_values):
        """Extract features from CSI values"""
        features = []
        
        # Basic statistical features
        features.append(np.mean(csi_values))
        features.append(np.std(csi_values))
        features.append(np.min(csi_values))
        features.append(np.max(csi_values))
        
        # Range and variance
        features.append(np.max(csi_values) - np.min(csi_values))
        features.append(np.var(csi_values))
        
        # Frequency domain features - for detecting repetitive motions like walking
        if len(csi_values) > 5:
            # Compute FFT
            fft_values = np.abs(np.fft.fft(csi_values))
            features.append(np.mean(fft_values))
            features.append(np.std(fft_values))
            features.append(np.max(fft_values))
            
            # Dominant frequency
            if len(fft_values) > 1:
                features.append(np.argmax(fft_values[1:]) + 1)
            else:
                features.append(0)
                
        # Shape features
        if len(csi_values) > 2:
            # Number of peaks as a feature
            peaks, _ = np.array(csi_values), np.array([1] * len(csi_values))
            features.append(len(peaks))
            
            # First and second derivatives (for changes)
            first_derivative = np.diff(csi_values)
            if len(first_derivative) > 0:
                features.append(np.mean(np.abs(first_derivative)))
                features.append(np.max(np.abs(first_derivative)))
                
                if len(first_derivative) > 1:
                    second_derivative = np.diff(first_derivative)
                    features.append(np.mean(np.abs(second_derivative)))
                    features.append(np.max(np.abs(second_derivative)))
        
        return np.array(features)
    
    def prepare_dataset(self, data):
        """Prepare dataset for training by extracting features"""
        X = []
        y = []
        
        total_samples = sum(len(data[posture]) for posture in self.posture_labels)
        processed = 0
        
        for posture in self.posture_labels:
            for csi_values in data[posture]:
                features = self.preprocess_csi(csi_values)
                X.append(features)
                y.append(posture)
                
                # Update progress
                processed += 1
                if processed % 100 == 0:
                    progress = processed / total_samples * 100
                    print(f"\rPreprocessing data: {progress:.1f}% ({processed}/{total_samples})", end="")
        
        print("\nPreprocessing complete!")
        return np.array(X), np.array(y)
    
    def train_model(self, X, y):
        """Train the classifier using the dataset"""
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        print("Training model...")
        start_time = time.time()
        self.classifier.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate the model
        y_pred = self.classifier.predict(X_test)
        
        print("\nModel Evaluation:")
        print("-" * 40)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.posture_labels)
        self.plot_confusion_matrix(cm)
        
        return self.classifier
    
    def plot_confusion_matrix(self, cm):
        """Plot the confusion matrix"""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(self.posture_labels))
        plt.xticks(tick_marks, self.posture_labels, rotation=45)
        plt.yticks(tick_marks, self.posture_labels)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save the confusion matrix plot
        cm_path = os.path.join(self.model_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        
        # Try to display the plot if running in an interactive environment
        try:
            plt.show()
        except:
            pass
    
    def save_model(self, model):
        """Save the trained model to disk"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(self.model_dir, f"posture_model_{timestamp}.joblib")
        
        joblib.dump(model, model_filename)
        
        # Also save latest model reference for easy loading
        latest_model = os.path.join(self.model_dir, "latest_model.joblib")
        joblib.dump(model, latest_model)
        
        # Save feature extractor function
        with open(os.path.join(self.model_dir, "preprocess_func.pkl"), 'wb') as f:
            pickle.dump(self.preprocess_csi, f)
        
        print(f"Model saved to {model_filename}")
        print(f"Latest model reference saved to {latest_model}")
        
        return model_filename

def main():
    trainer = PostureModelTrainer()
    
    print("=" * 50)
    print("Wi-Fi CSI Posture Model Trainer")
    print("=" * 50)
    
    # Check if we should use a specific data file
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        print(f"Using data file: {data_file}")
        base_data = trainer.load_real_data(data_file)
    else:
        print("Looking for most recent data file...")
        base_data = trainer.load_real_data()
    
    # Generate synthetic data
    samples_per_posture = 1000
    if base_data is not None:
        print("\nUsing real data as basis for synthetic data generation")
        print("You can generate a completely synthetic dataset by typing 'synthetic'")
        choice = input("Press Enter to continue with real data augmentation, or type 'synthetic': ")
        
        if choice.lower() == 'synthetic':
            print("Using completely synthetic data instead of real data")
            base_data = None
    else:
        print("No real data found. Using completely synthetic data.")
    
    # Let user specify number of synthetic samples
    try:
        user_samples = input("\nHow many samples per posture to generate? (default: 1000): ")
        if user_samples.strip():
            samples_per_posture = int(user_samples)
    except ValueError:
        print("Invalid input, using default 1000 samples per posture")
    
    # Generate the dataset
    print(f"\nGenerating {samples_per_posture} samples per posture...")
    synthetic_data = trainer.generate_synthetic_data(base_data, samples_per_posture)
    
    # Prepare dataset
    print("\nExtracting features from data...")
    X, y = trainer.prepare_dataset(synthetic_data)
    
    # Train model
    print("\nTraining posture detection model...")
    model = trainer.train_model(X, y)
    
    # Save model
    model_path = trainer.save_model(model)
    
    print("\nTraining completed successfully!")
    print(f"The model is ready to use with detect_posture.py")
    print(f"Run: python detect_posture.py")

if __name__ == "__main__":
    main()