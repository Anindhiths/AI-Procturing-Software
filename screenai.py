import os
import time
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageGrab
import shutil
from datetime import datetime

class ScreenMonitor:
    def __init__(self, templates_folder, screenshot_folder, interval=5):
        """
        Initialize the Screen Monitor.
        
        Args:
            templates_folder (str): Path to folder containing question templates
            screenshot_folder (str): Path to store temporary screenshots
            interval (int): Time interval between screenshots in seconds
        """
        self.templates_folder = templates_folder
        self.screenshot_folder = screenshot_folder
        self.interval = interval
        
        # Create folders if they don't exist
        os.makedirs(templates_folder, exist_ok=True)
        os.makedirs(screenshot_folder, exist_ok=True)
        
        # Set Tesseract command if not in PATH
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
        
        # Load template images
        self.templates = self._load_templates()
        
        # Initialize log file
        self.log_file = os.path.join(os.path.dirname(screenshot_folder), "screen_monitor_log.txt")

    def _load_templates(self):
        """Load all template images from the templates folder."""
        templates = []
        for filename in os.listdir(self.templates_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                template_path = os.path.join(self.templates_folder, filename)
                template_img = cv2.imread(template_path)
                template_text = self._extract_text(template_path)
                templates.append({
                    'name': filename,
                    'image': template_img,
                    'text': template_text
                })
        return templates

    def _extract_text(self, image_path):
        """Extract text from an image using Tesseract OCR."""
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            self._log(f"Error extracting text from {image_path}: {str(e)}")
            return ""

    def _take_screenshot(self):
        """Take a screenshot and save it to the screenshot folder."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(self.screenshot_folder, f"screenshot_{timestamp}.png")
        
        # Take screenshot
        screenshot = ImageGrab.grab()
        screenshot.save(screenshot_path)
        
        return screenshot_path

    def _compare_with_templates(self, screenshot_path):
        """Compare the screenshot with template images."""
        screenshot_text = self._extract_text(screenshot_path)
        
        matches = []
        for template in self.templates:
            # Simple text matching - can be improved with more sophisticated algorithms
            similarity = self._text_similarity(screenshot_text, template['text'])
            if similarity > 0.6:  # Threshold for similarity
                matches.append({
                    'template_name': template['name'],
                    'similarity': similarity
                })
                
        return matches

    def _text_similarity(self, text1, text2):
        """Calculate similarity between two text strings."""
        # This is a simple implementation - can be improved with NLP techniques
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def _log(self, message):
        """Write a message to the log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

    def start_monitoring(self, duration=None):
        """
        Start monitoring the screen.
        
        Args:
            duration (int, optional): Duration to run in seconds. If None, runs indefinitely.
        """
        self._log("Starting screen monitoring...")
        
        start_time = time.time()
        screenshot_count = 0
        
        try:
            while True:
                # Check if duration is specified and has elapsed
                if duration and (time.time() - start_time > duration):
                    self._log(f"Monitoring completed after {duration} seconds")
                    break
                
                # Take screenshot
                screenshot_path = self._take_screenshot()
                screenshot_count += 1
                self._log(f"Screenshot {screenshot_count} taken: {screenshot_path}")
                
                # Compare with templates
                matches = self._compare_with_templates(screenshot_path)
                
                # Log results
                if matches:
                    for match in matches:
                        self._log(f"Normal (similarity: {match['similarity']:.2f})")
                else:
                    self._log("Suspicious")
                
                # Delete screenshot after analysis
                os.remove(screenshot_path)
                self._log(f"Screenshot deleted: {screenshot_path}")
                
                # Wait for the next interval
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            self._log("Monitoring stopped by user")
        except Exception as e:
            self._log(f"Error during monitoring: {str(e)}")
        finally:
            # Clean up any remaining screenshots
            for filename in os.listdir(self.screenshot_folder):
                file_path = os.path.join(self.screenshot_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    self._log(f"Error deleting file {file_path}: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Define folder paths
    templates_folder = os.path.expanduser("~/Development/projects/onlineexams/test_templates")
    screenshot_folder = os.path.expanduser("~/Development/projects/onlineexams/temp_screenshots")
    
    # Initialize and start monitoring
    monitor = ScreenMonitor(
        templates_folder=templates_folder,
        screenshot_folder=screenshot_folder,
        interval=2  # Take screenshot every 5 seconds
    )
    
    # Start monitoring
    monitor.start_monitoring()