import os
import time
import threading
import queue
import logging
from datetime import datetime
import colorama
from colorama import Fore, Style

# Import the individual detection systems
# We're using these modules from your files
from screenai import ScreenMonitor
from keyboarddetect import KeyboardMonitor
from mousedetect import MouseDetector

# Initialize colorama for colored console output
colorama.init()

# Set up logging with color-coded output
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, Fore.WHITE)
        if 'SCREEN' in record.getMessage():
            prefix = f"{Fore.MAGENTA}[SCREEN]{Style.RESET_ALL} "
        elif 'KEYBOARD' in record.getMessage():
            prefix = f"{Fore.BLUE}[KEYBOARD]{Style.RESET_ALL} "
        elif 'MOUSE' in record.getMessage():
            prefix = f"{Fore.CYAN}[MOUSE]{Style.RESET_ALL} "
        else:
            prefix = ""
        
        record.msg = f"{prefix}{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

# Configure logging
logging_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter(logging_format))
file_handler = logging.FileHandler("master_ai.log")
file_handler.setFormatter(logging.Formatter(logging_format))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

# Create separate loggers for each AI component
screen_logger = logging.getLogger("ScreenAI")
keyboard_logger = logging.getLogger("KeyboardAI")
mouse_logger = logging.getLogger("MouseAI")
master_logger = logging.getLogger("MasterAI")

class MasterAI:
    def __init__(self):
        # Message queue instead of priority queue
        self.message_queue = queue.Queue()
        
        # Initialize detection components
        self.init_components()
        
        # Alert thresholds
        self.alert_thresholds = {
            "screen": 0.6,    # 60% confidence for screen alerts
            "keyboard": 0.5,  # 50% confidence for keyboard alerts
            "mouse": 0.4      # 40% confidence for mouse alerts
        }
        
        # Alert cooldown periods (in seconds)
        self.cooldown_periods = {
            "screen": 12,     # 12 seconds between screen alerts
            "keyboard": 6,    # 6 seconds between keyboard alerts
            "mouse": 3        # 3 seconds between mouse alerts
        }
        
        # Last alert times
        self.last_alert_times = {
            "screen": 0,
            "keyboard": 0,
            "mouse": 0
        }
        
        # Overall suspicion score (0-100)
        self.suspicion_score = 0
        self.suspicion_threshold = 60  # Alert when overall score exceeds this
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Running flag
        self.running = False
        
        master_logger.info("MasterAI initialized with equal visibility for all AI components")

    def init_components(self):
        """Initialize all detection components"""
        try:
            # Paths for screen monitor
            templates_folder = os.path.expanduser("~/Development/projects/onlineexams/test_templates")
            screenshot_folder = os.path.expanduser("~/Development/projects/onlineexams/temp_screenshots")
            
            # Create the folders if they don't exist
            os.makedirs(templates_folder, exist_ok=True)
            os.makedirs(screenshot_folder, exist_ok=True)
            
            master_logger.info("Initializing detection components...")
            
            # Initialize components
            self.screen_monitor = ScreenMonitor(
                templates_folder=templates_folder,
                screenshot_folder=screenshot_folder,
                interval=3  # Check screen every 3 seconds
            )
            screen_logger.info("Screen monitor initialized with templates folder: %s", templates_folder)
            
            self.keyboard_monitor = KeyboardMonitor()
            keyboard_logger.info("Keyboard monitor initialized")
            
            self.mouse_detector = MouseDetector()
            mouse_logger.info("Mouse detector initialized")
            
            master_logger.info("All detection components initialized successfully")
        except Exception as e:
            master_logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def screen_monitoring_thread(self):
        """Thread for screen monitoring"""
        screen_logger.info("SCREEN: Monitoring thread started")
        
        def process_screen_result(matches):
            """Custom callback to process screen monitoring results"""
            suspicious = len(matches) == 0
            if suspicious:
                confidence = 0.9
                screen_logger.warning("SCREEN: Suspicious activity detected - No matches found (confidence: %.2f)", confidence)
                self.add_message("screen", "Suspicious screen activity detected", confidence)
            else:
                # Calculate average similarity
                avg_similarity = sum(match['similarity'] for match in matches) / len(matches)
                if avg_similarity < self.alert_thresholds["screen"]:
                    screen_logger.warning("SCREEN: Unusual screen content detected (similarity: %.2f)", avg_similarity)
                    self.add_message("screen", f"Unusual screen content (similarity: {avg_similarity:.2f})", avg_similarity)
                else:
                    screen_logger.debug("SCREEN: Normal screen activity (similarity: %.2f)", avg_similarity)
        
        # Override the _compare_with_templates method to call our callback
        original_compare = self.screen_monitor._compare_with_templates
        
        def compare_wrapper(screenshot_path):
            matches = original_compare(screenshot_path)
            process_screen_result(matches)
            return matches
        
        self.screen_monitor._compare_with_templates = compare_wrapper
        
        # Start the screen monitoring
        self.screen_monitor.start_monitoring()
        
        # Keep the thread running
        while self.running:
            time.sleep(1)
        
        # Clean up
        self.screen_monitor.stop_monitoring()
    
    def keyboard_monitoring_thread(self):
        """Thread for keyboard monitoring"""
        keyboard_logger.info("KEYBOARD: Monitoring thread started")
        
        # Override the classify_activity method to send alerts to the master
        original_classify = self.keyboard_monitor.classify_activity
        
        def classify_wrapper():
            keyboard_logger.info("KEYBOARD: Analyzing keyboard activity")
            result = original_classify()
            if result:
                activity_type = result['activity_type']
                confidence = result['confidence']
                
                keyboard_logger.info("KEYBOARD: Activity classified as '%s' with confidence %.2f", 
                                    activity_type, confidence)
                
                if activity_type == "suspicious" and confidence > self.alert_thresholds["keyboard"]:
                    keyboard_logger.warning("KEYBOARD: Suspicious activity detected (confidence: %.2f)", confidence)
                    self.add_message("keyboard", f"Suspicious keyboard activity (confidence: {confidence:.2f})", confidence)
            
            return result
        
        self.keyboard_monitor.classify_activity = classify_wrapper
        
        # Start keyboard listener
        from pynput import keyboard
        keyboard_listener = keyboard.Listener(on_press=self.keyboard_monitor.on_press)
        keyboard_listener.start()
        
        # Start monitoring
        self.keyboard_monitor.start_monitoring()
        
        # Keep the thread running
        while self.running:
            time.sleep(1)
        
        # Clean up
        keyboard_listener.stop()
        self.keyboard_monitor.stop_monitoring()
    
    def mouse_monitoring_thread(self):
        """Thread for mouse monitoring"""
        mouse_logger.info("MOUSE: Monitoring thread started")
        
        # Override the detection logic to send alerts to the master
        original_analyze = self.mouse_detector.analyze_current_sequence
        
        def analyze_wrapper():
            mouse_logger.info("MOUSE: Analyzing mouse movement patterns")
            # Process the original logic first
            result = original_analyze()
            
            # Now check if there's a suspicious sequence
            if len(self.mouse_detector.current_sequence) >= self.mouse_detector.window_size:
                processed_data = []
                sequence = self.mouse_detector.current_sequence
                
                for i in range(1, len(sequence)):
                    prev = sequence[i-1]
                    curr = sequence[i]
                    
                    # Calculate features (similar to the original code)
                    duration = curr['timestamp'] - prev['timestamp']
                    distance = ((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)**0.5
                    speed = distance / duration if duration > 0 else 0
                    
                    # Calculate direction change
                    direction_change = 0
                    if i >= 2:
                        # This is simplified from the original code
                        prev_prev = sequence[i-2]
                        
                        v1_x, v1_y = prev['x'] - prev_prev['x'], prev['y'] - prev_prev['y']
                        v2_x, v2_y = curr['x'] - prev['x'], curr['y'] - prev['y']
                        
                        # Basic direction change calculation
                        mag_v1 = (v1_x**2 + v1_y**2)**0.5
                        mag_v2 = (v2_x**2 + v2_y**2)**0.5
                        
                        if mag_v1 > 0 and mag_v2 > 0:
                            dot_product = v1_x * v2_x + v1_y * v2_y
                            cos_angle = dot_product / (mag_v1 * mag_v2)
                            cos_angle = max(-1.0, min(1.0, cos_angle))  # Ensure valid range
                            direction_change = 1 - (cos_angle + 1) / 2  # Normalized change 0-1
                    
                    processed_data.append([duration, distance, speed, direction_change])
                
                if processed_data:
                    # Analyze data points for suspicious patterns
                    import numpy as np
                    
                    # Convert to numpy array
                    data = np.array(processed_data)
                    
                    # Look for unusual patterns
                    avg_speed = np.mean(data[:, 2])
                    max_speed = np.max(data[:, 2])
                    direction_changes = np.mean(data[:, 3])
                    
                    mouse_logger.debug("MOUSE: Analysis - avg_speed: %.2f, max_speed: %.2f, direction_changes: %.2f", 
                                      avg_speed, max_speed, direction_changes)
                    
                    # Detect unusual mouse behavior
                    is_suspicious = False
                    confidence = 0.0
                    suspicious_pattern = ""
                    
                    # Check for robotic movement
                    if direction_changes < 0.1 and max_speed > 1000:
                        is_suspicious = True
                        confidence = 0.8
                        suspicious_pattern = "robotic movement"
                    
                    # Check for erratic movement
                    if direction_changes > 0.8:
                        is_suspicious = True
                        confidence = 0.7
                        suspicious_pattern = "erratic movement"
                    
                    if is_suspicious and confidence > self.alert_thresholds["mouse"]:
                        mouse_logger.warning("MOUSE: Suspicious activity detected - %s (confidence: %.2f)", 
                                           suspicious_pattern, confidence)
                        self.add_message("mouse", f"Suspicious mouse movement - {suspicious_pattern} (confidence: {confidence:.2f})", confidence)
            
            return result
        
        self.mouse_detector.analyze_current_sequence = analyze_wrapper
        
        # Start the mouse listener but don't block this thread
        from pynput import mouse
        mouse_listener = mouse.Listener(
            on_move=self.mouse_detector.on_move,
            on_click=self.mouse_detector.on_click
        )
        mouse_listener.start()
        
        # Start the detection loop manually
        while self.running:
            if len(self.mouse_detector.current_sequence) >= self.mouse_detector.window_size:
                self.mouse_detector.analyze_current_sequence()
            time.sleep(1)
        
        # Clean up
        mouse_listener.stop()
    
    def add_message(self, source, message, confidence):
        """Add a message to the queue without prioritization"""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_alert_times[source] < self.cooldown_periods[source]:
            return
        
        # Update last alert time
        self.last_alert_times[source] = current_time
        
        # Add to regular queue (no priority)
        self.message_queue.put({
            'source': source,
            'message': message,
            'confidence': confidence,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Update overall suspicion score
        with self.lock:
            # Add to suspicion score based on confidence
            weight = {
                "screen": 0.5,      # Screen is 50% of overall score
                "keyboard": 0.3,    # Keyboard is 30% of overall score
                "mouse": 0.2        # Mouse is 20% of overall score
            }
            
            # Increase suspicion score
            score_increase = confidence * 100 * weight[source]
            self.suspicion_score = min(100, self.suspicion_score + score_increase)
            
            # Log the alert with the appropriate logger
            if source == "screen":
                screen_logger.warning("SCREEN: %s (confidence: %.2f)", message, confidence)
            elif source == "keyboard":
                keyboard_logger.warning("KEYBOARD: %s (confidence: %.2f)", message, confidence)
            else:  # mouse
                mouse_logger.warning("MOUSE: %s (confidence: %.2f)", message, confidence)
            
            master_logger.info(f"Current suspicion score: {self.suspicion_score:.2f}")
            
            # Check if we should trigger overall alert
            if self.suspicion_score > self.suspicion_threshold:
                self.trigger_major_alert()
    
    def message_monitoring_thread(self):
        """Thread to process messages from the queue without prioritization"""
        master_logger.info("Message monitoring thread started")
        
        while self.running:
            try:
                # Get message with timeout to allow checking running flag
                try:
                    message = self.message_queue.get(timeout=1)
                    
                    # Process the message
                    self.process_message(message)
                    
                    # Mark task as done
                    self.message_queue.task_done()
                except queue.Empty:
                    pass
                
                # Decay suspicion score over time
                with self.lock:
                    old_score = self.suspicion_score
                    # Decay by 5% every second
                    self.suspicion_score = max(0, self.suspicion_score * 0.95)
                    if old_score > 0 and int(old_score) != int(self.suspicion_score):
                        master_logger.debug(f"Suspicion score decayed: {old_score:.2f} -> {self.suspicion_score:.2f}")
            
            except Exception as e:
                master_logger.error(f"Error in message monitoring thread: {str(e)}")
    
    def process_message(self, message):
        """Process a single message without prioritization"""
        source = message['source']
        msg_content = message['message']
        confidence = message['confidence']
        timestamp = message['timestamp']
        
        # Get the appropriate logger and color
        if source == "screen":
            logger = screen_logger
            source_color = Fore.MAGENTA
        elif source == "keyboard":
            logger = keyboard_logger
            source_color = Fore.BLUE
        else:  # mouse
            logger = mouse_logger
            source_color = Fore.CYAN
        
        # Log to console and file with color coding
        message_display = f"{source_color}DETECTED ({source.upper()}){Style.RESET_ALL}: {msg_content}"
        details = f"Confidence: {confidence:.2f}, Time: {timestamp}"
        
        logger.warning(message_display)
        logger.warning(details)
        
        # Print to console for visibility in a formatted way
        print(f"\n{'='*20} ACTIVITY DETECTED {'='*20}")
        print(f"{source_color}Source: {source.upper()}{Style.RESET_ALL}")
        print(f"Message: {msg_content}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Time: {timestamp}")
        print(f"{'='*58}\n")
    
    def trigger_major_alert(self):
        """Trigger a major alert when overall suspicion is high"""
        major_alert_message = f"{Fore.RED}⚠️ MAJOR ALERT: High suspicion score ({self.suspicion_score:.2f}/100) ⚠️{Style.RESET_ALL}"
        master_logger.critical(major_alert_message)
        master_logger.critical("Multiple suspicious activities detected!")
        
        # Print to console in a highly visible format
        print(f"\n{'!'*60}")
        print(f"{Fore.RED}{'!'*5} MAJOR ALERT: HIGH SUSPICION DETECTED {'!'*5}{Style.RESET_ALL}")
        print(f"{Fore.RED}Suspicion Score: {self.suspicion_score:.2f}/100{Style.RESET_ALL}")
        print(f"{Fore.RED}Multiple suspicious activities across detection systems{Style.RESET_ALL}")
        print(f"{'!'*60}\n")
        
        # Reset suspicion score after major alert
        self.suspicion_score = self.suspicion_score * 0.5  # Reduce by half
    
    def start(self):
        """Start all monitoring threads"""
        master_logger.info("Starting Master AI system...")
        print(f"\n{Fore.GREEN}=== Master AI System Starting ==={Style.RESET_ALL}")
        print(f"{Fore.GREEN}Monitoring all AI detection systems equally{Style.RESET_ALL}")
        print(f"{Fore.GREEN}All alerts will be displayed with equal importance{Style.RESET_ALL}\n")
        
        self.running = True
        
        # Start individual threads
        self.threads = []
        
        # Screen monitoring thread
        screen_thread = threading.Thread(target=self.screen_monitoring_thread)
        screen_thread.daemon = True
        self.threads.append(screen_thread)
        
        # Keyboard monitoring thread
        keyboard_thread = threading.Thread(target=self.keyboard_monitoring_thread)
        keyboard_thread.daemon = True
        self.threads.append(keyboard_thread)
        
        # Mouse monitoring thread
        mouse_thread = threading.Thread(target=self.mouse_monitoring_thread)
        mouse_thread.daemon = True
        self.threads.append(mouse_thread)
        
        # Message processing thread
        message_thread = threading.Thread(target=self.message_monitoring_thread)
        message_thread.daemon = True
        self.threads.append(message_thread)
        
        # Start all threads
        for thread in self.threads:
            thread.start()
        
        master_logger.info("All monitoring threads started")
        
        try:
            # Main command loop
            while True:
                cmd = input(f"\n{Fore.GREEN}Enter command (status/reset/quit/debug): {Style.RESET_ALL}").strip().lower()
                
                if cmd == "status":
                    self.print_status()
                elif cmd == "reset":
                    self.reset_suspicion()
                elif cmd == "debug":
                    self.toggle_debug()
                elif cmd == "quit":
                    self.stop()
                    break
                else:
                    print(f"{Fore.YELLOW}Unknown command. Available commands: status, reset, debug, quit{Style.RESET_ALL}")
        
        except KeyboardInterrupt:
            master_logger.info("Keyboard interrupt received, shutting down...")
            self.stop()
    
    def toggle_debug(self):
        """Toggle debug logging"""
        current_level = logging.getLogger().level
        if current_level == logging.DEBUG:
            logging.getLogger().setLevel(logging.INFO)
            print(f"{Fore.YELLOW}Debug logging disabled{Style.RESET_ALL}")
            master_logger.info("Debug logging disabled")
        else:
            logging.getLogger().setLevel(logging.DEBUG)
            print(f"{Fore.YELLOW}Debug logging enabled{Style.RESET_ALL}")
            master_logger.info("Debug logging enabled")
    
    def print_status(self):
        """Print current system status"""
        print(f"\n{Fore.CYAN}=== Master AI System Status ==={Style.RESET_ALL}")
        print(f"Current suspicion score: {self.suspicion_score:.2f}/100")
        print(f"Score threshold for major alert: {self.suspicion_threshold}")
        
        print(f"\n{Fore.CYAN}Component Status:{Style.RESET_ALL}")
        print(f"- {Fore.MAGENTA}Screen monitoring{Style.RESET_ALL}: Active")
        print(f"- {Fore.BLUE}Keyboard monitoring{Style.RESET_ALL}: Active")
        print(f"- {Fore.CYAN}Mouse monitoring{Style.RESET_ALL}: Active")
        
        print(f"\n{Fore.CYAN}Recent Detections:{Style.RESET_ALL}")
        for source in ["screen", "keyboard", "mouse"]:
            last_alert = self.last_alert_times[source]
            source_color = Fore.MAGENTA if source == "screen" else Fore.BLUE if source == "keyboard" else Fore.CYAN
            if last_alert > 0:
                time_ago = time.time() - last_alert
                print(f"- Last {source_color}{source}{Style.RESET_ALL} detection: {time_ago:.1f} seconds ago")
            else:
                print(f"- No {source_color}{source}{Style.RESET_ALL} detections yet")
        
        print(f"\n{Fore.CYAN}Detection Thresholds:{Style.RESET_ALL}")
        for source, threshold in self.alert_thresholds.items():
            source_color = Fore.MAGENTA if source == "screen" else Fore.BLUE if source == "keyboard" else Fore.CYAN
            print(f"- {source_color}{source.capitalize()}{Style.RESET_ALL}: {threshold:.2f}")
        
        print(f"\n{Fore.CYAN}Cooldown Periods:{Style.RESET_ALL}")
        for source, period in self.cooldown_periods.items():
            source_color = Fore.MAGENTA if source == "screen" else Fore.BLUE if source == "keyboard" else Fore.CYAN
            print(f"- {source_color}{source.capitalize()}{Style.RESET_ALL}: {period} seconds")
    
    def reset_suspicion(self):
        """Reset the suspicion score"""
        with self.lock:
            old_score = self.suspicion_score
            self.suspicion_score = 0
            master_logger.info(f"Suspicion score reset from {old_score:.2f} to 0.00")
            print(f"{Fore.YELLOW}Suspicion score reset from {old_score:.2f} to 0.00{Style.RESET_ALL}")
    
    def stop(self):
        """Stop all monitoring threads"""
        master_logger.info("Stopping Master AI system...")
        print(f"\n{Fore.YELLOW}=== Shutting down Master AI System ==={Style.RESET_ALL}")
        self.running = False
        
        # Wait for threads to finish
        for i, thread in enumerate(self.threads):
            master_logger.info(f"Waiting for thread {i+1}/{len(self.threads)} to finish...")
            thread.join(timeout=2)
        
        master_logger.info("Master AI system stopped")
        print(f"{Fore.YELLOW}System shutdown complete{Style.RESET_ALL}")

def main():
    """Main function to run the Master AI system"""
    print(f"{Fore.GREEN}=== Master AI System ==={Style.RESET_ALL}")
    print("This system monitors and displays detections from all AI components equally:")
    print(f"- {Fore.MAGENTA}Screen monitoring{Style.RESET_ALL}")
    print(f"- {Fore.BLUE}Keyboard monitoring{Style.RESET_ALL}")
    print(f"- {Fore.CYAN}Mouse monitoring{Style.RESET_ALL}")
    print("\nStarting system...")
    
    try:
        master = MasterAI()
        master.start()
    except Exception as e:
        master_logger.error(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()