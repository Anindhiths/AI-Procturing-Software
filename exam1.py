import os
import time
import threading
import queue
import logging
from datetime import datetime
import colorama
from colorama import Fore, Style
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QRadioButton, QButtonGroup, QMessageBox,
                             QDialog, QSpacerItem, QSizePolicy, QFrame, QTextEdit, QScrollArea)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

# Import the individual detection systems
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
        self.message_queue = queue.Queue()
        self.init_components()
        
        self.alert_thresholds = {
            "screen": 0.6,
            "keyboard": 0.4,
            "mouse": 0.4
        }
        
        self.cooldown_periods = {
            "screen": 6,
            "keyboard": 3,
            "mouse": 1
        }
        
        self.last_alert_times = {
            "screen": 0,
            "keyboard": 0,
            "mouse": 0
        }
        
        self.suspicion_score = 0
        self.suspicion_threshold = 60
        
        # Strike counter for suspicious activities
        self.strike_counter = 0
        self.strike_threshold = 15  # Number of strikes before warning
        
        self.lock = threading.Lock()
        self.running = False
        
        master_logger.info("MasterAI initialized with equal visibility for all AI components")

    def init_components(self):
        try:
            templates_folder = os.path.expanduser("~/Development/projects/onlineexams/test_templates")
            screenshot_folder = os.path.expanduser("~/Development/projects/onlineexams/temp_screenshots")
            
            os.makedirs(templates_folder, exist_ok=True)
            os.makedirs(screenshot_folder, exist_ok=True)
            
            master_logger.info("Initializing detection components...")
            
            self.screen_monitor = ScreenMonitor(
                templates_folder=templates_folder,
                screenshot_folder=screenshot_folder,
                interval=3
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
        screen_logger.info("SCREEN: Monitoring thread started")
        
        def process_screen_result(matches):
            suspicious = len(matches) == 0
            if suspicious:
                confidence = 0.9
                screen_logger.warning("SCREEN: Suspicious activity detected - No matches found (confidence: %.2f)", confidence)
                self.add_message("screen", "Suspicious screen activity detected", confidence)
            else:
                avg_similarity = sum(match['similarity'] for match in matches) / len(matches)
                if avg_similarity < self.alert_thresholds["screen"]:
                    screen_logger.warning("SCREEN: Unusual screen content detected (similarity: %.2f)", avg_similarity)
                    self.add_message("screen", f"Unusual screen content (similarity: {avg_similarity:.2f})", avg_similarity)
                else:
                    screen_logger.debug("SCREEN: Normal screen activity (similarity: %.2f)", avg_similarity)
        
        original_compare = self.screen_monitor._compare_with_templates
        
        def compare_wrapper(screenshot_path):
            matches = original_compare(screenshot_path)
            process_screen_result(matches)
            return matches
        
        self.screen_monitor._compare_with_templates = compare_wrapper
        
        self.screen_monitor.start_monitoring()
        
        while self.running:
            time.sleep(1)
        
        self.screen_monitor.stop_monitoring()
    
    def keyboard_monitoring_thread(self):
        keyboard_logger.info("KEYBOARD: Monitoring thread started")
        
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
        
        from pynput import keyboard
        keyboard_listener = keyboard.Listener(on_press=self.keyboard_monitor.on_press)
        keyboard_listener.start()
        
        self.keyboard_monitor.start_monitoring()
        
        while self.running:
            time.sleep(1)
        
        keyboard_listener.stop()
        self.keyboard_monitor.stop_monitoring()
    
    def mouse_monitoring_thread(self):
        mouse_logger.info("MOUSE: Monitoring thread started")
        
        original_analyze = self.mouse_detector.analyze_current_sequence
        
        def analyze_wrapper():
            mouse_logger.info("MOUSE: Analyzing mouse movement patterns")
            result = original_analyze()
            
            if len(self.mouse_detector.current_sequence) >= self.mouse_detector.window_size:
                processed_data = []
                sequence = self.mouse_detector.current_sequence
                
                for i in range(1, len(sequence)):
                    prev = sequence[i-1]
                    curr = sequence[i]
                    
                    duration = curr['timestamp'] - prev['timestamp']
                    distance = ((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)**0.5
                    speed = distance / duration if duration > 0 else 0
                    
                    direction_change = 0
                    if i >= 2:
                        prev_prev = sequence[i-2]
                        
                        v1_x, v1_y = prev['x'] - prev_prev['x'], prev['y'] - prev_prev['y']
                        v2_x, v2_y = curr['x'] - prev['x'], curr['y'] - prev['y']
                        
                        mag_v1 = (v1_x**2 + v1_y**2)**0.5
                        mag_v2 = (v2_x**2 + v2_y**2)**0.5
                        
                        if mag_v1 > 0 and mag_v2 > 0:
                            dot_product = v1_x * v2_x + v1_y * v2_y
                            cos_angle = dot_product / (mag_v1 * mag_v2)
                            cos_angle = max(-1.0, min(1.0, cos_angle))
                            direction_change = 1 - (cos_angle + 1) / 2
                    
                    processed_data.append([duration, distance, speed, direction_change])
                
                if processed_data:
                    import numpy as np
                    
                    data = np.array(processed_data)
                    
                    avg_speed = np.mean(data[:, 2])
                    max_speed = np.max(data[:, 2])
                    direction_changes = np.mean(data[:, 3])
                    
                    mouse_logger.debug("MOUSE: Analysis - avg_speed: %.2f, max_speed: %.2f, direction_changes: %.2f", 
                                      avg_speed, max_speed, direction_changes)
                    
                    is_suspicious = False
                    confidence = 0.0
                    suspicious_pattern = ""
                    
                    if direction_changes < 0.1 and max_speed > 1000:
                        is_suspicious = True
                        confidence = 0.8
                        suspicious_pattern = "robotic movement"
                    
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
        
        from pynput import mouse
        mouse_listener = mouse.Listener(
            on_move=self.mouse_detector.on_move,
            on_click=self.mouse_detector.on_click
        )
        mouse_listener.start()
        
        while self.running:
            if len(self.mouse_detector.current_sequence) >= self.mouse_detector.window_size:
                self.mouse_detector.analyze_current_sequence()
            time.sleep(1)
        
        mouse_listener.stop()
    
    def add_message(self, source, message, confidence):
        current_time = time.time()
        
        if current_time - self.last_alert_times[source] < self.cooldown_periods[source]:
            return
        
        self.last_alert_times[source] = current_time
        
        self.message_queue.put({
            'source': source,
            'message': message,
            'confidence': confidence,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        with self.lock:
            weight = {
                "screen": 0.5,
                "keyboard": 0.3,
                "mouse": 0.2
            }
            
            score_increase = confidence * 100 * weight[source]
            self.suspicion_score = min(100, self.suspicion_score + score_increase)
            
            if source == "screen":
                screen_logger.warning("SCREEN: %s (confidence: %.2f)", message, confidence)
            elif source == "keyboard":
                keyboard_logger.warning("KEYBOARD: %s (confidence: %.2f)", message, confidence)
            else:
                mouse_logger.warning("MOUSE: %s (confidence: %.2f)", message, confidence)
            
            master_logger.info(f"Current suspicion score: {self.suspicion_score:.2f}")
            
            # Increment strike counter if suspicion score is high
            if self.suspicion_score > self.suspicion_threshold:
                self.strike_counter += 1
                master_logger.warning(f"Strike {self.strike_counter} detected! Suspicion score: {self.suspicion_score:.2f}")
                
                # If strike counter reaches the threshold, trigger warning and reset
                if self.strike_counter >= self.strike_threshold:
                    
                    self.trigger_warning_popup()
                    self.strike_counter = 0  # Reset strike counter after warning
                    self.suspicion_score = 0  # Reset suspicion score after warning
    
    def trigger_warning_popup(self):
        
        """Trigger a warning popup when strike counter reaches the threshold"""
        master_logger.critical("Strike threshold reached! Triggering warning popup.")
        
        # Create a QMessageBox for the warning
        warning_box = QMessageBox()
        warning_box.setWindowTitle("Warning")
        warning_box.setText("Multiple suspicious activities detected. You have been warned.")
        warning_box.setInformativeText("The exam will now exit to the start page.")
        warning_box.setIcon(QMessageBox.Warning)
        
        # Set font size for the warning box
        font = warning_box.font()
        font.setPointSize(24)
        warning_box.setFont(font)
        
        # Add buttons
        warning_box.addButton(QMessageBox.Ok)
        
        # Show the warning box
        warning_box.exec_()
        
        # Exit to the start page
        self.exit_to_start_page()
    
    def exit_to_start_page(self):
        """Exit to the start page"""
        master_logger.info("Exiting to the start page due to suspicious activities.")
        
        # Close the current exam page and show the welcome page
        if hasattr(self, 'exam_page'):
            self.exam_page.close()
        
        self.welcome_page = WelcomePage()
        self.welcome_page.showFullScreen()
    
    def message_monitoring_thread(self):
        master_logger.info("Message monitoring thread started")
        
        while self.running:
            try:
                try:
                    message = self.message_queue.get(timeout=1)
                    self.process_message(message)
                    self.message_queue.task_done()
                except queue.Empty:
                    pass
                
                with self.lock:
                    old_score = self.suspicion_score
                    self.suspicion_score = max(0, self.suspicion_score * 0.95)
                    if old_score > 0 and int(old_score) != int(self.suspicion_score):
                        master_logger.debug(f"Suspicion score decayed: {old_score:.2f} -> {self.suspicion_score:.2f}")
            
            except Exception as e:
                master_logger.error(f"Error in message monitoring thread: {str(e)}")
    
    def process_message(self, message):
        source = message['source']
        msg_content = message['message']
        confidence = message['confidence']
        timestamp = message['timestamp']
        
        if source == "screen":
            logger = screen_logger
            source_color = Fore.MAGENTA
        elif source == "keyboard":
            logger = keyboard_logger
            source_color = Fore.BLUE
        else:
            logger = mouse_logger
            source_color = Fore.CYAN
        
        message_display = f"{source_color}DETECTED ({source.upper()}){Style.RESET_ALL}: {msg_content}"
        details = f"Confidence: {confidence:.2f}, Time: {timestamp}"
        
        logger.warning(message_display)
        logger.warning(details)
        
        print(f"\n{'='*20} ACTIVITY DETECTED {'='*20}")
        print(f"{source_color}Source: {source.upper()}{Style.RESET_ALL}")
        print(f"Message: {msg_content}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Time: {timestamp}")
        print(f"{'='*58}\n")
    
    def trigger_major_alert(self):
        major_alert_message = f"{Fore.RED}⚠️ MAJOR ALERT: High suspicion score ({self.suspicion_score:.2f}/100) ⚠️{Style.RESET_ALL}"
        master_logger.critical(major_alert_message)
        master_logger.critical("Multiple suspicious activities detected!")
        
        print(f"\n{'!'*60}")
        print(f"{Fore.RED}{'!'*5} MAJOR ALERT: HIGH SUSPICION DETECTED {'!'*5}{Style.RESET_ALL}")
        print(f"{Fore.RED}Suspicion Score: {self.suspicion_score:.2f}/100{Style.RESET_ALL}")
        print(f"{Fore.RED}Multiple suspicious activities across detection systems{Style.RESET_ALL}")
        print(f"{'!'*60}\n")
        
        self.suspicion_score = self.suspicion_score * 0.5
    
    def start(self):
        master_logger.info("Starting Master AI system...")
        print(f"\n{Fore.GREEN}=== Master AI System Starting ==={Style.RESET_ALL}")
        print(f"{Fore.GREEN}Monitoring all AI detection systems equally{Style.RESET_ALL}")
        print(f"{Fore.GREEN}All alerts will be displayed with equal importance{Style.RESET_ALL}\n")
        
        self.running = True
        
        self.threads = []
        
        screen_thread = threading.Thread(target=self.screen_monitoring_thread)
        screen_thread.daemon = True
        self.threads.append(screen_thread)
        
        keyboard_thread = threading.Thread(target=self.keyboard_monitoring_thread)
        keyboard_thread.daemon = True
        self.threads.append(keyboard_thread)
        
        mouse_thread = threading.Thread(target=self.mouse_monitoring_thread)
        mouse_thread.daemon = True
        self.threads.append(mouse_thread)
        
        message_thread = threading.Thread(target=self.message_monitoring_thread)
        message_thread.daemon = True
        self.threads.append(message_thread)
        
        for thread in self.threads:
            thread.start()
        
        master_logger.info("All monitoring threads started")
        
        try:
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
        with self.lock:
            old_score = self.suspicion_score
            self.suspicion_score = 0
            master_logger.info(f"Suspicion score reset from {old_score:.2f} to 0.00")
            print(f"{Fore.YELLOW}Suspicion score reset from {old_score:.2f} to 0.00{Style.RESET_ALL}")
    
    def stop(self):
        master_logger.info("Stopping Master AI system...")
        print(f"\n{Fore.YELLOW}=== Shutting down Master AI System ==={Style.RESET_ALL}")
        self.running = False
        
        for i, thread in enumerate(self.threads):
            master_logger.info(f"Waiting for thread {i+1}/{len(self.threads)} to finish...")
            thread.join(timeout=2)
        
        master_logger.info("Master AI system stopped")
        print(f"{Fore.YELLOW}System shutdown complete{Style.RESET_ALL}")

class WelcomePage(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Programming Exam")
        self.setMinimumSize(800, 600)
        self.set_white_theme()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        exit_button_layout = QHBoxLayout()
        exit_button_layout.addStretch()
        exit_button = QPushButton("Exit Application")
        exit_button.setMinimumSize(200, 60)
        exit_button.setStyleSheet("background-color: red; color: white; font-size: 24px;")
        exit_button.clicked.connect(self.exit_application)
        exit_button_layout.addWidget(exit_button)
        main_layout.addLayout(exit_button_layout)
        
        title_label = QLabel("Welcome to the PyQt Programming Exam")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(36)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        main_layout.addWidget(title_label)
        
        description_label = QLabel("You will have 30 minutes to complete 5 questions. Please ensure you are ready before starting the exam.")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setWordWrap(True)
        description_font = QFont()
        description_font.setPointSize(28)
        description_label.setFont(description_font)
        main_layout.addSpacing(20)
        main_layout.addWidget(description_label)
        
        main_layout.addSpacing(40)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        start_button = QPushButton("Start Exam")
        start_button.setMinimumSize(300, 80)
        start_button.setStyleSheet("background-color: green; color: white; font-size: 32px;")
        start_button.clicked.connect(self.start_exam)
        button_layout.addWidget(start_button)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        note_label = QLabel("Note: For Mac users, you can use the button above or ⌘+Control+F to toggle full screen.")
        note_label.setAlignment(Qt.AlignCenter)
        note_font = QFont()
        note_font.setItalic(True)
        note_font.setPointSize(24)
        note_label.setFont(note_font)
        main_layout.addSpacing(20)
        main_layout.addWidget(note_label)
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
    
    def set_white_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(255, 255, 255))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        self.setPalette(palette)
    
    def exit_application(self):
        QApplication.quit()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def start_exam(self):
        self.exam_page = ExamPage("PyQt Programming Exam")
        self.exam_page.showFullScreen()
        self.close()

class ExamPage(QMainWindow):
    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(800, 600)
        self.set_white_theme()
        
        self.current_question_index = 0
        self.questions = [
            {
                'question': 'Which widget is used to create a scrollable area?',
                'options': ['QWidget', 'QScrollArea', 'QVBoxLayout', 'QFrame'],
                'correct_answer': 1,
                'selected_answer': None,
                'type': 'multiple_choice'
            },
            {
                'question': 'What is the main purpose of signals and slots in PyQt?',
                'options': [
                    'To display static content',
                    'To handle network requests',
                    'To connect events with functions to handle them',
                    'To define app themes'
                ],
                'correct_answer': 2,
                'selected_answer': None,
                'type': 'multiple_choice'
            },
            {
                'question': 'Which layout is used to arrange widgets horizontally?',
                'options': ['QVBoxLayout', 'QHBoxLayout', 'QGridLayout', 'QFormLayout'],
                'correct_answer': 1,
                'selected_answer': None,
                'type': 'multiple_choice'
            },
            {
                'question': 'What is the purpose of the QTimer class?',
                'options': [
                    'To create a new widget',
                    'To navigate to a new screen',
                    'To schedule and execute code at specified intervals',
                    'To reset the application'
                ],
                'correct_answer': 2,
                'selected_answer': None,
                'type': 'multiple_choice'
            },
            {
                'question': 'Write a Python function that takes a list of numbers and returns the sum of all even numbers in the list.',
                'correct_answer': """def sum_even_numbers(numbers):
    total = 0
    for num in numbers:
        if num % 2 == 0:
            total += num
    return total""",
                'selected_answer': None,
                'type': 'programming'
            }
        ]
        
        self.remaining_time = 30 * 60
        self.timer_running = False
        self.exam_completed = False
        
        central_widget = QWidget()
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)
        
        self.main_layout = QVBoxLayout(central_widget)
        
        fullscreen_layout = QHBoxLayout()
        fullscreen_button = QPushButton("Toggle Full Screen")
        fullscreen_button.setStyleSheet("background-color: blue; color: white; font-size: 24px;")
        fullscreen_button.setMinimumSize(200, 50)
        fullscreen_button.clicked.connect(self.toggle_fullscreen)
        fullscreen_layout.addWidget(fullscreen_button)
        fullscreen_layout.addStretch()
        self.main_layout.addLayout(fullscreen_layout)
        
        self.timer_label = QLabel(f"Time Remaining: {self.format_time(self.remaining_time)}")
        timer_font = QFont()
        timer_font.setBold(True)
        timer_font.setPointSize(32)
        self.timer_label.setFont(timer_font)
        self.timer_label.setAlignment(Qt.AlignRight)
        
        indicator_layout = QHBoxLayout()
        question_label = QLabel(f"Q.{self.current_question_index + 1}")
        question_font = QFont()
        question_font.setBold(True)
        question_font.setPointSize(32)
        question_label.setFont(question_font)
        indicator_layout.addWidget(question_label)
        self.question_label = question_label
        
        indicator_layout.addStretch()
        self.indicators = []
        for i in range(len(self.questions)):
            indicator = QPushButton(str(i + 1))
            indicator.setFixedSize(80, 80)
            indicator.setStyleSheet("background-color: gray; color: white; border-radius: 40px; font-size: 28px;")
            indicator.clicked.connect(lambda _, idx=i: self.go_to_question(idx))
            indicator_layout.addWidget(indicator)
            self.indicators.append(indicator)
        
        self.update_indicators()
        
        self.question_text = QLabel()
        self.question_text.setWordWrap(True)
        question_text_font = QFont()
        question_text_font.setPointSize(32)
        self.question_text.setFont(question_text_font)
        
        self.options_layout = QVBoxLayout()
        self.option_group = QButtonGroup(self)
        self.option_buttons = []
        
        for i in range(4):
            option = QRadioButton()
            option_font = QFont()
            option_font.setPointSize(28)
            option.setFont(option_font)
            option.setStyleSheet("QRadioButton::indicator { width: 30px; height: 30px; }")
            self.option_buttons.append(option)
            self.option_group.addButton(option, i)
            self.options_layout.addWidget(option)
        
        self.option_group.buttonClicked.connect(self.select_answer)
        
        self.code_editor = QTextEdit()
        code_font = QFont("Courier New", 28)
        self.code_editor.setFont(code_font)
        self.code_editor.setMinimumHeight(400)
        self.code_editor.textChanged.connect(self.update_code_answer)
        
        navigation_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous Question")
        self.prev_button.setFont(QFont("Arial", 24))
        self.prev_button.setMinimumSize(250, 70)
        self.prev_button.clicked.connect(self.go_to_previous_question)
        
        self.next_button = QPushButton("Next Question")
        self.next_button.setFont(QFont("Arial", 24))
        self.next_button.setMinimumSize(250, 70)
        self.next_button.clicked.connect(self.go_to_next_question)
        
        self.submit_button = QPushButton("Submit")
        self.submit_button.setFont(QFont("Arial", 24))
        self.submit_button.setMinimumSize(250, 70)
        self.submit_button.setStyleSheet("background-color: green; color: white;")
        self.submit_button.clicked.connect(self.submit_exam)
        
        navigation_layout.addWidget(self.prev_button)
        navigation_layout.addWidget(self.next_button)
        navigation_layout.addWidget(self.submit_button)
        
        self.main_layout.addWidget(self.timer_label)
        self.main_layout.addLayout(indicator_layout)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setLineWidth(3)
        self.main_layout.addWidget(separator)
        
        self.main_layout.addSpacing(30)
        self.main_layout.addWidget(self.question_text)
        self.main_layout.addSpacing(30)
        self.main_layout.addLayout(self.options_layout)
        self.main_layout.addWidget(self.code_editor)
        self.main_layout.addStretch()
        
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        separator2.setLineWidth(3)
        self.main_layout.addWidget(separator2)
        
        self.main_layout.addLayout(navigation_layout)
        
        self.load_question()
        
        self.start_timer()
    
    def set_white_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(255, 255, 255))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        self.setPalette(palette)
    
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def start_timer(self):
        self.timer_running = True
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)
    
    def update_timer(self):
        if not self.timer_running:
            return
        
        self.remaining_time -= 1
        self.timer_label.setText(f"Time Remaining: {self.format_time(self.remaining_time)}")
        
        if self.remaining_time <= 300:
            self.timer_label.setStyleSheet("color: red; font-weight: bold; font-size: 32pt;")
        
        if self.remaining_time <= 0:
            self.timer_running = False
            self.exam_completed = True
            self.submit_exam(True)
    
    def format_time(self, seconds):
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:02d}:{remaining_seconds:02d}"
    
    def load_question(self):
        self.update_indicators()
        
        current_question = self.questions[self.current_question_index]
        
        self.question_text.setText(current_question['question'])
        
        if current_question['type'] == 'multiple_choice':
            for button in self.option_buttons:
                self.option_group.removeButton(button)
                self.options_layout.removeWidget(button)
                button.deleteLater()
            
            self.option_buttons = []
            
            for i, option_text in enumerate(current_question['options']):
                option = QRadioButton(f"{chr(97 + i)}) {option_text}")
                option_font = QFont()
                option_font.setPointSize(28)
                option.setFont(option_font)
                option.setStyleSheet("""
                    QRadioButton { 
                        font-size: 28px; 
                        color: black; 
                        font-weight: bold;
                        padding: 5px;
                    }
                    QRadioButton::indicator { 
                        width: 30px; 
                        height: 30px; 
                    }
                """)
                self.option_buttons.append(option)
                self.option_group.addButton(option, i)
                self.options_layout.addWidget(option)
            
            if current_question['selected_answer'] is not None:
                self.option_buttons[current_question['selected_answer']].setChecked(True)
            
            self.code_editor.setVisible(False)
            for button in self.option_buttons:
                button.setVisible(True)
            
        elif current_question['type'] == 'programming':
            for button in self.option_buttons:
                button.setVisible(False)
            
            self.code_editor.setVisible(True)
            
            if current_question['selected_answer'] is not None:
                self.code_editor.setText(current_question['selected_answer'])
            else:
                self.code_editor.setText("# Write your code here")
        
        self.prev_button.setEnabled(self.current_question_index > 0)
        self.next_button.setEnabled(self.current_question_index < len(self.questions) - 1)
    
    def update_indicators(self):
        for i, indicator in enumerate(self.indicators):
            if i == self.current_question_index:
                indicator.setStyleSheet("background-color: blue; color: white; border-radius: 40px; border: 4px solid darkblue; font-size: 28px;")
            elif self.questions[i]['selected_answer'] is not None:
                indicator.setStyleSheet("background-color: green; color: white; border-radius: 40px; font-size: 28px;")
            else:
                indicator.setStyleSheet("background-color: gray; color: white; border-radius: 40px; font-size: 28px;")
        
        self.question_label.setText(f"Q.{self.current_question_index + 1}")
    
    def select_answer(self, button):
        if self.exam_completed:
            return
        
        option_index = self.option_group.id(button)
        if option_index != -1:
            self.questions[self.current_question_index]['selected_answer'] = option_index
            self.update_indicators()
    
    def update_code_answer(self):
        if self.exam_completed:
            return
        
        current_question = self.questions[self.current_question_index]
        if current_question['type'] == 'programming':
            current_question['selected_answer'] = self.code_editor.toPlainText()
            self.update_indicators()
    
    def go_to_question(self, index):
        self.current_question_index = index
        self.load_question()
    
    def go_to_next_question(self):
        if self.current_question_index < len(self.questions) - 1:
            self.current_question_index += 1
            self.load_question()
    
    def go_to_previous_question(self):
        if self.current_question_index > 0:
            self.current_question_index -= 1
            self.load_question()
    
    def submit_exam(self, time_up=False):
        if self.exam_completed:
            return
        
        self.timer_running = False
        self.exam_completed = True
        
        correct_answers = 0
        attempted_questions = 0
        
        for question in self.questions:
            if question['selected_answer'] is not None:
                attempted_questions += 1
                if question['type'] == 'multiple_choice':
                    if question['selected_answer'] == question['correct_answer']:
                        correct_answers += 1
                elif question['type'] == 'programming':
                    correct_answers += 1
        
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Time's Up!" if time_up else "Exam Results")
        dialog.setText(f"You scored {correct_answers} out of {len(self.questions)}.")
        dialog.setInformativeText(f"Questions attempted: {attempted_questions} of {len(self.questions)}\n"
                                f"Time remaining: {self.format_time(self.remaining_time)}")
        
        font = dialog.font()
        font.setPointSize(28)
        dialog.setFont(font)
        
        review_button = dialog.addButton("Review Answers", QMessageBox.AcceptRole)
        exit_button = dialog.addButton("Exit Exam", QMessageBox.RejectRole)
        
        for button in dialog.buttons():
            button_font = button.font()
            button_font.setPointSize(24)
            button.setFont(button_font)
            button.setMinimumHeight(60)
        
        dialog.exec_()
        
        clicked_button = dialog.clickedButton()
        if clicked_button == exit_button:
            self.close()
            self.timer_running = False
            self.welcome_page = WelcomePage()
            self.welcome_page.showFullScreen()
    
    def closeEvent(self, event):
        if not self.exam_completed:
            dialog = QMessageBox(self)
            dialog.setWindowTitle('Exit Exam?')
            dialog.setText('Are you sure you want to exit the exam? Your progress will be lost.')
            dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            dialog.setDefaultButton(QMessageBox.No)
            
            font = dialog.font()
            font.setPointSize(28)
            dialog.setFont(font)
            
            for button in dialog.buttons():
                button_font = button.font()
                button_font.setPointSize(24)
                button.setFont(button_font)
                button.setMinimumHeight(60)
            
            reply = dialog.exec_()
            
            if reply == QMessageBox.Yes:
                event.accept()
                self.timer_running = False
                self.welcome_page = WelcomePage()
                self.welcome_page.showFullScreen()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    app = QApplication(sys.argv)
    font = QFont()
    font.setPointSize(18)
    app.setFont(font)
    
    window = WelcomePage()
    window.showFullScreen()
    
    # Start the MasterAI system
    master = MasterAI()
    master_thread = threading.Thread(target=master.start)
    master_thread.daemon = True
    master_thread.start()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
