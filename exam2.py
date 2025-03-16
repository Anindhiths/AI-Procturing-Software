import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QRadioButton, QButtonGroup, QFrame, 
                             QStackedWidget, QProgressBar, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette, QLinearGradient, QGradient, QBrush, QPixmap

class ProctorExamApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Proctor Exam System")
        self.setMinimumSize(1000, 700)
        
        # Set application style
        self.set_app_style()
        
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header section
        self.setup_header()
        
        # Content section (stacked widget for different exam pages)
        self.content = QStackedWidget()
        self.content.setStyleSheet("""
            QStackedWidget {
                background-color: #f8f9fa;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
        """)
        self.main_layout.addWidget(self.content)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 4)
        self.progress_bar.setValue(1)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #e0e0e0;
                border-radius: 4px;
                border: none;
            }
            QProgressBar::chunk {
                background-color: #6200EE;
                border-radius: 4px;
            }
        """)
        self.main_layout.addWidget(self.progress_bar)
        
        # Create question pages
        self.create_question_pages()
        
        # Footer section with navigation
        self.setup_footer()
        
        # Set initial page
        self.current_question = 0
        self.content.setCurrentIndex(self.current_question)
        self.update_question_indicators()
        
        # Start the exam timer
        self.remaining_time = 20 * 60  # 20 minutes in seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.timer.start(1000)  # Update every second
        self.update_timer()
        
    def set_app_style(self):
        # Set application-wide style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QLabel {
                color: #303030;
            }
            QPushButton {
                background-color: #6200EE;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7722ff;
            }
            QPushButton:pressed {
                background-color: #5000cc;
            }
            QRadioButton {
                spacing: 10px;
                color: #303030;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QRadioButton::indicator:checked {
                background-color: #6200EE;
                border: 2px solid #6200EE;
                border-radius: 9px;
            }
            QRadioButton::indicator:unchecked {
                border: 2px solid #a0a0a0;
                border-radius: 9px;
                background-color: white;
            }
        """)
        
    def setup_header(self):
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(10, 10, 10, 20)
        
        # Exam title and logo area
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        
        # Logo (placeholder - in real app, use actual logo)
        logo_label = QLabel()
        logo_label.setFixedSize(40, 40)
        logo_label.setStyleSheet("""
            background-color: #6200EE;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            text-align: center;
        """)
        logo_label.setText("PE")
        logo_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(logo_label)
        
        # Exam title
        exam_title = QLabel("Advanced Proctoring System")
        exam_title.setFont(QFont("Arial", 20, QFont.Bold))
        exam_title.setStyleSheet("color: #303030;")
        title_layout.addWidget(exam_title)
        
        header_layout.addWidget(title_container)
        
        # Spacer to push timer to the right
        header_layout.addStretch()
        
        # Controls container
        controls_container = QWidget()
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setSpacing(15)
        
        # Fullscreen button with better styling
        fullscreen_btn = QPushButton("Enter Full Screen")
        fullscreen_btn.setIcon(QIcon.fromTheme("view-fullscreen"))
        fullscreen_btn.setStyleSheet("""
            QPushButton {
                background-color: #03DAC5;
                color: black;
            }
            QPushButton:hover {
                background-color: #00c4b0;
            }
        """)
        fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        controls_layout.addWidget(fullscreen_btn)
        
        # Timer display with improved styling
        self.timer_label = QLabel("Time Remaining: 20:00")
        self.timer_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.timer_label.setStyleSheet("""
            padding: 10px; 
            border-radius: 8px; 
            background-color: #FFF;
            border: 1px solid #e0e0e0;
            color: #303030;
        """)
        controls_layout.addWidget(self.timer_label)
        
        header_layout.addWidget(controls_container)
        self.main_layout.addWidget(header_container)
        
        # Separator line with better styling
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #e0e0e0; min-height: 1px;")
        self.main_layout.addWidget(separator)
        
    def setup_footer(self):
        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #e0e0e0; min-height: 1px;")
        self.main_layout.addWidget(separator)
        
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 15, 0, 0)
        
        # Previous button with improved styling
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.go_to_previous)
        self.prev_btn.setStyleSheet("""
            QPushButton {
                background-color: #BB86FC;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #a275e3;
            }
        """)
        footer_layout.addWidget(self.prev_btn)
        
        # Question indicators with better styling
        indicator_container = QWidget()
        self.indicator_layout = QHBoxLayout(indicator_container)
        self.indicator_layout.setAlignment(Qt.AlignCenter)
        self.indicator_layout.setSpacing(15)
        self.question_indicators = []
        
        for i in range(4):  # Create 4 question indicators
            indicator = QPushButton(str(i+1))
            indicator.setFixedSize(45, 45)
            indicator.setStyleSheet("""
                QPushButton { 
                    border-radius: 22px; 
                    background-color: #e0e0e0; 
                    color: #303030;
                    font-size: 16px;
                    font-weight: bold;
                }
                QPushButton:checked { 
                    background-color: #6200EE; 
                    color: white;
                }
            """)
            indicator.setCheckable(True)
            indicator.clicked.connect(lambda checked, idx=i: self.jump_to_question(idx))
            self.question_indicators.append(indicator)
            self.indicator_layout.addWidget(indicator)
            
        footer_layout.addWidget(indicator_container)
        
        # Next button with improved styling
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.go_to_next)
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #BB86FC;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #a275e3;
            }
        """)
        footer_layout.addWidget(self.next_btn)
        
        # Submit button with improved styling
        self.submit_btn = QPushButton("Submit Exam")
        self.submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #03DAC5;
                color: black;
                padding: 10px 25px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #02b3a0;
            }
        """)
        self.submit_btn.clicked.connect(self.submit_exam)
        footer_layout.addWidget(self.submit_btn)
        
        self.main_layout.addWidget(footer)
        
    def create_question_pages(self):
        # Sample questions
        questions = [
            {
                "number": 1,
                "text": "Which widget is used to create a scrollable list of widgets?",
                "options": ["Container", "ListView", "Column", "Row"]
            },
            {
                "number": 2,
                "text": "In PyQt, which class is the base class for all user interface objects?",
                "options": ["QWidget", "QMainWindow", "QApplication", "QObject"]
            },
            {
                "number": 3,
                "text": "Which layout manager arranges widgets vertically in PyQt?",
                "options": ["QHBoxLayout", "QVBoxLayout", "QGridLayout", "QFormLayout"]
            },
            {
                "number": 4,
                "text": "What is the purpose of QStackedWidget?",
                "options": ["To create tab interfaces", "To show only one widget at a time", "To create scrollable areas", "To manage dialog windows"]
            }
        ]
        
        # Create a page for each question
        for question in questions:
            page = self.create_question_page(question)
            self.content.addWidget(page)
    
    def create_question_page(self, question):
        # Question page container with scroll area
        container = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Question card
        question_card = QWidget()
        question_card.setStyleSheet("""
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
        """)
        card_layout = QVBoxLayout(question_card)
        
        # Question number and badge
        question_header = QWidget()
        header_layout = QHBoxLayout(question_header)
        header_layout.setContentsMargins(0, 0, 0, 15)
        
        question_badge = QLabel(f"{question['number']}")
        question_badge.setAlignment(Qt.AlignCenter)
        question_badge.setFixedSize(35, 35)
        question_badge.setStyleSheet("""
            background-color: #6200EE;
            color: white;
            border-radius: 17px;
            font-weight: bold;
            font-size: 16px;
        """)
        header_layout.addWidget(question_badge)
        
        question_label = QLabel("Question")
        question_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(question_label)
        header_layout.addStretch()
        
        card_layout.addWidget(question_header)
        
        # Question text with better styling
        question_text = QLabel(question["text"])
        question_text.setWordWrap(True)
        question_text.setFont(QFont("Arial", 14))
        question_text.setStyleSheet("color: #303030; line-height: 1.4;")
        card_layout.addWidget(question_text)
        
        # Separator between question and options
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #e0e0e0; min-height: 1px; margin: 10px 0;")
        card_layout.addWidget(separator)
        
        # Options title
        options_label = QLabel("Select one option:")
        options_label.setFont(QFont("Arial", 12))
        options_label.setStyleSheet("color: #606060; margin-top: 5px;")
        card_layout.addWidget(options_label)
        
        # Radio button options with better styling
        option_group = QButtonGroup(container)
        option_letters = ["A", "B", "C", "D"]
        
        options_container = QWidget()
        options_layout = QVBoxLayout(options_container)
        options_layout.setContentsMargins(10, 10, 10, 10)
        options_layout.setSpacing(15)
        
        for i, option_text in enumerate(question["options"]):
            option_card = QWidget()
            option_card.setStyleSheet("""
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                padding: 5px;
            """)
            option_layout = QHBoxLayout(option_card)
            
            option = QRadioButton(f"{option_letters[i]}) {option_text}")
            option.setFont(QFont("Arial", 12))
            option_group.addButton(option)
            option_layout.addWidget(option)
            
            options_layout.addWidget(option_card)
        
        options_container.setLayout(options_layout)
        card_layout.addWidget(options_container)
        
        layout.addWidget(question_card)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        return scroll
    
    def update_question_indicators(self):
        for i, indicator in enumerate(self.question_indicators):
            indicator.setChecked(i == self.current_question)
        
        # Also update progress bar
        self.progress_bar.setValue(self.current_question + 1)
            
    def jump_to_question(self, question_index):
        self.current_question = question_index
        self.content.setCurrentIndex(question_index)
        self.update_question_indicators()
        
    def go_to_next(self):
        if self.current_question < self.content.count() - 1:
            self.current_question += 1
            self.content.setCurrentIndex(self.current_question)
            self.update_question_indicators()
            
    def go_to_previous(self):
        if self.current_question > 0:
            self.current_question -= 1
            self.content.setCurrentIndex(self.current_question)
            self.update_question_indicators()
            
    def update_timer(self):
        self.remaining_time -= 1
        if self.remaining_time <= 0:
            self.timer.stop()
            self.submit_exam()
        else:
            minutes = self.remaining_time // 60
            seconds = self.remaining_time % 60
            self.timer_label.setText(f"Time Remaining: {minutes:02d}:{seconds:02d}")
            
            # Change timer color to red when less than 5 minutes remain
            if self.remaining_time < 300:
                self.timer_label.setStyleSheet("""
                    padding: 10px; 
                    border-radius: 8px; 
                    background-color: #fff0f0;
                    border: 1px solid #ffcccc;
                    color: #d32f2f;
                    font-weight: bold;
                """)
                
                # Flash effect when less than 1 minute remains
                if self.remaining_time < 60 and self.remaining_time % 2 == 0:
                    self.timer_label.setStyleSheet("""
                        padding: 10px; 
                        border-radius: 8px; 
                        background-color: #d32f2f;
                        border: 1px solid #b71c1c;
                        color: white;
                        font-weight: bold;
                    """)
                
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
            
    def submit_exam(self):
        # In a real application, you would save answers and show results
        print("Exam submitted!")
        
        # Create a stylish submission confirmation dialog
        submission_dialog = QMainWindow()
        submission_dialog.setWindowTitle("Exam Submitted")
        submission_dialog.setFixedSize(500, 300)
        
        central_widget = QWidget()
        submission_dialog.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Success icon or image could be added here
        title = QLabel("Success!")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #03DAC5;")
        layout.addWidget(title)
        
        message = QLabel("Your exam has been submitted successfully!")
        message.setFont(QFont("Arial", 16))
        message.setAlignment(Qt.AlignCenter)
        message.setWordWrap(True)
        layout.addWidget(message)
        
        details = QLabel("You will receive your results by email once grading is complete.")
        details.setFont(QFont("Arial", 12))
        details.setAlignment(Qt.AlignCenter)
        details.setWordWrap(True)
        details.setStyleSheet("color: #606060;")
        layout.addWidget(details)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setFont(QFont("Arial", 14))
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #6200EE;
                color: white;
                border-radius: 5px;
                padding: 10px 40px;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #7722ff;
            }
        """)
        close_btn.clicked.connect(submission_dialog.close)
        
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        
        layout.addWidget(btn_container)
        layout.addStretch()
        
        # Set dialog stylesheet
        submission_dialog.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
        """)
        
        submission_dialog.show()
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style 
    app.setStyle("Fusion")
    
    # Create and show the main window
    window = ProctorExamApp()
    window.show()
    
    sys.exit(app.exec_())
