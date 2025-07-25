# import os
# from PyQt5.QtWidgets import (
#     QWidget, QHBoxLayout, QTextEdit, QVBoxLayout,
#     QLineEdit, QPushButton, QFileDialog, QApplication, QLabel
# )
# from PyQt5.QtCore import QSize, QThread, pyqtSignal, QTimer
# from PyQt5.QtGui import QIcon
# from PIL import Image
# import pytesseract
# import speech_recognition as sr
# import pyttsx3
# from textwrap import fill

# from configuration.Appconfig import Appconfig
# from chatbot.chatbot_thread import ESIMCopilot

# class OllamaWorker(QThread):
#     response_signal = pyqtSignal(str)

#     def __init__(self, user_text: str, image_path: str = None):
#         super().__init__()
#         self.user_text = user_text
#         self.image_path = image_path
#         self.copilot = ESIMCopilot({
#             "text_model": "qwen:1.8b",
#             "vision_model": "qwen:1.8b",
#             "text_data_dir": "src/knowledge_base",
#             "device": "cpu",
#             "screenshot_dir": "src/screenshots"
#         })

#     def run(self):
#         try:
#             response = self.copilot.process_query(self.user_text, self.image_path)
#         except Exception as e:
#             response = f"[Error] {e}"
#         self.response_signal.emit(response)

# if os.name == 'nt':
#     from frontEnd import pathmagic  # noqa:F401
#     init_path = ''
# else:
#     import pathmagic  # noqa:F401
#     init_path = '../../'

# class ChatbotGUI(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("eSim Copilot")
#         self.setFixedSize(540, 500)
#         self.chat_history = []
#         self.is_loading = False
#         self.history_file = "chat_history.txt"
#         self.tts_engine = pyttsx3.init()

#         self.setStyleSheet("""
#             QWidget {
#                 background-color: #f6f9fc;
#                 font-family: 'Segoe UI', sans-serif;
#                 font-size: 14px;
#             }
#             QTextEdit, QLineEdit {
#                 background-color: #ffffff;
#                 border: 1px solid #d1d9e6;
#                 border-radius: 6px;
#                 padding: 10px;
#             }
#             QPushButton {
#                 background-color: #dbe7f4;
#                 border-radius: 6px;
#                 padding: 6px 14px;
#                 font-weight: bold;
#             }
#             QPushButton:hover {
#                 background-color: #c3d8ed;
#             }
#             QLabel {
#                 padding-left: 4px;
#             }
#         """)

#         main_layout = QVBoxLayout(self)
#         main_layout.setContentsMargins(15, 15, 15, 15)
#         main_layout.setSpacing(10)

#         self.welcome_message = QTextEdit(self)
#         self.welcome_message.setReadOnly(True)
#         self.welcome_message.setHtml("""
#             <h3 style='text-align:center; color:#2f4f60;'>👋 Welcome to eSim Copilot</h3>
#             <p style='text-align:center;'>Ask questions about your circuit simulations, error logs, or upload screenshots.</p>
#         """)
#         self.welcome_message.setFixedHeight(70)
#         self.welcome_message.setStyleSheet("border: none; background-color: transparent;")
#         main_layout.addWidget(self.welcome_message)

#         self.chat_display = QTextEdit(self, readOnly=True)
#         self.chat_display.setPlaceholderText("Your conversation will appear here...")
#         main_layout.addWidget(self.chat_display)

#         self.typing_label = QLabel("🤖 Bot is typing", self)
#         self.typing_label.setStyleSheet("color: gray; padding: 5px;")
#         self.typing_label.hide()
#         main_layout.addWidget(self.typing_label)

#         self.ellipsis = ""
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_typing_animation)

#         input_layout = QHBoxLayout()
#         input_layout.setSpacing(8)

#         self.user_input = QLineEdit(self, placeholderText="Type your query here...")
#         self.user_input.returnPressed.connect(self.ask_ollama)
#         self.user_input.setMinimumHeight(32)
#         input_layout.addWidget(self.user_input)

#         self.clear_button = QPushButton(self, icon=QIcon(init_path + 'images/clear.png'))
#         self.clear_button.setToolTip("Clear chat")
#         self.clear_button.setIconSize(QSize(20, 20))
#         self.clear_button.setFixedSize(34, 34)
#         self.clear_button.clicked.connect(self.clear_session)
#         input_layout.addWidget(self.clear_button)

#         self.upload_button = QPushButton("📷", self)
#         self.upload_button.setToolTip("Upload image")
#         self.upload_button.setFixedSize(40, 34)
#         self.upload_button.clicked.connect(self.upload_image)
#         input_layout.addWidget(self.upload_button)

#         self.voice_button = QPushButton("🎙", self)
#         self.voice_button.setToolTip("Speak")
#         self.voice_button.setFixedSize(40, 34)
#         self.voice_button.clicked.connect(self.voice_input)
#         input_layout.addWidget(self.voice_button)

#         main_layout.addLayout(input_layout)
#         self.move_to_bottom_right()
#         self.load_chat_history()

#     def ask_ollama(self, image_path=None):
#         if self.is_loading:
#             return

#         user_text = self.user_input.text().strip()
#         if not user_text:
#             return

#         self.chat_display.append(f"<div style='margin-top:10px;'><b style='color:#2c3e50;'>You:</b> {user_text}</div>")
#         self.chat_display.repaint()

#         self.chat_history = (self.chat_history + [f"User: {user_text}"])[-4:]

#         self.worker = OllamaWorker(user_text, image_path=image_path)
#         self.worker.response_signal.connect(self.display_response)
#         self.worker.start()

#         self.set_loading_state(True)
#         self.user_input.clear()

#     def display_response(self, bot_response):
#         formatted_response = fill(bot_response, width=80)

#         html_response = f"""
#         <br>
#     <div style='margin-top:15px;'>
#         <b style='color:#16a085;'>🤖 Bot:</b><br>
#         <div style='margin-top:5px; text-align:justify; color:#2c3e50; line-height:1.5;'>
#             <pre style='white-space:pre-wrap; word-wrap:break-word; font-family:inherit;'>{formatted_response}</pre>
#         </div>
#     </div>
# """


#         cursor = self.chat_display.textCursor()
#         cursor.movePosition(cursor.End)
#         cursor.insertHtml(html_response)
#         cursor.insertBlock()

#         self.chat_history.append(f"Bot: {bot_response}")
#         self.tts_engine.say(bot_response.replace(". ", ".\n"))
#         self.tts_engine.runAndWait()

#         self.set_loading_state(False)
#         self.save_chat_history()
#         self.scroll_to_bottom()

#     def update_typing_animation(self):
#         self.ellipsis = "." if self.ellipsis == "..." else self.ellipsis + "."
#         self.typing_label.setText(f"🤖 Copilot is typing{self.ellipsis}")

#     def set_loading_state(self, state: bool):
#         self.is_loading = state
#         self.user_input.setDisabled(state)
#         self.clear_button.setDisabled(state)
#         self.upload_button.setDisabled(state)
#         self.voice_button.setDisabled(state)
#         self.typing_label.setVisible(state)

#         if state:
#             self.ellipsis = ""
#             self.timer.start(500)
#         else:
#             self.timer.stop()
#             self.typing_label.hide()

#     def scroll_to_bottom(self):
#         self.chat_display.verticalScrollBar().setValue(
#             self.chat_display.verticalScrollBar().maximum())

#     def upload_image(self):
#         file_path, _ = QFileDialog.getOpenFileName(self, "Upload Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
#         if file_path:
#             try:
#                 extracted_text = pytesseract.image_to_string(Image.open(file_path))
#                 self.chat_display.append(f"<div style='margin-top:10px; color:#7f8c8d;'><i>🖼 Extracted text from image:</i><br>{extracted_text}</div>")
#                 self.user_input.setText(extracted_text.strip())
#                 self.ask_ollama(image_path=file_path)
#             except Exception as e:
#                 self.chat_display.append(f"<b style='color:red;'>Error:</b> Could not process image - {str(e)}")

#     def voice_input(self):
#         recognizer = sr.Recognizer()
#         mic = sr.Microphone()
#         self.chat_display.append("<i style='color:gray;'>🎙 Listening...</i>")
#         QApplication.processEvents()

#         with mic as source:
#             recognizer.adjust_for_ambient_noise(source)
#             try:
#                 audio = recognizer.listen(source, timeout=5)
#                 text = recognizer.recognize_google(audio)
#                 self.chat_display.append(f"<div style='margin-top:10px;'><b style='color:#2c3e50;'>You (voice):</b> {text}</div>")
#                 self.user_input.setText(text)
#                 self.ask_ollama()
#             except sr.WaitTimeoutError:
#                 self.chat_display.append("<i style='color:red;'>⏱ No speech detected.</i>")
#             except sr.UnknownValueError:
#                 self.chat_display.append("<i style='color:red;'>❌ Could not understand audio.</i>")
#             except Exception as e:
#                 self.chat_display.append(f"<b style='color:red;'>Error:</b> {str(e)}")

#     def clear_session(self):
#         self.chat_display.clear()
#         self.chat_history = []
#         self.welcome_message.setHtml("<h3 style='text-align:center; color:#2f4f60;'>👋 Welcome back to eSim Copilot</h3>")
#         if os.path.exists(self.history_file):
#             os.remove(self.history_file)

#     def save_chat_history(self):
#         with open(self.history_file, "w", encoding="utf-8") as f:
#             f.write(self.chat_display.toHtml())

#     def load_chat_history(self):
#         if os.path.exists(self.history_file):
#             with open(self.history_file, "r", encoding="utf-8") as f:
#                 content = f.read()
#                 self.chat_display.setHtml(content)

#     def move_to_bottom_right(self):
#         screen = QApplication.desktop().screenGeometry()
#         widget = self.geometry()
#         x = screen.width() - widget.width() - 10
#         y = screen.height() - widget.height() - 50
#         self.move(x, y)





import os
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QTextEdit, QVBoxLayout,
    QLineEdit, QPushButton, QFileDialog, QApplication, QLabel
)
from PyQt5.QtCore import QSize, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon
from PIL import Image
import pytesseract
import speech_recognition as sr
import pyttsx3

from configuration.Appconfig import Appconfig
from chatbot.chatbot_thread import ESIMCopilot

class OllamaWorker(QThread):
    response_signal = pyqtSignal(str)     # Each chunk
    finished_signal = pyqtSignal()        # End of stream

    def __init__(self, user_text: str, image_path: str = None):
        super().__init__()
        self.user_text = user_text
        self.image_path = image_path
        self.copilot = ESIMCopilot({
            "text_model": "qwen:1.8b",
            "vision_model": "qwen:1.8b",
            "text_data_dir": "./src/knowledge_base",
            "device": "cpu",
            "screenshot_dir": "./src/screenshots"
        })

    def run(self):
        try:
            for chunk in self.copilot.process_query(self.user_text, self.image_path):
                self.response_signal.emit(chunk)
        except Exception as e:
            self.response_signal.emit(f"[Error] {e}")
        self.finished_signal.emit()


if os.name == 'nt':
    from frontEnd import pathmagic  # noqa:F401
    init_path = ''
else:
    import pathmagic  # noqa:F401
    init_path = '../../'


class ChatbotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("eSim Copilot")
        self.setFixedSize(540, 500)
        self.chat_history = []
        self.is_loading = False
        self.history_file = "chat_history.txt"
        self.tts_engine = pyttsx3.init()
        self.streaming_buffer = ""

        self.setStyleSheet("""
            QWidget {
                background-color: #f6f9fc;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
            QTextEdit, QLineEdit {
                background-color: #ffffff;
                border: 1px solid #d1d9e6;
                border-radius: 6px;
                padding: 10px;
            }
            QPushButton {
                background-color: #dbe7f4;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c3d8ed;
            }
            QLabel {
                padding-left: 4px;
            }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        self.welcome_message = QTextEdit(self)
        self.welcome_message.setReadOnly(True)
        self.welcome_message.setHtml("""
            <h3 style='text-align:center; color:#2f4f60;'>👋 Welcome to eSim Copilot</h3>
            <p style='text-align:center;'>Ask questions about your circuit simulations, error logs, or upload screenshots.</p>
        """)
        self.welcome_message.setFixedHeight(70)
        self.welcome_message.setStyleSheet("border: none; background-color: transparent;")
        main_layout.addWidget(self.welcome_message)

        self.chat_display = QTextEdit(self, readOnly=True)
        self.chat_display.setPlaceholderText("Your conversation will appear here...")
        main_layout.addWidget(self.chat_display)

        self.typing_label = QLabel("🤖 Bot is typing", self)
        self.typing_label.setStyleSheet("color: gray; padding: 5px;")
        self.typing_label.hide()
        main_layout.addWidget(self.typing_label)

        self.ellipsis = ""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_typing_animation)

        input_layout = QHBoxLayout()
        input_layout.setSpacing(8)

        self.user_input = QLineEdit(self, placeholderText="Type your query here...")
        self.user_input.returnPressed.connect(self.ask_ollama)
        self.user_input.setMinimumHeight(32)
        input_layout.addWidget(self.user_input)

        self.clear_button = QPushButton(self, icon=QIcon(init_path + 'images/clear.png'))
        self.clear_button.setToolTip("Clear chat")
        self.clear_button.setIconSize(QSize(20, 20))
        self.clear_button.setFixedSize(34, 34)
        self.clear_button.clicked.connect(self.clear_session)
        input_layout.addWidget(self.clear_button)

        self.upload_button = QPushButton("📷", self)
        self.upload_button.setToolTip("Upload image")
        self.upload_button.setFixedSize(40, 34)
        self.upload_button.clicked.connect(self.upload_image)
        input_layout.addWidget(self.upload_button)

        self.voice_button = QPushButton("🎙", self)
        self.voice_button.setToolTip("Speak")
        self.voice_button.setFixedSize(40, 34)
        self.voice_button.clicked.connect(self.voice_input)
        input_layout.addWidget(self.voice_button)

        main_layout.addLayout(input_layout)
        self.move_to_bottom_right()
        self.load_chat_history()

    def ask_ollama(self, image_path=None):
        if self.is_loading:
            return

        user_text = self.user_input.text().strip()
        if not user_text:
            return

        self.chat_display.append(f"<div style='margin-top:10px;'><b style='color:#2c3e50;'>You:</b> {user_text}</div>")
        self.chat_display.append("<b style='color:#16a085;'>🤖 Bot:</b><br>")
        self.chat_display.repaint()

        self.chat_history = (self.chat_history + [f"User: {user_text}"])[-4:]
        self.streaming_buffer = ""

        self.worker = OllamaWorker(user_text, image_path=image_path)
        self.worker.response_signal.connect(self.stream_response)
        self.worker.finished_signal.connect(self.finalize_response)
        self.worker.start()

        self.set_loading_state(True)
        self.user_input.clear()

    def stream_response(self, chunk: str):
        self.streaming_buffer += chunk
        self.chat_display.moveCursor(self.chat_display.textCursor().End)
        self.chat_display.insertPlainText(chunk)  # ✅ Corrected line
        QApplication.processEvents()
        self.scroll_to_bottom()


    def finalize_response(self):
        self.tts_engine.say(self.streaming_buffer.replace(". ", ".\n"))
        self.tts_engine.runAndWait()
        self.chat_history.append(f"Bot: {self.streaming_buffer}")
        self.set_loading_state(False)
        self.save_chat_history()

    def update_typing_animation(self):
        self.ellipsis = "." if self.ellipsis == "..." else self.ellipsis + "."
        self.typing_label.setText(f"🤖 Copilot is typing{self.ellipsis}")

    def set_loading_state(self, state: bool):
        self.is_loading = state
        self.user_input.setDisabled(state)
        self.clear_button.setDisabled(state)
        self.upload_button.setDisabled(state)
        self.voice_button.setDisabled(state)
        self.typing_label.setVisible(state)

        if state:
            self.ellipsis = ""
            self.timer.start(500)
        else:
            self.timer.stop()
            self.typing_label.hide()

    def scroll_to_bottom(self):
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum())

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            try:
                extracted_text = pytesseract.image_to_string(Image.open(file_path))
                self.chat_display.append(f"<div style='margin-top:10px; color:#7f8c8d;'><i>🖼 Extracted text from image:</i><br>{extracted_text}</div>")
                self.user_input.setText(extracted_text.strip())
                self.ask_ollama(image_path=file_path)
            except Exception as e:
                self.chat_display.append(f"<b style='color:red;'>Error:</b> Could not process image - {str(e)}")

    def voice_input(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        self.chat_display.append("<i style='color:gray;'>🎙 Listening...</i>")
        QApplication.processEvents()

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                self.chat_display.append(f"<div style='margin-top:10px;'><b style='color:#2c3e50;'>You (voice):</b> {text}</div>")
                self.user_input.setText(text)
                self.ask_ollama()
            except sr.WaitTimeoutError:
                self.chat_display.append("<i style='color:red;'>⏱ No speech detected.</i>")
            except sr.UnknownValueError:
                self.chat_display.append("<i style='color:red;'>❌ Could not understand audio.</i>")
            except Exception as e:
                self.chat_display.append(f"<b style='color:red;'>Error:</b> {str(e)}")

    def clear_session(self):
        self.chat_display.clear()
        self.chat_history = []
        self.welcome_message.setHtml("<h3 style='text-align:center; color:#2f4f60;'>👋 Welcome back to eSim Copilot</h3>")
        if os.path.exists(self.history_file):
            os.remove(self.history_file)

    def save_chat_history(self):
        with open(self.history_file, "w", encoding="utf-8") as f:
            f.write(self.chat_display.toHtml())

    def load_chat_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, "r", encoding="utf-8") as f:
                content = f.read()
                self.chat_display.setHtml(content)

    def move_to_bottom_right(self):
        screen = QApplication.desktop().screenGeometry()
        widget = self.geometry()
        x = screen.width() - widget.width() - 10
        y = screen.height() - widget.height() - 50
        self.move(x, y)
