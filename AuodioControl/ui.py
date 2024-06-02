import os
import shutil
import time
from datetime import datetime

import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
import sounddevice as sd
from PyQt5.QtGui import QIcon
from soundfile import write
import threading
import sys
from speech_recognition.my_whisper import MyWhisper
from text_match.sentence_similarity_model import SentenceSimilarityModel
from text_match.text_matcher import TextMatcher


class LoginWindow(QtWidgets.QDialog):
      def __init__(self):
            super().__init__()
            self.setWindowTitle("Login")
            self.resize(300, 150)
            self.setWindowIcon(QIcon("ico.png"))
            self.username_label = QtWidgets.QLabel("Username:")
            self.username_edit = QtWidgets.QLineEdit()

            self.password_label = QtWidgets.QLabel("Password:")
            self.password_edit = QtWidgets.QLineEdit()
            self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)

            self.login_button = QtWidgets.QPushButton("Login")
            self.login_button.clicked.connect(self.login)

            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(self.username_label)
            layout.addWidget(self.username_edit)
            layout.addWidget(self.password_label)
            layout.addWidget(self.password_edit)
            layout.addWidget(self.login_button)

            self.setLayout(layout)

      def login(self):
            username = self.username_edit.text()
            password = self.password_edit.text()
            # 连接到MySQL数据库
            # try:
            #       connection = mysql.connector.connect(host='localhost:8080', database='audio_control',
            #                                            user='root',
            #                                            password='123456')
            #       if connection.is_connected():
            #             cursor = connection.cursor()
            #             # 编写SQL查询语句
            #             query = "SELECT * FROM users WHERE username = %s AND password = %s"
            #             # 执行查询
            #             cursor.execute(query, (username, password))
            #             # 获取查询结果
            #             result = cursor.fetchone()
            #             # 如果查询到结果，则用户名和密码匹配
            #             if result:
            #                   self.accept()
            #             else:
            #                   QtWidgets.QMessageBox.warning(self, "Login Failed", "Invalid username or password")
            # except Error as e:
            #       print("Error while connecting to MySQL", e)
            # finally:
            #       # 关闭游标和连接
            #       if connection.is_connected():
            #             cursor.close()
            #             connection.close()
            if username == "1" and password == "1":
                  self.accept()
            else:
                  QtWidgets.QMessageBox.warning(self, "Login Failed", "Invalid username or password")


class Ui_mainWindow(object):
      def __init__(self, device):
            self.is_recording = False
            self.samplerate = 16000  # 采样率
            self.channels = 1  # 声道数
            self.max_record_time = 10  # 最大录音时长，单位为秒
            self.audio_data = []
            self.immediate_execute = False  # 是否识别后立即执行
            # 中文预训练的Whisper模型
            self.speech_recognizer = MyWhisper(device, model_path='./speech_recognition/model/whisper-base')
            # 基于bert的命令匹配模型
            self.command_matcher = TextMatcher(device, model_path='./text_match/model/bert_base',
                                               command_path='command.json')
            # self.command_matcher = SentenceSimilarityModel(device, command_path='command.json')
            # self.command_matcher.load_state_dict(torch.load('./text_match/model/model.pth'))
            # self.command_matcher.eval()
            self.recognize_result = {}

      # noinspection PyAttributeOutsideInit
      def setupUi(self, mainWindow):
            mainWindow.setObjectName("mainWindow")
            mainWindow.setWindowIcon(QIcon("ico.png"))
            mainWindow.resize(567, 330)
            self.centralwidget = QtWidgets.QWidget(mainWindow)
            self.centralwidget.setObjectName("centralwidget")
            self.pushButton = QtWidgets.QPushButton(self.centralwidget)  # 长按录音按钮
            self.pushButton.setGeometry(QtCore.QRect(420, 20, 101, 41))
            self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)  # 执行命令按钮
            self.pushButton_2.setGeometry(QtCore.QRect(420, 90, 101, 41))
            self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)  # 上传文件按钮
            self.pushButton_3.setGeometry(QtCore.QRect(420, 160, 101, 41))
            self.saveButton = QtWidgets.QPushButton(self.centralwidget)  # 保存录音按钮
            self.saveButton.setGeometry(QtCore.QRect(420, 230, 101, 41))
            self.textEdit = QtWidgets.QTextEdit(self.centralwidget)  # 识别结果展示文本框
            self.textEdit.setGeometry(QtCore.QRect(30, 20, 361, 260))
            mainWindow.setCentralWidget(self.centralwidget)
            self.menubar = QtWidgets.QMenuBar(mainWindow)  # 菜单栏
            self.menubar.setGeometry(QtCore.QRect(0, 0, 567, 26))
            self.menuOptions = QtWidgets.QMenu(self.menubar)
            mainWindow.setMenuBar(self.menubar)
            self.menubar.addAction(self.menuOptions.menuAction())
            self.actionImmediateExecute = QtWidgets.QAction(mainWindow)
            self.actionImmediateExecute.setCheckable(True)
            self.menuOptions.addAction(self.actionImmediateExecute)
            # 为组件绑定方法
            self.pushButton.pressed.connect(self.start_recording)  #
            self.pushButton.released.connect(self.stop_recording)
            self.pushButton_2.clicked.connect(self.manual_execute)
            self.pushButton_3.clicked.connect(self.upload_file)
            self.saveButton.clicked.connect(self.save_audio)
            self.actionImmediateExecute.triggered.connect(self.set_immediate_execute)

            self.retranslateUi(mainWindow)
            QtCore.QMetaObject.connectSlotsByName(mainWindow)

      def retranslateUi(self, mainWindow):
            _translate = QtCore.QCoreApplication.translate
            mainWindow.setWindowTitle("智能语音控制系统_V1.0")
            self.pushButton.setText("长按录音")
            self.pushButton_2.setText("执行命令")
            self.pushButton_3.setText("上传文件")
            self.saveButton.setText("保存录音")
            self.textEdit.setPlaceholderText("确认识别结果无误后，点击执行按钮即可执行相应操作")
            self.menuOptions.setTitle("选项")
            self.actionImmediateExecute.setText("识别后立即执行")

      def start_recording(self):
            self.textEdit.clear()
            self.is_recording = True
            self.audio_data = []  # 清空本地录音缓存
            self.recognize_result = None
            threading.Thread(target=self.record_audio).start()
            self.textEdit.setPlaceholderText("请讲话...")

      def stop_recording(self):
            self.textEdit.clear()
            self.textEdit.setPlaceholderText("识别中，请稍后...")
            self.is_recording = False

      def save_audio(self):
            # 获取当前时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            destination_dir = './data/user_save/'
            os.makedirs(destination_dir, exist_ok=True)
            new_file_name = f'{timestamp}.wav'
            # 完整的目标文件路径
            destination_file = os.path.join(destination_dir, new_file_name)
            source_file = 'recorded_audio.wav'
            # 检查源文件是否存在
            if not os.path.exists(source_file):
                  # 如果文件不存在，在文本编辑框中显示消息
                  self.textEdit.clear()
                  self.textEdit.append("录音文件不存在，请检查!")
                  return  # 退出函数，因为源文件不存在，无法继续执行
            # 复制并重命名文件
            shutil.copy(source_file, destination_file)
            # 可以在这里添加代码来处理复制后的逻辑，例如通知用户
            self.textEdit.append(f"文件已保存为：{destination_file[:-1]}")

      def manual_execute(self):
            text = self.textEdit.toPlainText()
            matched_command_describe, matched_command = self.command_matcher.match(text)
            self.execute_command(matched_command)
            self.textEdit.clear()
            self.textEdit.append(f"执行操作 {matched_command_describe}")
            self.save_train_data(text)

      def save_train_data(self, text):
            # 源文件路径
            source_file_path = 'recorded_audio.wav'
            if not os.path.exists(source_file_path):
                  return
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            destination_file_dir = "./data/train_data/"
            os.makedirs(destination_file_dir, exist_ok=True)
            # 完整的目标文件路径
            destination_file_path = os.path.join(destination_file_dir, f'{timestamp}.wav')
            # 确保目标路径的目录存在
            os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
            # 复制文件并使用时间戳重命名
            shutil.copy(source_file_path, destination_file_path)
            with open('./data/train_data/record.txt', 'a', encoding='utf-8') as f:
                  f.write(destination_file_path + '\t' + text)
                  f.write('\n')

      def set_immediate_execute(self, checked):
            self.immediate_execute = checked

      def execute_command(self, command):
            os.system(command)

      def record_audio(self):
            def callback(indata, frames, time, status):
                  if self.is_recording:
                        self.audio_data.append(indata.copy())

            with sd.InputStream(samplerate=self.samplerate, device=None,
                                channels=self.channels, callback=callback):
                  while self.is_recording:  # 检查当前录音状态
                        pass
            audio = np.concatenate(self.audio_data, axis=0)
            write('recorded_audio.wav', audio, self.samplerate)
            thread = threading.Thread(target=self.recognize_speech)  # 定义语音识别子线程
            thread.start()  # 调用语音识别方法
            thread.join()  # 等待识别结束
            self.textEdit.append(self.recognize_result['text'])
            print(self.recognize_result['text'])

            if self.immediate_execute:  # 如果设置了立即执行，则执行命令
                  matched_command = self.recognize_result
                  self.execute_command(matched_command)

      def recognize_speech(self, file_path='recorded_audio.wav'):
            text = self.speech_recognizer.recognize_audio(self.speech_recognizer.read_wav(file_path))
            matched_command_describe, matched_command = self.command_matcher.match(text)
            if self.immediate_execute:
                  self.execute_command(matched_command)
                  self.recognize_result = {
                        "text": text[0],
                        "matched_command_describe": matched_command_describe,
                        "matched_command": matched_command
                  }
            else:
                  self.recognize_result = {
                        "text": text[0],
                        "matched_command_describe": None,
                        "matched_command": None
                  }

      def upload_file(self):
            self.textEdit.clear()
            options = QtWidgets.QFileDialog.Options()
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                  None, "选择文件", "", "All Files (*);;WAV Files (*.wav)", options=options
            )
            if file_path:
                  # 检查文件扩展名是否是 .wav
                  _, file_extension = os.path.splitext(file_path)
                  if file_extension.lower() == '.wav':
                        # 启动线程处理语音识别
                        thread = threading.Thread(target=self.recognize_speech, args=(file_path,))
                        thread.start()
                        # 等待线程执行完毕
                        thread.join()
                        # 显示识别结果
                        self.textEdit.append(self.recognize_result['text'])
                  else:
                        # 如果文件不是 .wav 格式，显示错误消息
                        self.textEdit.append("不支持的文件格式，请选择 WAV 文件。")


if __name__ == "__main__":
      app = QtWidgets.QApplication(sys.argv)
      login_window = LoginWindow()
      if login_window.exec_() == QtWidgets.QDialog.Accepted:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            # device = "cpu"
            mainWindow = QtWidgets.QMainWindow()
            ui = Ui_mainWindow(device)
            ui.setupUi(mainWindow)
            mainWindow.show()
            sys.exit(app.exec_())
