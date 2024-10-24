import _thread
import argparse
import asyncio
import functools
import json
import os
import queue
import time
import tkinter.font
import tkinter.messagebox
import wave
from tkinter import *
from tkinter.filedialog import askopenfilename
import yaml
import re

import numpy as np
import pyaudio
import soundcard
import requests
import soundfile
import websockets

from masr.predict import MASRPredictor
from masr.utils.logger import setup_logger
from masr.utils.utils import add_arguments, print_arguments

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/conformer.yml',       "配置文件")
add_arg('use_server',       bool,   False,         "是否使用服务器服务进行识别，否则使用本地识别")
add_arg("host",             str,    "127.0.0.1",   "服务器IP地址")
add_arg("port_server",      int,    5000,          "普通识别服务端口号")
add_arg("port_stream",      int,    5001,          "流式识别服务端口号")
add_arg('use_gpu',          bool,   False,         "是否使用GPU预测")
add_arg('use_pun',          bool,   False,         "是否给识别结果加标点符号")
add_arg('model_path',       str,    'models/inference.pt',       "导出的预测模型文件路径")
add_arg('pun_model_dir',    str,    'models/pun_models/',        "加标点符号的模型文件夹路径")
args = parser.parse_args()
print_arguments(args=args)

with open(args.configs, 'r', encoding='utf-8') as f:
    args.configs = yaml.load(f.read(), Loader=yaml.FullLoader)
# 数据字典的路径
args.configs['dataset_conf']['dataset_vocab'] = 'models/vocabulary.txt'
# 结果解码方法，支持：ctc_beam_search、ctc_greedy
args.configs['decoder'] = 'ctc_greedy'
# ctc_beam_search 解码器的语言模型文件路径
args.configs['ctc_beam_search_decoder_conf']['language_model_path'] = 'models/lm.klm'
print_arguments(configs=args.configs)

tones_raw = '0123456'
tones_to_display = {
    '数字': '⁰¹²³⁴⁵⁶',
    '符号': ['', '˩', '˨', '˧', '˦', '˥', '˥'],
    '隐藏': [''] * 7,
}


class SpeechRecognitionApp:
    def format_result(self, text):
        text = ''.join(text)
        text = text.replace('.', ' ')
        text = text.replace('¦', ', ')
        text = re.sub(r'([0-9]+) *', r'\1 ', text)
        text = re.sub(r' +,', r',', text)
        tone_style = self.tone_style_selected.get()
        trans_dict = dict(zip(tones_raw, tones_to_display[tone_style]))
        text = text.translate(str.maketrans(trans_dict))
        if tone_style == '符号':
            text = re.sub(r'([˩˨˧˦˥])\1+', r'\1', text)
        text = text.strip(', ')
        text = text.strip()
        return text

    def __init__(self, window: Tk, args):
        self.window = window
        self.wav_path = None
        self.predicting = False
        self.playing = False
        self.recording = False
        self.stream = None
        self.is_itn = False
        self.use_server = args.use_server
        # 录音参数
        self.frames = []
        self.data_queue = queue.Queue()
        self.sample_rate = 16000
        interval_time = 0.5
        self.block_size = int(self.sample_rate * interval_time)
        # 最大录音时长
        self.max_record = 600
        # 录音保存的路径
        self.output_path = 'recordings/'
        # 调整默认字号
        tkinter.font.nametofont("TkDefaultFont").configure(size=12)
        # 指定窗口标题
        self.window.title("国际音标识别")
        # 固定窗口大小
        self.window.geometry('1200x600')
        self.window.resizable(False, False)
        # 识别短语音按钮
        self.short_button = Button(self.window, text="打开音频", width=14, command=self.predict_audio_thread)
        self.short_button.place(x=170, y=10)
        # 识别长语音按钮
        # self.long_button = Button(self.window, text="打开长音频", width=14, command=self.predict_long_audio_thread)
        # self.long_button.place(x=490, y=10)
        # 录音按钮
        self.record_button = Button(self.window, text="录制音频", width=14, command=self.record_audio_thread)
        self.record_button.place(x=10, y=10)
        # 播放音频按钮
        self.play_button = Button(self.window, text="播放音频", width=14, command=self.play_audio_thread)
        self.play_button.place(x=330, y=10)
        # 选择声调风格
        Label(self.window, text="声调标记：").place(x=520, y=14)
        self.tone_style_selected = StringVar(None, '数字')
        for i, tone_style in enumerate(tones_to_display.keys()):
            r = Radiobutton(self.window, text=tone_style, value=tone_style, variable=self.tone_style_selected)
            r.place(x=610 + i * 70, y=12)
        # 输出结果文本框
        self.result_label = Label(self.window, text="输出：")
        self.result_label.place(x=10, y=60)
        self.result_text = Text(self.window, font=('Doulos SIL', 20), wrap=WORD, padx=20, pady=15)
        self.result_text.place(x=10, y=90, width=1180, height=500)
        self.result_text.configure(state='disabled')
        # 对文本进行反标准化
        # self.an_frame = Frame(self.window)
        # self.check_var = BooleanVar(value=False)
        # self.is_itn_check = Checkbutton(self.an_frame, text='是否对文本进行反标准化', variable=self.check_var, command=self.is_itn_state)
        # self.is_itn_check.grid(row=0)
        # self.an_frame.grid(row=1)
        # self.an_frame.place(x=700, y=10)

        if not self.use_server:
            # 获取识别器
            self.predictor = MASRPredictor(configs=args.configs,
                                           model_path=args.model_path,
                                           use_gpu=args.use_gpu,
                                           use_pun=args.use_pun,
                                           pun_model_dir=args.pun_model_dir)

    # 是否对文本进行反标准化
    def is_itn_state(self):
        self.is_itn = self.check_var.get()

    # 预测短语音线程
    def predict_audio_thread(self):
        if not self.predicting:
            self.wav_path = askopenfilename(filetypes=[("音频文件", "*.wav"), ("音频文件", "*.mp3")], initialdir=self.output_path)
            if self.wav_path == '': return
            self.result_text.configure(state='normal')
            self.result_text.delete('1.0', 'end')
            self.result_text.insert(END, f"已选择音频文件：{os.path.basename(self.wav_path)}\n\n")
            self.result_text.insert(END, "【音频文件识别】\n")
            self.result_text.insert(END, "正在识别……\n")
            self.result_text.configure(state='disabled')
            _thread.start_new_thread(self.predict_audio, (self.wav_path, ))
        else:
            tkinter.messagebox.showwarning('警告', '正在预测，请等待上一轮预测结束！')

    # 预测长语音线程
    def predict_long_audio_thread(self):
        if not self.predicting:
            self.wav_path = askopenfilename(filetypes=[("音频文件", "*.wav"), ("音频文件", "*.mp3")], initialdir=self.output_path)
            if self.wav_path == '': return
            self.result_text.configure(state='normal')
            self.result_text.delete('1.0', 'end')
            self.result_text.insert(END, f"已选择音频文件：{os.path.basename(self.wav_path)}\n\n")
            self.result_text.insert(END, "【音频文件识别】\n")
            self.result_text.insert(END, "正在识别……\n")
            self.result_text.configure(state='disabled')
            _thread.start_new_thread(self.predict_long_audio, (self.wav_path, ))
        else:
            tkinter.messagebox.showwarning('警告', '正在预测，请等待上一轮预测结束！')

    # 录音识别线程
    def record_audio_thread(self):
        if not self.playing and not self.recording:
            self.recording = True
            if not self.use_server:
                _thread.start_new_thread(self.record_audio, ())
            _thread.start_new_thread(self.predict_stream, ())
        else:
            if self.playing:
                tkinter.messagebox.showwarning('警告', '正在录音，无法播放音频！')
            else:
                # 停止录音
                self.recording = False

    # 播放音频线程
    def play_audio_thread(self):
        if self.wav_path is None or self.wav_path == '':
            tkinter.messagebox.showwarning('警告', '音频路径为空！')
        else:
            if not self.playing and not self.recording:
                _thread.start_new_thread(self.play_audio, ())
            else:
                if self.recording:
                    tkinter.messagebox.showwarning('警告', '正在录音，无法播放音频！')
                else:
                    # 停止播放
                    self.playing = False

    def record_audio(self):
        self.frames = []
        self.record_button.configure(text='停止录音')
        self.result_text.configure(state='normal')
        self.result_text.delete('1.0', 'end')
        self.result_text.insert(END, "正在录音\n")
        self.result_text.configure(state='disabled')
        # 打开默认的输入设备
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.block_size)
        while True:
            # 开始录制并获取数据
            data = stream.read(self.block_size)
            data = np.frombuffer(data, dtype=np.int16)
            self.frames.append(data)
            self.data_queue.put(data)
            if not self.recording: break
        stream.close()
        p.terminate()
        self.recording = False

    # 播放音频
    def play_audio(self):
        self.play_button.configure(text='停止播放')
        self.playing = True
        default_speaker = soundcard.default_speaker()
        data, sr = soundfile.read(self.wav_path)
        with default_speaker.player(samplerate=sr) as player:
            for i in range(0, data.shape[0], sr):
                if not self.playing: break
                d = data[i:i + sr]
                player.play(d)
        self.playing = False
        self.play_button.configure(text='播放音频')

    def predict_stream(self):
        if not self.use_server:
            # 本地识别
            while self.recording:
                try:
                    data = self.data_queue.get(timeout=1)
                except queue.Empty:
                    continue
                # 在这里处理数据
                result = self.predictor.predict_stream(audio_data=data, use_pun=args.use_pun, is_itn=self.is_itn,
                                                       is_end=not self.recording, sample_rate=self.sample_rate)
                if result is None: continue
                score, text = result['score'], self.format_result(result['text'])
                self.result_text.configure(state='normal')
                self.result_text.delete('1.0', 'end')
                self.result_text.insert(END, "【实时识别】\n")
                self.result_text.insert(END, f"结　果：{text}\n")
                self.result_text.insert(END, f"可靠度：{round(score, 2)}%\n")
                self.result_text.configure(state='disabled')
            self.predictor.reset_stream()
            # 拼接录音数据
            data = np.concatenate(self.frames)
            # 保存音频数据
            os.makedirs(self.output_path, exist_ok=True)
            self.wav_path = os.path.join(self.output_path, '%s.wav' % time.strftime('%Y%m%d-%H%M%S'))
            soundfile.write(self.wav_path, data=data, samplerate=self.sample_rate)
            self.record_button.configure(text='录制音频')

            if not self.predicting:
                _thread.start_new_thread(self.predict_audio, (self.wav_path, True))
            else:
                tkinter.messagebox.showwarning('警告', '正在预测，请等待上一轮预测结束！')

        else:
            # 调用服务接口
            new_loop = asyncio.new_event_loop()
            new_loop.run_until_complete(self.run_websocket())

    # 预测短语音
    def predict_audio(self, wav_file, is_just_recorded=False):
        self.predicting = True
        try:
            start = time.time()
            # 判断使用本地识别还是调用服务接口
            if not self.use_server:
                result = self.predictor.predict(audio_data=wav_file, use_pun=args.use_pun, is_itn=self.is_itn)
                score, text = result['score'], self.format_result(result['text'])
            else:
                # 调用用服务接口识别
                url = f"http://{args.host}:{args.port_server}/recognition"
                files = [('audio', ('test.wav', open(wav_file, 'rb'), 'audio/wav'))]
                headers = {'accept': 'application/json'}
                response = requests.post(url, headers=headers, files=files)
                data = json.loads(response.text)
                if data['code'] != 0:
                    raise Exception(f'服务请求失败，错误信息：{data["msg"]}')
                text, score = data['result'], data['score']
            self.result_text.configure(state='normal')
            if is_just_recorded:
                self.result_text.insert(END, "\n")
                self.result_text.insert(END, "【整体识别】\n")
            else:
                txt = self.result_text.get('1.0', 'end')
                idx = txt.find('正在识别……')
                self.result_text.delete(f'1.0', 'end')
                self.result_text.insert(END, txt[:idx])
            self.result_text.insert(END, f"结　果：{text}\n")
            self.result_text.insert(END, f"可靠度：{round(score, 2)}%\n")
            self.result_text.insert(END, f"耗　时：{int(round((time.time() - start) * 1000))} ms\n")
            self.result_text.configure(state='disabled')
        except Exception as e:
            self.result_text.configure(state='normal')
            self.result_text.insert(END, str(e))
            self.result_text.configure(state='disabled')
            raise e
        self.predicting = False

    # 预测长语音
    def predict_long_audio(self, wav_path):
        self.predicting = True
        try:
            start = time.time()
            # 判断使用本地识别还是调用服务接口
            if not self.use_server:
                result = self.predictor.predict_long(audio_data=wav_path, use_pun=args.use_pun, is_itn=self.is_itn)
                score, text = result['score'], self.format_result(result['text'])
            else:
                # 调用用服务接口识别
                url = f"http://{args.host}:{args.port_server}/recognition_long_audio"
                files = [('audio', ('test.wav', open(wav_path, 'rb'), 'audio/wav'))]
                headers = {'accept': 'application/json'}
                response = requests.post(url, headers=headers, files=files)
                data = json.loads(response.text)
                if data['code'] != 0:
                    raise Exception(f'服务请求失败，错误信息：{data["msg"]}')
                text, score = data['result'], data['score']
            self.result_text.configure(state='normal')
            txt = self.result_text.get('1.0', 'end')
            idx = txt.find('正在识别……')
            self.result_text.delete(f'1.0', 'end')
            self.result_text.insert(END, txt[:idx])
            self.result_text.insert(END, f"结　果：{text}\n")
            self.result_text.insert(END, f"可靠度：{round(score, 2)}%\n")
            self.result_text.insert(END, f"耗　时：{int(round((time.time() - start) * 1000))} ms\n")
            self.result_text.configure(state='disabled')
        except Exception as e:
            self.result_text.configure(state='normal')
            self.result_text.insert(END, str(e))
            self.result_text.configure(state='disabled')
            raise e
        self.predicting = False

    # 使用WebSocket调用实时语音识别服务
    async def run_websocket(self):
        self.frames = []
        self.record_button.configure(text='停止录音')
        self.result_text.insert(END, "正在录音...\n")
        # 创建一个播放器
        p = pyaudio.PyAudio()
        # 打开录音
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.block_size)
        async with websockets.connect(f"ws://{args.host}:{args.port_stream}") as websocket:
            while not websocket.closed:
                data = stream.read(self.block_size)
                self.frames.append(data)
                send_data = data
                # 用户点击停止录音按钮
                if not self.recording:
                    send_data += b'end'
                await websocket.send(send_data)
                result = await websocket.recv()
                self.result_text.delete('1.0', 'end')
                self.result_text.insert(END, f"{json.loads(result)['result']}\n")
                # 停止录音后，需要把"end"发给服务器才最终停止
                if not self.recording and b'end' == send_data[-3:]:
                    logger.info('识别结束')
                    break
            # await websocket.close()
        stream.close()
        # 录音的字节数据，用于后面的预测和保存
        audio_bytes = b''.join(self.frames)
        # 保存音频数据
        os.makedirs(self.output_path, exist_ok=True)
        self.wav_path = os.path.join(self.output_path, '%s.wav' % str(int(time.time())))
        wf = wave.open(self.wav_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(audio_bytes)
        wf.close()
        self.recording = False
        self.result_text.insert(END, "录音已结束，录音文件保存在：%s\n" % self.wav_path)
        self.record_button.configure(text='录音识别')
        logger.info('close websocket')


tk = Tk()
myapp = SpeechRecognitionApp(tk, args)

if __name__ == '__main__':
    tk.mainloop()
