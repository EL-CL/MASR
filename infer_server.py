import argparse
import functools
import os
import random
import sys
import time
import wave
from datetime import datetime
import yaml
import re

import aiofiles
import uvicorn
from fastapi import FastAPI, WebSocket, UploadFile, File, Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.websockets import WebSocketState

from masr.predict import MASRPredictor
from masr.utils.logger import setup_logger
from masr.utils.utils import add_arguments, print_arguments

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/conformer.yml', "配置文件")
add_arg("host",             str,    '0.0.0.0',            "监听主机的IP地址")
add_arg("port",             int,    5000,                 "服务所使用的端口号")
add_arg("save_path",        str,    'recordings/uploaded/', "上传音频文件的保存目录")
add_arg('use_gpu',          bool,   False,  "是否使用GPU预测")
add_arg('use_pun',          bool,   False,  "是否给识别结果加标点符号")
add_arg('is_itn',           bool,   False,  "是否对文本进行反标准化")
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

numbers = '0123456789'
superscripts = '⁰¹²³⁴⁵⁶⁷⁸⁹'
number2superscript = str.maketrans(numbers, superscripts)


def format_result(text):
    text = ''.join(text)
    text = text.replace('.', ' ')
    text = text.replace('¦', ', ')
    text = re.sub(r'([0-9]+) *', r'\1 ', text)
    text = re.sub(r' +,', r',', text)
    text = text.translate(number2superscript)
    text = text.strip(', ')
    text = text.strip()
    return text


app = FastAPI(title="国际音标识别")
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory="templates")

# 创建预测器
predictor = MASRPredictor(configs=args.configs,
                          model_path=args.model_path,
                          use_gpu=args.use_gpu,
                          use_pun=args.use_pun,
                          pun_model_dir=args.pun_model_dir)


# 语音识别接口
@app.post("/recognition")
async def recognition(audio: UploadFile = File(..., description="音频文件")):
    # 保存路径
    save_dir = os.path.join(args.save_path, datetime.now().strftime('%Y-%m'))
    os.makedirs(save_dir, exist_ok=True)
    suffix = audio.filename.split('.')[-1]
    file_path = os.path.join(save_dir, f'{time.strftime("%Y%m%d-%H%M%S")}-{random.randint(100, 999)}.{suffix}')
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await audio.read()
        await out_file.write(content)
    try:
        start = time.time()
        # 执行识别
        # TODO: 读取音频看时长
        func = predictor.predict if True else predictor.predict_long
        result = func(audio_data=file_path, use_pun=args.use_pun, is_itn=args.is_itn)
        score, text = result['score'], format_result(result['text'])
        end = time.time()
        print("结　果：%s\n可靠度：%f\n耗　时：%d ms" % (text, score, round((end - start) * 1000)))
        result = {"code": 0, "msg": "success", "result": text, "score": round(score, 2)}
        return result
    except Exception as e:
        print(f'[{datetime.now()}] 语音识别失败，错误信息：{e}', file=sys.stderr)
        return {"error": 1, "msg": "音频读取失败！"}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "id": id})


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f'有WebSocket连接建立')
    if not predictor.running:
        frames = []
        score, text = 0, ""
        while True:
            try:
                data = await websocket.receive_bytes()
                frames.append(data)
                if len(data) == 0: continue
                is_end = False
                # 判断是不是结束预测
                if b'end' == data[-3:]:
                    is_end = True
                    data = data[:-3]
                # 开始预测
                result = predictor.predict_stream(audio_data=data, use_pun=args.use_pun, is_itn=args.is_itn,
                                                  is_end=is_end)
                if result is not None:
                    score, text = result['score'], format_result(result['text'])
                send_data = {"code": 0, "result": text, "score": round(score, 2)}
                logger.info(f'向客户端发生消息：{send_data}')
                await websocket.send_json(send_data)
                # 结束了要关闭当前的连接
                if is_end: await websocket.close()
            except Exception as e:
                if websocket.client_state == WebSocketState.DISCONNECTED:
                    logger.info("用户已断开连接")
                    break
                logger.error(f'识别发生错误：错误信息：{e}')
                try:
                    await websocket.send_json({"code": 2, "msg": "recognition fail!"})
                except:
                    break
        # 重置流式识别
        predictor.reset_stream()
        predictor.running = False
        # 保存录音
        save_dir = os.path.join(args.save_path, datetime.now().strftime('%Y-%m'))
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'{time.strftime("%Y%m%d-%H%M%S")}-{random.randint(100, 999)}.wav')
        audio_bytes = b''.join(frames)
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)
        wf.close()
    else:
        logger.error(f'语音识别失败，预测器不足')
        await websocket.send_json({"code": 1, "msg": "recognition fail, no resource!"})
        await websocket.close()


if __name__ == '__main__':
    # 创建保存路径
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    uvicorn.run(app, host=args.host, port=args.port)
