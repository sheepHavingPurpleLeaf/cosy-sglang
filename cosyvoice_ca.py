#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import logging
from typing import List, Dict, Optional, Generator
from pathlib import Path
import threading
from contextlib import nullcontext
import os
import sys
sys.path.append('third_party/Matcha-TTS')
# 添加yaml配置文件加载
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.utils.common import fade_in_out
# 导入所需模型类
import tensorrt as trt
# 导入请求库
import requests
import json
import time
import pdb
import torchaudio
import torch.nn.functional as F
from cosyvoice.utils.mask import make_pad_mask
from collections import deque
import queue

class CosyVoiceCA:
    """
    一个简化版的CosyVoice2类，仅加载Flow和HiFiGAN模型，
    只执行token2wav的流式逻辑，不执行LLM部分
    """
    
    def __init__(self, model_dir):
        """
        初始化CosyVoiceCA类，使用固定配置：
        load_jit=False, load_trt=True, fp16=True, use_flow_cache=True
        
        Args:
            model_dir: 模型目录，包含flow和hift模型
            device: 指定使用的设备，默认为None（自动检测）
        """
        self.device = torch.device('cuda')
        self.model_dir = Path(model_dir)
        
        # 固定配置
        self.fp16 = True
        self.is_synthesizing = False
        
        # 加载yaml配置文件，与原版方式一致
        hyper_yaml_path = os.path.join(self.model_dir, "cosyvoice2ca.yaml")
        if not os.path.exists(hyper_yaml_path):
            raise ValueError(f"{hyper_yaml_path} 配置文件不存在!")
        
        with open(hyper_yaml_path, 'r') as f:
            self.configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(self.model_dir, 'CosyVoice-BlankEN')})
            self.sgl_config = self.configs.get('sgl_config', {})
            self.base_url = self.sgl_config.get('base_url', 'http://localhost:30000')
            
        # 使用deque替代list，提高性能
        self.speech_tokens = deque()
        self.offset = 151936

        self.token_hop_len = 25
        self.token_overlap_len = 20
        self.llm_end = False
        # 从配置获取采样率和其他参数
        self.sample_rate = self.configs.get('sample_rate', 24000)
        self.token_mel_ratio = self.configs.get('token_mel_ratio', 2)
        self.token_frame_rate = self.configs.get('token_frame_rate', 25)
        
        # 初始化模型
        logging.info(f"Loading models from {self.model_dir}...")
        self._init_models()
        
        # 设置flow缓存大小
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        self.speech_window = np.hamming(2 * self.source_cache_len)
        
        # 使用与CosyVoice2相同的上下文管理
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        
        # 多线程相关 - 使用事件替代轮询
        self.lock = threading.Lock()
        self.token_ready_event = threading.Event()
        self.token_queue = queue.Queue()
        
        self.to_save_tokens = []
        
        # 预分配常用的张量
        self.zero_cache_source = torch.zeros(1, 1, 0, device=self.device)
        
        # 预分配张量缓冲区，避免频繁创建
        max_token_len = self.token_hop_len + self.flow.pre_lookahead_len
        self.token_buffer = torch.zeros(1, max_token_len, dtype=torch.int32, device=self.device)
        self.token_len_buffer = torch.zeros(1, dtype=torch.int32, device=self.device)
        
        # 复用session以减少连接开销
        self.session = requests.Session()
        
        # warmup_text = ["今天天气不错，适合喝一杯咖啡。", "前方八百米向右前方并入匝道。", "已经帮你操作了", "车门已经打开了", "已将音量调大到百分之三十", "副驾座椅已调直，座椅加热已打开"]
        # for i, text in enumerate(warmup_text, 1):
        #     print(f"Warmup {i}/{len(warmup_text)}: {text}")
        #     # 等待上一句完全结束
        #     while self.is_synthesizing:
        #         time.sleep(0.01)
        #     # 完全消费生成器，确保这句话完全合成完毕
        #     for _ in self.synthesize(text):
        #         pass
        #     print(f"Warmup {i} completed")
        #     torch.save(self.to_save_tokens, os.path.join(self.model_dir, "warmup_tokens.pt"))
        if os.path.exists(os.path.join(self.model_dir, "warmup_tokens.pt")):
            logging.info("warmup start")
            warmup_tokens = torch.load(os.path.join(self.model_dir, "warmup_tokens.pt"))
            if isinstance(warmup_tokens, list):
                warmup_tokens = torch.stack(warmup_tokens)
            warmup_pointer = 0
            is_first_chunk = True
            while warmup_pointer < len(warmup_tokens):
                if is_first_chunk:
                    self.first_chunk_params['token'] = warmup_tokens[warmup_pointer:warmup_pointer + 28].unsqueeze(0)
                    self.first_chunk_params['token_len'] = torch.tensor([28], dtype=torch.int32).to(self.device)
                    _ = self.token2wav(**self.first_chunk_params)
                    is_first_chunk = False
                else:
                    self.other_chunk_params['token'] = warmup_tokens[warmup_pointer:warmup_pointer + 28].unsqueeze(0)
                    self.other_chunk_params['token_len'] = torch.tensor([28], dtype=torch.int32).to(self.device)
                    this_tts_speech = self.token2wav(**self.other_chunk_params)
                warmup_pointer += 28
            self.refresh_flow_cache()
        else:
            logging.info("Warmup tokens not found, skipping warmup")
        # Convert list of tensors to a single tensor
        logging.info(f"CosyVoiceCA初始化完成")
        
    def _init_models(self):
        self.flow = self.configs['flow']
        self.flow_decoder_required_cache_size = self.token_hop_len * self.flow.token_mel_ratio
        self.hift = self.configs['hift']
        self.hift_cache = None
        self.tokenizer = self.configs['get_tokenizer']()
        
        # Add allowed_special attribute for tokenizer
        self.allowed_special = self.configs['allowed_special']

        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        
        # 应用FP16设置
        if self.fp16 and torch.cuda.is_available():
            self.flow.half()
            
        # 加载模型参数
        self._load_models()
            
    def _load_models(self):
        """按照原版加载模型权重"""
        try:
            # 确定加载的模型路径
            flow_path = os.path.join(self.model_dir, "flow.cache.pt")
                
            hift_path = os.path.join(self.model_dir, "hift.pt")
            # spk2info = os.path.join(self.model_dir, "spk2info-old-newcreate.pt")
            spk2info = os.path.join(self.model_dir, "spk2info-tongtong-short.pt")
            # spk2info = os.path.join(self.model_dir, "spk2info-tong.pt")
            # 检查文件是否存在
            if not os.path.exists(flow_path):
                raise ValueError(f"Flow模型文件不存在: {flow_path}")
            if not os.path.exists(hift_path):
                raise ValueError(f"HiFT模型文件不存在: {hift_path}")
            if not os.path.exists(spk2info):
                raise ValueError(f"spk2info文件不存在: {spk2info}")
            
            self.spk2info = torch.load(spk2info, map_location=self.device)['my_zero_shot_spk']
            
            # 加载模型权重，与原版CosyVoice2一致
            self.flow.load_state_dict(torch.load(flow_path, map_location=self.device), strict=True)
            self.flow.to(self.device).eval()
            # flow_encoder = torch.jit.load(os.path.join(self.model_dir, "flow.encoder.fp16.zip"), map_location=self.device)
            # self.flow.encoder = flow_encoder
            self.flow_cache = self.init_flow_cache()
            flow_embedding = F.normalize(self.spk2info['flow_embedding'].half().to(self.device), dim=1)
            self.spk2info['flow_embedding'] = self.flow.spk_embed_affine_layer(flow_embedding).to(self.device)
            self.spk2info['mask'] = (~make_pad_mask(torch.tensor([self.token_hop_len + self.flow.pre_lookahead_len + self.spk2info['flow_prompt_speech_token'].size(1)]))).unsqueeze(-1).to(self.device)
            self.first_chunk_params = {
                'prompt_token': self.spk2info['flow_prompt_speech_token'].to(self.device),
                'prompt_token_len': torch.tensor([self.spk2info['flow_prompt_speech_token'].size(1)], dtype=torch.int32).to(self.device),
                'prompt_feat': self.spk2info['prompt_speech_feat'].to(self.device),
                'prompt_feat_len': torch.tensor([self.spk2info['prompt_speech_feat'].size(1)], dtype=torch.int32).to(self.device),
                'embedding': self.spk2info['flow_embedding'].to(self.device),
                'finalize': False,
                'is_first_chunk': True,
                'mask': self.spk2info['mask'].to(self.device)
            }  
            
            self.zero_prompt_token = torch.zeros(1, 0, dtype=torch.int32, device=self.device)
            self.zero_prompt_feat = torch.zeros(1, 0, 80, device=self.device)
            self.other_chunk_params = {
                'prompt_token': self.zero_prompt_token.to(self.device),
                'prompt_token_len': torch.tensor([self.zero_prompt_token.size(1)], dtype=torch.int32).to(self.device),
                'prompt_feat': self.zero_prompt_feat.to(self.device),
                'prompt_feat_len': torch.tensor([self.zero_prompt_feat.size(1)], dtype=torch.int32).to(self.device),
                'embedding': self.spk2info['flow_embedding'].to(self.device),
                'finalize': False,
                'is_first_chunk': False,
            } 
            self.last_chunk_params = {
                'prompt_token': self.zero_prompt_token.to(self.device),
                'prompt_token_len': torch.tensor([self.zero_prompt_token.size(1)], dtype=torch.int32).to(self.device),
                'prompt_feat': self.zero_prompt_feat.to(self.device),
                'prompt_feat_len': torch.tensor([self.zero_prompt_feat.size(1)], dtype=torch.int32).to(self.device),
                'embedding': self.spk2info['flow_embedding'].to(self.device),
                'finalize': True,
                'is_first_chunk': False,
            }
            
            # 处理HiFT模型，按照CosyVoice2处理方式
            hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_path, map_location=self.device).items()}
            self.hift.load_state_dict(hift_state_dict, strict=True)
            self.hift.to(self.device).eval()
            
            self._load_trt_models()
            
            logging.info(f"模型加载成功: token_mel_ratio={self.token_mel_ratio}, token_frame_rate={self.token_frame_rate}")
        except Exception as e:
            logging.error(f"模型加载失败: {e}")
            raise
            
    def _load_trt_models(self):
        """加载TensorRT模型，如果需要"""
        try:
            # TensorRT引擎文件路径
            flow_decoder_estimator_model = os.path.join(
                self.model_dir, 
                f"flow.decoder.estimator.fp16.mygpu.plan"
                # f"flow.decoder.estimator.fp16.4090.plan"
                # f"flow.decoder.estimator.fp16.A10.plan"
                # f"flow.decoder.estimator.fp16.3070.plan"
            )
            
            if os.path.getsize(flow_decoder_estimator_model) == 0:
                raise ValueError(f"{flow_decoder_estimator_model}是空文件，请删除后重新导出!")
                
            # 加载TensorRT引擎
            del self.flow.decoder.estimator
            with open(flow_decoder_estimator_model, 'rb') as f:
                self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
                
            assert self.flow.decoder.estimator_engine is not None, f"加载TensorRT失败: {flow_decoder_estimator_model}"
            self.flow.decoder.estimator = self.flow.decoder.estimator_engine.create_execution_context()
            
            logging.info("TensorRT模型加载成功")
        except Exception as e:
            logging.error(f"TensorRT模型加载失败: {e}")

    def init_flow_cache(self):
        print(f"self.flow_decoder_required_cache_size: {self.flow_decoder_required_cache_size}")
        encoder_cache = {'offset': 0,
                         'pre_lookahead_layer_conv2_cache': torch.zeros(1, 512, 2).to(self.device),
                         'encoders_kv_cache': torch.zeros(6, 1, 8, 0, 64 * 2).to(self.device),
                         'upsample_offset': 0,
                         'upsample_conv_cache': torch.zeros(1, 512, 4).to(self.device),
                         'upsample_kv_cache': torch.zeros(4, 1, 8, 0, 64 * 2).to(self.device)}
        decoder_cache = {'offset': 0,
                         'down_blocks_conv_cache': torch.zeros(10, 1, 2, 832, 2).to(self.device),
                         'down_blocks_kv_cache': torch.zeros(10, 1, 4, 2, self.flow_decoder_required_cache_size, 512, 2).to(self.device),
                         'mid_blocks_conv_cache': torch.zeros(10, 12, 2, 512, 2).to(self.device),
                         'mid_blocks_kv_cache': torch.zeros(10, 12, 4, 2, self.flow_decoder_required_cache_size, 512, 2).to(self.device),
                         'up_blocks_conv_cache': torch.zeros(10, 1, 2, 1024, 2).to(self.device),
                         'up_blocks_kv_cache': torch.zeros(10, 1, 4, 2, self.flow_decoder_required_cache_size, 512, 2).to(self.device),
                         'final_blocks_conv_cache': torch.zeros(10, 2, 256, 2).to(self.device),
                         'down_blocks_kv_cache_out': torch.zeros(10, 1, 4, 2, 224, 512, 2).to(self.device),
                         'mid_blocks_kv_cache_out': torch.zeros(10, 12, 4, 2, 224, 512, 2).to(self.device),
                         'up_blocks_kv_cache_out': torch.zeros(10, 1, 4, 2, 224, 512, 2).to(self.device),
                         }
        if self.fp16 is True:
            for cache in [encoder_cache, decoder_cache]:
                for k, v in cache.items():
                    if isinstance(v, torch.Tensor):
                        cache[k] = v.half()
        cache = {'encoder_cache': encoder_cache, 'decoder_cache': decoder_cache}
        return cache

    def refresh_flow_cache(self):
        """重置flow cache，恢复初始形状但复用显存"""
        if hasattr(self, 'flow_cache') and self.flow_cache is not None:
            # 重置encoder cache
            encoder_cache = self.flow_cache['encoder_cache']
            encoder_cache['offset'] = 0
            dtype = torch.float16 if self.fp16 else torch.float32
            encoder_cache['pre_lookahead_layer_conv2_cache'] = torch.zeros(1, 512, 2, dtype=dtype, device=self.device)
            encoder_cache['encoders_kv_cache'] = torch.zeros(6, 1, 8, 0, 64 * 2, dtype=dtype, device=self.device)
            encoder_cache['upsample_offset'] = 0
            encoder_cache['upsample_conv_cache'] = torch.zeros(1, 512, 4, dtype=dtype, device=self.device)
            encoder_cache['upsample_kv_cache'] = torch.zeros(4, 1, 8, 0, 64 * 2, dtype=dtype, device=self.device)
            
            # 重置decoder cache  
            decoder_cache = self.flow_cache['decoder_cache']
            decoder_cache['offset'] = 0
            decoder_cache['down_blocks_conv_cache'] = torch.zero_(decoder_cache['down_blocks_conv_cache'])
            decoder_cache['down_blocks_kv_cache'] = torch.zero_(decoder_cache['down_blocks_kv_cache'])
            decoder_cache['mid_blocks_conv_cache'] = torch.zero_(decoder_cache['mid_blocks_conv_cache'])
            decoder_cache['mid_blocks_kv_cache'] = torch.zero_(decoder_cache['mid_blocks_kv_cache'])
            decoder_cache['up_blocks_conv_cache'] = torch.zero_(decoder_cache['up_blocks_conv_cache'])
            decoder_cache['up_blocks_kv_cache'] = torch.zero_(decoder_cache['up_blocks_kv_cache'])
            decoder_cache['final_blocks_conv_cache'] = torch.zero_(decoder_cache['final_blocks_conv_cache'])
            decoder_cache['down_blocks_kv_cache_out'] = torch.zero_(decoder_cache['down_blocks_kv_cache_out'])
            decoder_cache['mid_blocks_kv_cache_out'] = torch.zero_(decoder_cache['mid_blocks_kv_cache_out'])
            decoder_cache['up_blocks_kv_cache_out'] = torch.zero_(decoder_cache['up_blocks_kv_cache_out'])
        else:
            # 如果cache不存在，则初始化
            self.flow_cache = self.init_flow_cache()

    def _send_stream_request(self, payload):
        """发送流式请求并处理响应流 - 优化版本"""

        processed_tokens = 0
        try:
            # 发送请求 - 复用session
            response = self.session.post(
                self.base_url + "/generate",
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                logging.error(f"请求失败，状态码: {response.status_code}")
                yield {"error": f"请求失败，状态码: {response.status_code}"}
                return
            
            # 处理流式响应 - 减少不必要的计算
            for chunk in response.iter_lines(decode_unicode=False):
                if not chunk:
                    continue
                chunk_str = chunk.decode("utf-8")
                
                if chunk_str == "data: [DONE]":
                    self.llm_end = True
                    break
                    
                if chunk_str.startswith("data:"):
                    try:
                        data = json.loads(chunk_str[5:].strip())
                        if "output_ids" in data:
                            current_tokens = data["output_ids"]
                            
                            # 计算新增的token
                            new_tokens = current_tokens[processed_tokens:]
                            processed_tokens = len(current_tokens)
                            
                            # 添加到总token列表
                            yield {"tokens": new_tokens, "is_partial": True}
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"解析JSON时出错: {e} - 原始数据: {chunk_str}")
            
                
        except requests.RequestException as e:
            logging.error(f"请求错误: {str(e)}")
            yield {"error": f"请求错误: {str(e)}"}
        except Exception as e:
            logging.error(f"处理响应时出错: {str(e)}")
            yield {"error": f"处理响应错误: {str(e)}"}
    
    def _prepare_input_features(self, text: str):
        """准备输入特征"""
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        # print(text_token, text)
        text_len = len(text_token)
        model_input = self.spk2info
        prompt_text = model_input['prompt_text'].squeeze().tolist()
        llm_input = [158500] + prompt_text +text_token + [158497]
        # text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        # text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)

        speech_prompt_token = model_input['llm_prompt_speech_token'] + self.offset
        llm_input = llm_input + speech_prompt_token.squeeze().tolist()
        model_input['llm_input'] = llm_input
        model_input['text_len'] = text_len
        return model_input


        # 获取输入文本的token
    def _llm_job(self, input_ids: List[int], text_len: int):
        """
        处理LLM任务
        
        Args:
            input_ids: 输入的token IDs
        """
        payload = {
            "input_ids": input_ids,
            "stream": True,
            "sampling_params": {
                "top_p": 0.8, 
                "top_k": 25, 
                "temperature": 1.0, 
                "max_new_tokens": text_len * self.sgl_config.get('max_token_text_ratio', 20), 
                "ras_penalty": 3.0,
                "stop_token_ids": [158497]
            }
        }
        for i in self._send_stream_request(payload):
            
            # 使用锁保护对speech_tokens的修改
            with self.lock:
                self.speech_tokens.extend(i["tokens"])
                if self.speech_tokens and self.speech_tokens[-1] == 6561 + self.offset:
                    self.speech_tokens.pop()
            
            # if len(self.speech_tokens) >= self.token_hop_len + self.flow.pre_lookahead_len:
            self.token_ready_event.set()
        
        # LLM任务结束标志
        with self.lock:
            self.llm_end = True
        self.token_ready_event.set()  # 确保主线程能被唤醒
    
    def synthesize(self, text: str, session_id: str = "default") -> Generator[Dict, None, None]:
        """
        合成语音
        
        Args:
            text: 输入文本
            session_id: 会话ID
            
        Returns:
            生成器，产生包含生成的token的字典
        """
        if self.is_synthesizing is True:
            logging.info("正在合成中，请稍后再试")
            return
        
        self.is_synthesizing = True
        self.hift_cache = None  # 重置hift缓存
        
        # 使用锁保护状态初始化
        with self.lock:
            self.speech_tokens = []  # 重置语音token
            self.llm_end = False  # 重置LLM结束标志
        
        model_input = self._prepare_input_features(text)
        is_first_chunk = True
        p = threading.Thread(target=self._llm_job, args=(model_input['llm_input'], model_input['text_len']))
        p.start()
        print("synthesize text:", text)
        while True:
            # Wait for tokens to be ready, but don't rely on timeout alone
            self.token_ready_event.wait(timeout=0.1)
            
            # 使用锁保护对speech_tokens的访问和状态检查
            with self.lock:
                # Check if we have tokens available before attempting to pop
                if not self.speech_tokens:
                    # If no tokens available but LLM has ended, break the loop
                    if self.llm_end:
                        break
                    # Reset the event and continue waiting
                    self.token_ready_event.clear()
                    continue
                
                # 复制需要的tokens，避免长时间持有锁
                tokens_needed = min(len(self.speech_tokens), 
                                   self.token_hop_len + self.flow.pre_lookahead_len)
                current_tokens = list(self.speech_tokens[:tokens_needed])
            
            # Clear the event after consuming a token
            self.token_ready_event.clear()
            this_tts_speech_token = torch.tensor(current_tokens, device=self.device).unsqueeze(dim=0) - self.offset
            # if len(this_tts_speech_token[0]) == 28:
            #     self.to_save_tokens.extend(this_tts_speech_token[0])
            
            if is_first_chunk:
                self.first_chunk_params['token'] = this_tts_speech_token
                self.first_chunk_params['token_len'] = torch.tensor([this_tts_speech_token.shape[1]], dtype=torch.int32).to(self.device)
                this_tts_speech = self.token2wav(**self.first_chunk_params)
                is_first_chunk = False
            else:
                self.other_chunk_params['token'] = this_tts_speech_token
                self.other_chunk_params['token_len'] = torch.tensor([this_tts_speech_token.shape[1]], dtype=torch.int32).to(self.device)
                this_tts_speech = self.token2wav(**self.other_chunk_params)
            yield {'tts_speech': this_tts_speech.cpu()}
            
            # 使用锁保护移除操作和状态检查
            with self.lock:
                self.speech_tokens = self.speech_tokens[self.token_hop_len:]
                should_break = (self.llm_end is True and 
                               len(self.speech_tokens) < self.token_hop_len + self.flow.pre_lookahead_len)
            
            if should_break:
                break
        p.join()
        
        # 使用锁保护最后的token处理
        with self.lock:
            remaining_tokens = list(self.speech_tokens) if self.speech_tokens else []
        
        if len(remaining_tokens) >= 4:
            this_tts_speech_token = torch.tensor(remaining_tokens).unsqueeze(dim=0) - self.offset
            self.last_chunk_params['token'] = this_tts_speech_token
            self.last_chunk_params['token_len'] = torch.tensor([this_tts_speech_token.shape[1]], dtype=torch.int32).to(self.device)
            this_tts_speech = self.token2wav(**self.last_chunk_params)
            
            # 舍弃最后40ms的音频 (40ms * sample_rate = 样本数)
            samples_to_remove = int(0.04 * self.sample_rate)  # 40ms对应的采样点数
            if this_tts_speech.shape[1] > samples_to_remove:
                this_tts_speech = this_tts_speech[:, :-samples_to_remove]
            
            yield {'tts_speech': this_tts_speech.cpu()}
        
        #重置flow cache
        self.refresh_flow_cache()
        
        # 使用锁保护状态重置
        with self.lock:
            self.speech_tokens = []
            self.llm_end = False
        self.is_synthesizing = False
            
    def token2wav(self, token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding, finalize=False, is_first_chunk=False, mask=None):
        with torch.cuda.amp.autocast(self.fp16):
            flow_start = time.time()
            tts_mel, self.flow_cache = self.flow.inference(token=token.to(self.device),
                                                                      token_len=token_len,
                                                                      prompt_token=prompt_token,
                                                                      prompt_token_len=prompt_token_len,
                                                                      prompt_feat=prompt_feat,
                                                                      prompt_feat_len=prompt_feat_len,
                                                                      embedding=embedding,
                                                                      cache=self.flow_cache,
                                                                      is_first_chunk=is_first_chunk,
                                                                      finalize=finalize,
                                                                      mask=mask)
                # append hift cache
            flow_end = time.time()
            # print(f"flow inference time: {flow_end - flow_start:.2f}s")
            if self.hift_cache is not None:
                hift_cache_mel, hift_cache_source = self.hift_cache['mel'], self.hift_cache['source']
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
            else:
                hift_cache_source = self.zero_cache_source  # 使用预分配的零张量
            # keep overlap mel and hift cache
            if finalize is False:
                tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
                if self.hift_cache is not None:
                    tts_speech = fade_in_out(tts_speech, self.hift_cache['speech'], self.speech_window)
                self.hift_cache = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                            'source': tts_source[:, :, -self.source_cache_len:],
                                            'speech': tts_speech[:, -self.source_cache_len:]}
                tts_speech = tts_speech[:, :-self.source_cache_len]
            else:
                tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
                if self.hift_cache is not None:
                    tts_speech = fade_in_out(tts_speech, self.hift_cache['speech'], self.speech_window)
            # print(f"hift inference time: {time.time() - flow_end:.2f}s")
        return tts_speech

    def _safe_cache_copy(self, cache_dict, deep_copy=False):
        """
        安全的cache复制方法
        
        Args:
            cache_dict: 要复制的cache字典
            deep_copy: 是否进行深拷贝
            
        Returns:
            复制后的cache字典
        """
        if cache_dict is None:
            return None
            
        safe_cache = {}
        for key, value in cache_dict.items():
            if isinstance(value, torch.Tensor):
                if deep_copy:
                    # 深拷贝：完全独立的内存
                    safe_cache[key] = value.clone().detach()
                else:
                    # 浅拷贝：共享数据但独立的张量对象
                    safe_cache[key] = value.detach()
            else:
                # 非张量值直接复制
                safe_cache[key] = value
        return safe_cache
    
    def _validate_cache_memory_safety(self, cache_dict):
        """
        验证cache的内存安全性
        
        Args:
            cache_dict: 要验证的cache字典
            
        Returns:
            bool: 是否通过验证
        """
        if cache_dict is None:
            return False
            
        try:
            for key, tensor in cache_dict.items():
                if isinstance(tensor, torch.Tensor):
                    # 检查张量是否连续
                    if not tensor.is_contiguous():
                        print(f"警告: cache[{key}] 不是连续内存布局")
                        return False
                    
                    # 检查内存是否有效（通过访问data_ptr）
                    _ = tensor.data_ptr()
                    
                    # 检查设备一致性
                    if tensor.device != self.device:
                        print(f"警告: cache[{key}] 设备不匹配，期望{self.device}，实际{tensor.device}")
                        return False
                        
            return True
        except Exception as e:
            print(f"cache内存验证失败: {e}")
            return False
    
    def _safe_forward_chunk_call(self, encoder, token, token_len, context=None, cache=None, use_safe_copy=True):
        """
        安全的forward_chunk调用
        
        Args:
            encoder: encoder对象
            token: 输入token
            token_len: token长度
            context: 上下文（可选）
            cache: cache字典
            use_safe_copy: 是否使用安全复制
            
        Returns:
            tuple: (h, h_lengths, new_cache)
        """
        if cache is None:
            cache = {}
            
        # 验证输入cache的安全性
        if not self._validate_cache_memory_safety(cache):
            print("输入cache验证失败，使用默认cache")
            cache = {}
        
        # 准备安全的cache参数
        if use_safe_copy and cache:
            safe_cache = self._safe_cache_copy(cache, deep_copy=False)
        else:
            safe_cache = cache
            
        try:
            # 调用forward_chunk
            if context is not None:
                result = encoder.forward_chunk(token, token_len, context=context, **safe_cache)
            else:
                result = encoder.forward_chunk(token, token_len, **safe_cache)
                
            # 验证输出
            h, h_lengths, new_cache = result
            
            # 检查输出的有效性
            if torch.isnan(h).any() or torch.isinf(h).any():
                raise RuntimeError("forward_chunk输出包含NaN或Inf值")
                
            return h, h_lengths, new_cache
            
        except RuntimeError as e:
            if "CUDA error" in str(e) or "device-side assert" in str(e):
                print(f"CUDA错误，可能与cache内存状态有关: {e}")
                # 清理可能损坏的内存状态
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            raise e

    def demonstrate_cache_passing_risks(self):
        """
        演示不同cache传递方式的风险和对比
        
        这个方法展示了：
        1. 直接传递的风险
        2. 使用data_ptr()的场景
        3. 安全传递的最佳实践
        """
        print("=== Cache传递方式风险分析演示 ===")
        
        # 创建一个示例cache
        test_cache = {
            'offset': 0,
            'conv_cache': torch.randn(1, 512, 2, device=self.device),
            'kv_cache': torch.randn(6, 1, 8, 0, 128, device=self.device)
        }
        
        print("1. 原始cache状态:")
        for key, value in test_cache.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: shape={value.shape}, data_ptr={hex(value.data_ptr())}")
        
        # 方式1: 直接传递引用（风险较高）
        print("\n2. 直接传递引用的风险:")
        def risky_function(**cache_params):
            """模拟一个可能修改cache的函数"""
            if 'conv_cache' in cache_params:
                # 直接修改传入的张量
                cache_params['conv_cache'].fill_(999.0)  # 风险操作
                print(f"   函数内修改后 conv_cache 前几个值: {cache_params['conv_cache'].flatten()[:3]}")
        
        print("   调用前 conv_cache 前几个值:", test_cache['conv_cache'].flatten()[:3])
        risky_function(**test_cache)  # 直接传递
        print("   调用后 conv_cache 前几个值:", test_cache['conv_cache'].flatten()[:3])
        print("   ❌ 原始cache被意外修改！")
        
        # 方式2: 使用data_ptr()（底层操作）
        print("\n3. data_ptr()方式的特点:")
        conv_cache = test_cache['conv_cache']
        print(f"   data_ptr: {hex(conv_cache.data_ptr())}")
        print(f"   is_contiguous: {conv_cache.is_contiguous()}")
        print(f"   element_size: {conv_cache.element_size()} bytes")
        print(f"   总内存大小: {conv_cache.numel() * conv_cache.element_size()} bytes")
        
        # 检查内存连续性的重要性
        non_contiguous = conv_cache.transpose(0, 1)  # 创建非连续张量
        print(f"   转置后 is_contiguous: {non_contiguous.is_contiguous()}")
        print("   💡 非连续张量需要调用 .contiguous() 才能安全使用data_ptr()")
        
        # 方式3: 安全复制（推荐）
        print("\n4. 安全复制方式:")
        safe_cache = self._safe_cache_copy(test_cache, deep_copy=False)
        
        def safe_function(**cache_params):
            if 'conv_cache' in cache_params:
                cache_params['conv_cache'].fill_(777.0)
                print(f"   安全函数内修改后: {cache_params['conv_cache'].flatten()[:3]}")
        
        print("   原始cache:", test_cache['conv_cache'].flatten()[:3])
        safe_function(**safe_cache)
        print("   调用后原始cache:", test_cache['conv_cache'].flatten()[:3])
        print("   ✅ 原始cache未被修改，但共享底层内存（适合大多数场景）")
        
        # 方式4: 深拷贝（最安全但开销大）
        print("\n5. 深拷贝方式:")
        deep_cache = self._safe_cache_copy(test_cache, deep_copy=True)
        print(f"   原始 data_ptr: {hex(test_cache['conv_cache'].data_ptr())}")
        print(f"   深拷贝 data_ptr: {hex(deep_cache['conv_cache'].data_ptr())}")
        print("   ✅ 完全独立的内存，但消耗更多显存")
        
        print("\n=== 推荐使用场景 ===")
        print("1. 普通推理: 使用浅拷贝 (_safe_cache_copy(deep_copy=False))")
        print("2. 批量测试: 使用深拷贝 (_safe_cache_copy(deep_copy=True))")
        print("3. TensorRT/C++调用: 使用 tensor.contiguous().data_ptr()")
        print("4. 调试问题: 使用 _validate_cache_memory_safety() 检查")
    
    def get_cache_memory_info(self, cache_dict, name="cache"):
        """
        获取cache的详细内存信息
        
        Args:
            cache_dict: cache字典
            name: cache名称
            
        Returns:
            str: 内存信息摘要
        """
        if cache_dict is None:
            return f"{name}: None"
        
        info_lines = [f"{name} 内存信息:"]
        total_memory = 0
        
        for key, value in cache_dict.items():
            if isinstance(value, torch.Tensor):
                memory_size = value.numel() * value.element_size()
                total_memory += memory_size
                info_lines.append(
                    f"  {key}: {value.shape} | {memory_size/1024/1024:.2f}MB | "
                    f"ptr={hex(value.data_ptr())} | contiguous={value.is_contiguous()}"
                )
            else:
                info_lines.append(f"  {key}: {value} (非张量)")
        
        info_lines.append(f"总内存使用: {total_memory/1024/1024:.2f}MB")
        return "\n".join(info_lines)

# 使用示例
if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="CosyVoiceCA演示程序")
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录路径")
    parser.add_argument("--output", type=str, default="output.wav", help="输出音频文件路径")
    args = parser.parse_args()
    
    # 设置日志
    cosyvoice_ca = CosyVoiceCA(args.model_dir)
    all_tokens = []
    
    # 创建输出文件夹
    import os
    os.makedirs("outputs", exist_ok=True)
    
    # 创建日志文件
    log_file = open("log.txt", "w", encoding="utf-8")
    
    def log_and_print(message):
        """同时输出到控制台和日志文件"""
        print(message)
        log_file.write(message + "\n")
        log_file.flush()
    
    # 从test.txt文件读取测试文本
    log_and_print(f"\n=== 批量测试：从test.txt文件读取文本 ===")
    txt_file_path = "test.txt"
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            text_lines = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        log_and_print(f"警告：找不到文件 {txt_file_path}，退出程序")
        log_file.close()
        exit(1)
    except Exception as e:
        log_and_print(f"读取文件时出错: {e}，退出程序")
        log_file.close()
        exit(1)
    
    if not text_lines:
        log_and_print(f"文件 {txt_file_path} 为空，退出程序")
        log_file.close()
        exit(1)
        
    log_and_print(f"开始从 {txt_file_path} 逐行合成文本...")
    log_and_print(f"共找到 {len(text_lines)} 行文本")
    
    # 存储性能数据
    first_packet_times = []
    rtf_values = []
    successful_count = 0
    last_audio = None
    
    for i, text in enumerate(text_lines, 1):
        if len(text) > 50:  # 如果文本过长，显示省略版本
            display_text = text[:50] + "..."
        else:
            display_text = text
        log_and_print(f"\n=== 第 {i}/{len(text_lines)} 句: '{display_text}' ===")
        
        # 记录开始时间
        start_time = time.time()
        first_packet_time = None
        
        try:
            current_line_segments = []
            for result in cosyvoice_ca.synthesize(text):
                # 记录首包时间
                if first_packet_time is None:
                    first_packet_time = time.time()
                    current_first_packet_delay = first_packet_time - start_time
                    log_and_print(f"首包时延: {current_first_packet_delay:.2f}秒")
                    first_packet_times.append(current_first_packet_delay)
                    
                current_line_segments.append(result['tts_speech'])
            
            # 记录合成结束时间
            synthesis_time = time.time()
            synthesis_duration = synthesis_time - start_time
            log_and_print(f"合成耗时: {synthesis_duration:.2f}秒")
            
            # 合并当前行的所有音频片段并保存
            if current_line_segments:
                combined_line_audio = torch.cat(current_line_segments, dim=1)
                audio_length = combined_line_audio.shape[1] / cosyvoice_ca.sample_rate
                
                # 计算实时率 (RTF = 合成时间 / 音频长度)
                rtf = synthesis_duration / audio_length
                rtf_values.append(rtf)
                log_and_print(f"实时率(RTF): {rtf:.2f}")
                log_and_print(f"音频长度: {audio_length:.2f}秒")
                
                line_output_path = f"outputs/batch_test_{i:04d}.wav"
                torchaudio.save(line_output_path, combined_line_audio, cosyvoice_ca.sample_rate)
                log_and_print(f"✓ 已保存音频: {line_output_path}")
                successful_count += 1
                
                # 记录最后一句的音频用于最终报告
                if i == len(text_lines):
                    last_audio = combined_line_audio
            else:
                log_and_print(f"❌ 第 {i} 行没有生成音频片段")
                
        except Exception as e:
            log_and_print(f"第 {i} 行合成失败: {e}")
            continue
    
    # 统计结果
    log_and_print("\n" + "="*50)
    log_and_print("📊 批量测试统计报告")
    log_and_print("="*50)
    
    log_and_print(f"总文本数量: {len(text_lines)}")
    log_and_print(f"成功合成: {successful_count} 句")
    log_and_print(f"失败合成: {len(text_lines) - successful_count} 句")
    log_and_print(f"成功率: {(successful_count/len(text_lines)*100):.1f}%")
    
    if first_packet_times and rtf_values:
        avg_first_packet = sum(first_packet_times) / len(first_packet_times)
        avg_rtf = sum(rtf_values) / len(rtf_values)
        
        log_and_print("\n⏱️ 首包时延统计:")
        for i, delay in enumerate(first_packet_times[:10], 1):  # 只显示前10个
            log_and_print(f"  第{i}句: {delay:.2f}秒")
        if len(first_packet_times) > 10:
            log_and_print(f"  ... (共{len(first_packet_times)}句)")
        log_and_print(f"  平均首包时延: {avg_first_packet:.2f}秒")
        
        log_and_print("\n🚀 RTF统计:")
        for i, rtf in enumerate(rtf_values[:10], 1):  # 只显示前10个
            log_and_print(f"  第{i}句: {rtf:.2f}")
        if len(rtf_values) > 10:
            log_and_print(f"  ... (共{len(rtf_values)}句)")
        log_and_print(f"  平均RTF: {avg_rtf:.2f}")
        
        log_and_print(f"\n📈 性能指标:")
        log_and_print(f"  最快首包时延: {min(first_packet_times):.2f}秒")
        log_and_print(f"  最慢首包时延: {max(first_packet_times):.2f}秒")
        log_and_print(f"  最佳RTF: {min(rtf_values):.2f}")
        log_and_print(f"  最差RTF: {max(rtf_values):.2f}")
        
        # 总结保存的音频文件
        log_and_print(f"\n💾 共保存了 {successful_count} 个音频文件:")
        log_and_print(f"  音频文件保存在: outputs/batch_test_XXXX.wav")
        
        # 同时保存最后一句到指定的输出文件
        if last_audio is not None:
            # 如果用户指定的输出文件没有路径，也放到outputs文件夹
            if not os.path.dirname(args.output):
                final_output_path = f"outputs/{args.output}"
            else:
                final_output_path = args.output
            torchaudio.save(final_output_path, last_audio, cosyvoice_ca.sample_rate)
            final_audio_length = last_audio.shape[1] / cosyvoice_ca.sample_rate
            log_and_print(f"\n📁 最后一句也保存为: {final_output_path}, 长度: {final_audio_length:.2f}秒")
        
        log_and_print("="*50)
    else:
        log_and_print("\n⚠️ 没有成功的合成记录，无法生成性能统计")
        log_and_print("="*50)
    
    # 最后的日志信息
    log_and_print(f"\n日志已保存到: log.txt")
    
    # 关闭日志文件
    log_file.close()
