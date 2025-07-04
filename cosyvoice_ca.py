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
from cosyvoice.utils.common import TrtContextWrapper
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
        self.first_chunk_buffer_len = 120
        self._init_models()
        
        # 设置flow缓存大小
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        self.speech_window = np.hamming(2 * self.source_cache_len)
        
        # 多线程相关 - 使用事件替代轮询
        self.lock = threading.Lock()
        self.token_ready_event = threading.Event()
        self.token_queue = queue.Queue()
        
        # 预分配常用的张量
        self.zero_cache_source = torch.zeros(1, 1, 0, device=self.device)
        
        # 预分配张量缓冲区，避免频繁创建
        max_token_len = self.token_hop_len + self.flow.pre_lookahead_len
        self.token_buffer = torch.zeros(1, max_token_len, dtype=torch.int32, device=self.device)
        self.token_len_buffer = torch.zeros(1, dtype=torch.int32, device=self.device)
        
        # 复用session以减少连接开销
        self.session = requests.Session()

        
        if os.path.exists(os.path.join(self.model_dir, "warmup_tokens.pt")):
            logging.info("warmup start")
            warmup_tokens = torch.load(os.path.join(self.model_dir, "warmup_tokens.pt"))
            if isinstance(warmup_tokens, list):
                warmup_tokens = torch.stack(warmup_tokens)
            warmup_pointer = 0
            hop_size = self.token_hop_len + self.prompt_token_pad + self.flow.pre_lookahead_len
            while warmup_pointer + hop_size < len(warmup_tokens):
                self.flow_inputs['token'] = warmup_tokens[warmup_pointer:warmup_pointer + hop_size].unsqueeze(0)
                self.flow_inputs['token_offset'] = 0
                self.flow_inputs['mask'] = self.spk2info['mask']
                _ = self.token2wav(**self.flow_inputs)
                warmup_pointer += self.token_hop_len
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
            spk2info = os.path.join(self.model_dir, "spk2info-normal.pt")
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
            flow_embedding = F.normalize(self.spk2info['flow_embedding'].half().to(self.device), dim=1)
            self.spk2info['flow_embedding'] = self.flow.spk_embed_affine_layer(flow_embedding).to(self.device)
            self.prompt_token_pad = int(np.ceil(self.spk2info['flow_prompt_speech_token'].size(1) / self.token_hop_len) * self.token_hop_len - self.spk2info['flow_prompt_speech_token'].size(1))
    
            self.spk2info['mask'] = (~make_pad_mask(torch.tensor([self.flow.pre_lookahead_len + self.token_hop_len + self.prompt_token_pad + self.spk2info['flow_prompt_speech_token'].size(1)]))).unsqueeze(-1).to(self.device)
            self.flow_inputs = {
                'prompt_token': self.spk2info['flow_prompt_speech_token'].to(self.device),
                'prompt_token_len': torch.tensor([self.spk2info['flow_prompt_speech_token'].size(1)], dtype=torch.int32).to(self.device),
                'prompt_feat': self.spk2info['prompt_speech_feat'].to(self.device),
                'prompt_feat_len': torch.tensor([self.spk2info['prompt_speech_feat'].size(1)], dtype=torch.int32).to(self.device),
                'embedding': self.spk2info['flow_embedding'],
                'finalize': False,
                'mask': self.spk2info['mask']
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
                # f"flow.decoder.estimator.fp16.mygpu.plan"
                f"flow.decoder.estimator.fp16.new.plan"
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
            self.flow.decoder.estimator = TrtContextWrapper(self.flow.decoder.estimator_engine, trt_concurrent = 1, device=self.device)

            
            logging.info("TensorRT模型加载成功")
        except Exception as e:
            logging.error(f"TensorRT模型加载失败: {e}")

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
    
    def _reset_cuda_state(self):
        """重置CUDA状态和实例状态，用于错误恢复"""
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # 重置实例状态
            with self.lock:
                self.speech_tokens.clear()  # 使用clear()来清空deque
                self.llm_end = False
            self.is_synthesizing = False
            self.hift_cache = None
            
            # 重置 flow_inputs 的动态部分
            if hasattr(self, 'flow_inputs'):
                self.flow_inputs['finalize'] = False
                self.flow_inputs['mask'] = self.spk2info['mask']
                if 'token' in self.flow_inputs:
                    del self.flow_inputs['token']
                if 'token_offset' in self.flow_inputs:
                    del self.flow_inputs['token_offset']
            
            logging.info("CUDA状态和实例状态已重置")
            return True
        except Exception as e:
            logging.error(f"重置状态时出错: {e}")
            return False

    def _safe_token2wav(self, **kwargs):
        """带容错的 token2wav 方法"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                return self.token2wav(**kwargs)
            except RuntimeError as e:
                error_msg = str(e)
                if "device-side assert triggered" in error_msg or "CUDA error" in error_msg:
                    logging.warning(f"CUDA错误 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
                    
                    if attempt < max_retries - 1:
                        # 尝试恢复
                        logging.info("尝试恢复CUDA状态...")
                        self._reset_cuda_state()
                        time.sleep(0.1)  # 短暂等待
                        continue
                    else:
                        logging.error("CUDA错误恢复失败，跳过当前chunk")
                        return None
                else:
                    # 非CUDA错误，直接抛出
                    raise
        
        return None

    def synthesize(self, text: str, session_id: str = "default") -> Generator[Dict, None, None]:
        """
        合成语音 - 带容错机制
        
        Args:
            text: 输入文本
            session_id: 会话ID
            
        Returns:
            生成器，产生包含生成的token的字典
        """
        if self.is_synthesizing is True:
            logging.info("正在合成中，请稍后再试")
            return
        
        # 初始状态设置
        self.is_synthesizing = True
        synthesis_success = False
        
        try:
            # 重置状态
            self.hift_cache = None
            
                                      # 使用锁保护状态初始化
            with self.lock:
                self.speech_tokens.clear()
                self.llm_end = False
            
            model_input = self._prepare_input_features(text)
            time_llm_start = time.time()
            p = threading.Thread(target=self._llm_job, args=(model_input['llm_input'], model_input['text_len']))
            p.start()
            print("synthesize text:", text)
            is_first_chunk = True
            token_offset = 0
            chunks_generated = 0
            
            while True:
                # Wait for tokens to be ready
                self.token_ready_event.wait(timeout=0.1)
                
                # 使用锁保护对speech_tokens的访问和状态检查
                with self.lock:
                    if not self.speech_tokens:
                        if self.llm_end:
                            break
                        self.token_ready_event.clear()
                        continue
                
                this_token_hop_len = self.token_hop_len + self.prompt_token_pad if token_offset == 0 else self.token_hop_len
                required_tokens = this_token_hop_len + self.flow.pre_lookahead_len
                
                if len(self.speech_tokens) - token_offset >= required_tokens:
                    time_llm_first_chunk_end = time.time()
                    print(f"llm first chunk time in ms: {(time_llm_first_chunk_end - time_llm_start) * 1000}")
                    # 修复：将 deque 转换为 list 以支持切片操作
                    speech_tokens_list = list(self.speech_tokens)
                    this_tts_speech_token = torch.tensor(speech_tokens_list[:token_offset + required_tokens]).unsqueeze(dim=0) - self.offset
                    print(f"当前token数: {len(self.speech_tokens)}, 需要: {required_tokens}, 跳跃长度: {this_token_hop_len}")
                    
                    # 准备参数
                    self.flow_inputs['token'] = this_tts_speech_token
                    self.flow_inputs['token_offset'] = token_offset
                    if is_first_chunk:
                        self.flow_inputs['mask'] = self.spk2info['mask']
                        is_first_chunk = False
                    else: 
                        self.flow_inputs['mask'] = None
                    
                    # 安全调用 token2wav
                    try:
                        this_tts_speech = self._safe_token2wav(**self.flow_inputs)
                        
                        if this_tts_speech is not None:
                            token_offset += this_token_hop_len
                            chunks_generated += 1
                            yield {'tts_speech': this_tts_speech.cpu()}
                        else:
                            # 如果当前chunk失败，记录并继续
                            logging.warning(f"Chunk {chunks_generated + 1} 生成失败，跳过")
                            token_offset += this_token_hop_len
                            
                    except Exception as e:
                        logging.error(f"处理音频chunk时出错: {e}")
                        # 尝试恢复并继续
                        self._reset_cuda_state()
                        token_offset += this_token_hop_len
                        continue
                
                if self.llm_end and len(self.speech_tokens) - token_offset < required_tokens:
                    break
            
            # 等待LLM线程结束
            p.join()
            
            # 最终处理
            try:
                if len(self.speech_tokens) > token_offset:
                    # 修复：将 deque 转换为 list 以支持 torch.tensor 转换
                    speech_tokens_list = list(self.speech_tokens)
                    this_tts_speech_token = torch.tensor(speech_tokens_list).unsqueeze(dim=0) - self.offset
                    self.flow_inputs['token'] = this_tts_speech_token
                    self.flow_inputs['token_offset'] = token_offset
                    self.flow_inputs['finalize'] = True
                    self.flow_inputs['mask'] = None
                    
                    final_speech = self._safe_token2wav(**self.flow_inputs)
                    if final_speech is not None:
                        yield {'tts_speech': final_speech.cpu()}
                        synthesis_success = True
                    else:
                        logging.warning("最终chunk生成失败")
                
            except Exception as e:
                logging.error(f"最终处理时出错: {e}")
                # 即使最终处理失败，如果之前有成功的chunk，也算部分成功
                if chunks_generated > 0:
                    synthesis_success = True
        
        except Exception as e:
            logging.error(f"合成过程中出现未预期错误: {e}")
            import traceback
            logging.error(f"详细错误信息: {traceback.format_exc()}")
        
        finally:
            # 确保状态正确重置
            try:
                # 使用锁保护状态重置
                with self.lock:
                    self.speech_tokens.clear()
                    self.llm_end = False
                    self.is_synthesizing = False
                    if hasattr(self, 'flow_inputs'):
                        self.flow_inputs['finalize'] = False
                
                # 清理CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.current_stream().synchronize()
                
                if synthesis_success:
                    logging.info(f"文本合成完成: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                else:
                    logging.warning(f"文本合成失败或部分失败: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                    
            except Exception as cleanup_error:
                logging.error(f"清理状态时出错: {cleanup_error}")
                # 强制重置关键状态
                self.is_synthesizing = False

    def token2wav(self, token, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding, token_offset, finalize=False, mask=None):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=prompt_token_len,
                                             prompt_feat=prompt_feat,
                                             prompt_feat_len=prompt_feat_len,
                                             embedding=embedding,
                                             streaming=True,
                                             finalize=finalize,
                                             mask=mask)
            time_start = time.time()
            tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]                # append hift cache
            if self.hift_cache is not None:
                hift_cache_mel, hift_cache_source = self.hift_cache['mel'], self.hift_cache['source']
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
            else:
                hift_cache_source = self.zero_cache_source  # 使用预分配的零张量
            
            # keep overlap mel and hift cache
            if finalize is False:
                time_hift_inference_start = time.time()
                tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
                time_hift_inference_end = time.time()
                print(f"hift inference time in ms: {(time_hift_inference_end - time_hift_inference_start) * 1000}")
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
            time_end = time.time()
            print(f"hift time in ms: {(time_end - time_start) * 1000}")
        return tts_speech

# 使用示例
if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="CosyVoiceCA演示程序")
    parser.add_argument("--model_dir", type=str, required=True, help="模型目录路径")
    parser.add_argument("--output", type=str, default="output.wav", help="输出音频文件路径")
    parser.add_argument("--mode", type=str, choices=['single', 'batch'], default='batch', help="测试模式: single=单条测试, batch=批量测试")
    parser.add_argument("--text", type=str, default="今天天气怎么样", help="单条测试时使用的文本")
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
    
    # 根据模式选择测试方式
    if args.mode == 'single':
        # 单条测试模式
        log_and_print(f"\n=== 单条测试模式 ===")
        log_and_print(f"测试文本: '{args.text}'")
        text_lines = [args.text]
    else:
        # 从test.txt文件读取测试文本 - 批量测试模式
        log_and_print(f"\n=== 批量测试：从test.txt文件读取所有文本 ===")
        txt_file_path = "test.txt"
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                text_lines = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            log_and_print(f"警告：找不到文件 {txt_file_path}，使用默认测试文本")
            text_lines = ["今天天气不错，适合出去走走。"]
        except Exception as e:
            log_and_print(f"读取文件时出错: {e}，使用默认测试文本")
            text_lines = ["今天天气不错，适合出去走走。"]
        
        if not text_lines:
            log_and_print(f"文件 {txt_file_path} 为空，使用默认测试文本")
            text_lines = ["今天天气不错，适合出去走走。"]
    
    log_and_print(f"开始批量测试，共找到 {len(text_lines)} 条文本")
    if len(text_lines) <= 5:
        for i, text in enumerate(text_lines, 1):
            display_text = text[:50] + "..." if len(text) > 50 else text
            log_and_print(f"  第{i}条: '{display_text}'")
    else:
        log_and_print(f"  前3条:")
        for i in range(3):
            display_text = text_lines[i][:50] + "..." if len(text_lines[i]) > 50 else text_lines[i]
            log_and_print(f"    第{i+1}条: '{display_text}'")
        log_and_print(f"  ... 共{len(text_lines)}条文本")
    
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
            chunk_count = 0
            log_and_print("开始合成，实时输出音频块...")
            
            for result in cosyvoice_ca.synthesize(text):
                chunk_count += 1
                # 记录首包时间
                if first_packet_time is None:
                    first_packet_time = time.time()
                    current_first_packet_delay = first_packet_time - start_time
                    log_and_print(f"📦 首包时延: {current_first_packet_delay:.2f}秒")
                    first_packet_times.append(current_first_packet_delay)
                
                current_line_segments.append(result['tts_speech'])
                current_audio_length = result['tts_speech'].shape[1] / cosyvoice_ca.sample_rate
                log_and_print(f"  💫 第{chunk_count}个音频块: {current_audio_length:.2f}秒")
            
            # 记录合成结束时间
            synthesis_time = time.time()
            synthesis_duration = synthesis_time - start_time
            log_and_print(f"⏱️ 总合成耗时: {synthesis_duration:.2f}秒")
            log_and_print(f"📊 总共生成了 {chunk_count} 个音频块")
            
            # 合并当前行的所有音频片段并保存
            if current_line_segments:
                combined_line_audio = torch.cat(current_line_segments, dim=1)
                audio_length = combined_line_audio.shape[1] / cosyvoice_ca.sample_rate
                
                # 计算实时率 (RTF = 合成时间 / 音频长度)
                rtf = synthesis_duration / audio_length
                rtf_values.append(rtf)
                log_and_print(f"🚀 实时率(RTF): {rtf:.2f}")
                log_and_print(f"🎵 音频总长度: {audio_length:.2f}秒")
                
                if args.mode == 'single':
                    line_output_path = f"outputs/single_test.wav"
                else:
                    line_output_path = f"outputs/batch_test_{i:04d}.wav"
                torchaudio.save(line_output_path, combined_line_audio, cosyvoice_ca.sample_rate)
                log_and_print(f"✅ 已保存音频: {line_output_path}")
                successful_count += 1
                
                # 记录最后一句的音频用于最终报告
                if i == len(text_lines):
                    last_audio = combined_line_audio
            else:
                log_and_print(f"❌ 没有生成音频片段")
                
        except Exception as e:
            log_and_print(f"❌ 第 {i} 行合成失败: {e}")
            import traceback
            log_and_print(f"详细错误信息: {traceback.format_exc()}")
            continue
    
    # 统计结果
    log_and_print("\n" + "="*50)
    if args.mode == 'single':
        log_and_print("📊 单条测试统计报告")
    else:
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
        if args.mode == 'single':
            log_and_print(f"  音频文件保存在: outputs/single_test.wav")
        else:
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