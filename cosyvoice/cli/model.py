# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Generator
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt
from cosyvoice.utils.file_utils import logging
import pdb
import requests
import json


class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()

        self.speech_tokens = []



        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model):
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(), flow_decoder_onnx_model, fp16)
        if os.path.getsize(flow_decoder_estimator_model) == 0:
            raise ValueError('{} is empty file, delete it and export again!'.format(flow_decoder_estimator_model))
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert self.flow.decoder.estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        self.flow.decoder.estimator = self.flow.decoder.estimator_engine.create_execution_context()

    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 200), (2, 1, 200), (2, 80, 200), (2, 80, 200)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}
    
    def _send_stream_request(self, payload):
        """发送流式请求并处理响应流
        
        Args:
            payload: 请求负载
            
        Yields:
            字典，包含生成的token或错误信息
        """
        # 从配置获取服务器URL
        base_url = 'http://localhost:30000'
        
        try:
            # 记录请求开始时间
            request_start_time = time.time()
            first_response_received = False
            
            logging.info(f"开始发送流式请求到: {base_url}/generate")
            
            # 计时：JSON序列化
            json_start = time.time()
            request_json = json.dumps(payload)
            json_end = time.time()
            json_time = (json_end - json_start) * 1000
            logging.info(f"JSON序列化耗时: {json_time:.2f}ms, 大小: {len(request_json)/1024:.2f}KB")
            
            # 计时：发送请求
            # 使用session可以减少连接建立的开销
            session = requests.Session()
            
            send_start = time.time()
            # 发送请求
            response = session.post(
                base_url + "/generate",
                json=payload,  # 直接使用json参数，简单明了
                stream=True,
                timeout=60
            )
            send_end = time.time()
            send_time = (send_end - send_start) * 1000
            
            # 记录请求发送完成时间
            request_sent_time = time.time()
            total_request_time = (request_sent_time - request_start_time) * 1000
            logging.info(f"请求发送完成，总耗时: {total_request_time:.2f}ms")
            logging.info(f"  - 其中连接和传输耗时: {send_time:.2f}ms ({send_time/total_request_time*100:.1f}%)")
            logging.info(f"  - 其中JSON序列化耗时: {json_time:.2f}ms ({json_time/total_request_time*100:.1f}%)")
            logging.info(f"  - 其他操作耗时: {(total_request_time-send_time-json_time):.2f}ms ({(total_request_time-send_time-json_time)/total_request_time*100:.1f}%)")
            
            if response.status_code != 200:
                logging.error(f"请求失败，状态码: {response.status_code}, 响应内容: {response.text}")
                yield {"error": f"请求失败，状态码: {response.status_code}"}
                return
            
            # 用于存储所有生成的token IDs
            all_tokens = []
            # 用于跟踪已经处理过的token数量
            processed_tokens = 0
            
            logging.info("开始接收流式输出...")
            
            # 处理流式响应
            for chunk in response.iter_lines(decode_unicode=False):
                # 如果这是第一个响应块，记录首包时间
                if not first_response_received:
                    first_response_time = time.time()
                    first_response_received = True
                    first_package_latency = (first_response_time - request_start_time) * 1000  # 转换为毫秒
                    server_processing_time = (first_response_time - request_sent_time) * 1000  # 服务器处理时间
                    logging.info(f"收到首包，总延迟: {first_package_latency:.2f}ms")
                    logging.info(f"  - 请求发送耗时: {total_request_time:.2f}ms")
                    logging.info(f"  - 服务器处理耗时: {server_processing_time:.2f}ms")
                
                if not chunk:
                    continue
                    
                chunk = chunk.decode("utf-8")
                
                # 只处理包含数据的行
                if chunk.startswith("data:"):
                    chunk_received_time = time.time()
                    
                    if chunk == "data: [DONE]":
                        total_time = (chunk_received_time - request_start_time) * 1000
                        logging.info(f"流式输出完成，总耗时: {total_time:.2f}ms")
                        if all_tokens:
                            # 计算token生成速度
                            tokens_per_second = len(all_tokens) / (total_time / 1000) if total_time > 0 else 0
                            logging.info(f"生成了 {len(all_tokens)} 个tokens，速度: {tokens_per_second:.2f} tokens/秒")
                            yield {"tokens": all_tokens}
                        break
                        
                    try:
                        # 解析JSON数据
                        data = json.loads(chunk[5:].strip("\n"))
                        
                        # 提取output_ids数组
                        if "output_ids" in data:
                            current_tokens = data["output_ids"]
                            
                            # 计算新增的token
                            new_tokens = current_tokens[processed_tokens:]
                            processed_tokens = len(current_tokens)
                            
                            # 添加到总token列表
                            all_tokens.extend(new_tokens)
                            
                            # 返回新增的token
                            if new_tokens:
                                token_time = (chunk_received_time - request_start_time) * 1000
                                logging.debug(f"新增 {len(new_tokens)} 个tokens{new_tokens}，已生成 {len(all_tokens)} 个tokens，总耗时: {token_time:.2f}ms")
                                yield {"tokens": new_tokens, "is_partial": True}
                            
                    except json.JSONDecodeError as e:
                        logging.error(f"解析JSON时出错: {e} - 原始数据: {chunk}")
            
            # 如果没有正常结束，但有收集到tokens，也返回结果
            if all_tokens and not first_response_received:
                yield {"tokens": all_tokens}
                
        except requests.RequestException as e:
            logging.error(f"请求错误: {str(e)}")
            yield {"error": f"请求错误: {str(e)}"}
        except Exception as e:
            logging.error(f"处理响应时出错: {str(e)}")
            yield {"error": f"处理响应错误: {str(e)}"}

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        with self.llm_context, torch.cuda.amp.autocast(self.fp16):
            if isinstance(text, Generator):
                assert isinstance(self, CosyVoice2Model), 'streaming input text is only implemented for CosyVoice2!'
                for i in self.llm.inference_bistream(text=text,
                                                     prompt_text=prompt_text.to(self.device),
                                                     prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                     prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                     prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                     embedding=llm_embedding.to(self.device)):
                    self.tts_speech_token_dict[uuid].append(i)
            else:
                # for i in self.llm.inference(text=text.to(self.device),
                #                             text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                #                             prompt_text=prompt_text.to(self.device),
                #                             prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                #                             prompt_speech_token=llm_prompt_speech_token.to(self.device),
                #                             prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                #                             embedding=llm_embedding.to(self.device)):
                #     self.tts_speech_token_dict[uuid].append(i)   
                start_id = [151936 + 6564]  
                eos_id = [151936 + 6564 + 1]
                prompt_text = prompt_text.squeeze().tolist()
                text = text.squeeze().tolist()
                llm_prompt_speech_token = llm_prompt_speech_token.squeeze().tolist()
                llm_prompt_speech_token = [id + 151936 for id in llm_prompt_speech_token]
                input_ids = start_id +  prompt_text + text + eos_id + llm_prompt_speech_token
                print(input_ids)
                stop_id = 6561 + 151936
                payload = {
                    "model": '/home/yangzy/CosyVoice-main/sft_model_concat',
                    "input_ids": input_ids,
                    "stream": True,
                    "sampling_params": {
                        "top_p": 0.8, 
                        "top_k": 25, 
                        "temperature": 1.0, 
                        "max_new_tokens": len(input_ids) * 20, 
                        "stop_token_ids": [stop_id]
                    }
                }
                for i in self._send_stream_request(payload):
                    # pdb.set_trace()
                    current_token = i["tokens"][0] - 151936
                    if current_token != 6561:
                        self.tts_speech_token_dict[uuid].append(current_token)
        self.llm_end_dict[uuid] = True

    def vc_job(self, source_speech_token, uuid):
        self.tts_speech_token_dict[uuid] = source_speech_token.flatten().tolist()
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, self.flow_cache_dict[uuid] = self.flow.inference(token=token.to(self.device),
                                                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_token=prompt_token.to(self.device),
                                                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_feat=prompt_feat.to(self.device),
                                                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                                      embedding=embedding.to(self.device),
                                                                      cache=self.flow_cache_dict[uuid],
                                                                      finalize=finalize)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                # time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)
        torch.cuda.empty_cache()


class CosyVoice2Model(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False,
                 use_flow_cache: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.use_flow_cache = use_flow_cache
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()
        # stream related params, check examples/libritts/cosyvoice2/conf/cosyvoice2.yaml
        self.token_hop_len = 25
        self.flow_decoder_required_cache_size = 0 if use_flow_cache is False else 1 * self.token_hop_len * self.flow.token_mel_ratio
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}
        self.flow_cache = self.init_flow_cache()

    def init_flow_cache(self):
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
                         'final_blocks_conv_cache': torch.zeros(10, 2, 256, 2).to(self.device)}
        if self.fp16 is True:
            for cache in [encoder_cache, decoder_cache]:
                for k, v in cache.items():
                    if isinstance(v, torch.Tensor):
                        cache[k] = v.half()
        cache = {'encoder_cache': encoder_cache, 'decoder_cache': decoder_cache}
        return cache

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4), (1, 4, 2, 0, 512, 2), (12, 4, 2, 0, 512, 2), (1, 4, 2, 0, 512, 2)]
        opt_shape = [(2, 80, 200), (2, 1, 200), (2, 80, 200), (2, 80, 200), (1, 4, 2, 100, 512, 2), (12, 4, 2, 100, 512, 2), (1, 4, 2, 100, 512, 2)]
        max_shape = [(2, 80, 1500), (2, 1, 1500), (2, 80, 1500), (2, 80, 1500), (1, 4, 2, 200, 512, 2), (12, 4, 2, 200, 512, 2), (1, 4, 2, 200, 512, 2)]
        input_names = ["x", "mask", "mu", "cond", 'down_blocks_kv_cache', 'mid_blocks_kv_cache', 'up_blocks_kv_cache']
        assert self.use_flow_cache is True, "get_trt_kwargs is set for flow cache mode. If you want to use trt with use_flow_cache=False, please set higher max_shape"
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0, is_first_chunk=False):
        start_time = time.time()
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, self.flow_cache_dict[uuid] = self.flow.inference(token=token.to(self.device),
                                                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_token=prompt_token.to(self.device),
                                                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_feat=prompt_feat.to(self.device),
                                                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                                      embedding=embedding.to(self.device),
                                                                      cache=self.flow_cache_dict[uuid],
                                                                      is_first_chunk=is_first_chunk,
                                                                      finalize=finalize)
        flow_end_time = time.time()
        logging.info(f"flow_time: {flow_end_time - start_time}")
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        hifi_end_time = time.time()
        logging.info(f"hifi_time: {hifi_end_time - flow_end_time}")
        return tts_speech

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        is_first_chunk = True
        this_uuid = str(uuid.uuid1())
        start_time = time.time()
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            # self.flow_cache_dict[this_uuid] = self.init_flow_cache()
            self.flow_cache_dict[this_uuid] = self.flow_cache
        cache_init_time = time.time()
        logging.info(f"cache_init_time: {cache_init_time - start_time}")
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        prepare_thread_time = time.time()
        logging.info(f"prepare_thread_time: {prepare_thread_time - cache_init_time}")
        if stream is True:
            assert self.use_flow_cache is True, "set use_flow_cache=True if you want to use stream inference to avoid OOM"
            # NOTE in cache mode, trim flow_prompt to same size as flow_decoder_required_cache_size
            flow_prompt_speech_token = flow_prompt_speech_token[:, -int(self.flow_decoder_required_cache_size / self.flow.token_mel_ratio):]
            prompt_speech_feat = prompt_speech_feat[:, -self.flow_decoder_required_cache_size:]
            print("buffer size:", self.token_hop_len + self.flow.pre_lookahead_len)
            while True:
                time.sleep(0.001)
                # logging.info(f"len(self.tts_speech_token_dict[this_uuid]): {len(self.tts_speech_token_dict[this_uuid])}")
                if len(self.tts_speech_token_dict[this_uuid]) >= self.token_hop_len + self.flow.pre_lookahead_len:
                    logging.info("one llm package ready")
                    llm_time = time.time()

                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:self.token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     is_first_chunk=is_first_chunk,
                                                     finalize=False)
                    flow_time = time.time()
                    logging.info(f"llm_time: {llm_time - start_time}, flow_hifi_time: {flow_time - llm_time}")
                    is_first_chunk = False
                    # NOTE in cache inference mode, we only use flow_prompt_speech_token/prompt_speech_feat in first chunk
                    flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32).to(self.device)
                    prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][self.token_hop_len:]
                    start_time = time.time()
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < self.token_hop_len + self.flow.pre_lookahead_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            assert self.use_flow_cache is False, "set use_flow_cache=False for nonstream inference"
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)
        torch.cuda.empty_cache()
