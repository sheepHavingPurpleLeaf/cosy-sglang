import logging
import os
import subprocess
import re
import argparse

def get_gpu_info():
    """获取GPU信息并返回相关参数"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, check=True)
        line = result.stdout.strip()
        if line:
            parts = line.split(', ')
            gpu_name = parts[0].strip()
            memory_mb = int(parts[1])
            compute_cap = float(parts[2])
            return gpu_name, memory_mb, compute_cap
    except:
        logging.warning("无法获取GPU信息，使用默认设置")
        return "Unknown GPU", 8192, 8.6
    
    return "Unknown GPU", 8192, 8.6

def get_gpu_optimized_config(gpu_name, memory_mb, compute_cap):
    """根据GPU型号返回优化配置"""
    config = {
        'workspace_size': 1 << 33,  # 默认8GB
        'fp16_enabled': True,
        'int8_enabled': False,
        'fp8_enabled': False,  # 新增FP8支持
        'dla_core': None,
        'gpu_suffix': 'unknown',
        'best_precision': 'fp16'  # 默认推荐精度
    }
    
    # 检查FP8硬件支持 - 需要计算能力8.9+（Ada Lovelace/Hopper架构）
    fp8_hardware_support = compute_cap >= 8.9
    
    # 检查TensorRT FP8软件支持
    fp8_software_support = False
    try:
        import tensorrt as trt
        version = trt.__version__
        version_parts = version.split('.')
        major_ver = int(version_parts[0])
        minor_ver = int(version_parts[1])
        
        # TensorRT 8.6+ 支持FP8
        version_ok = major_ver > 8 or (major_ver == 8 and minor_ver >= 6)
        fp8_flag_available = hasattr(trt.BuilderFlag, 'FP8')
        fp8_software_support = version_ok and fp8_flag_available
        
        logging.info(f"TensorRT版本: {version}, FP8软件支持: {fp8_software_support}")
    except ImportError:
        logging.warning("TensorRT未安装，无法检查FP8支持")
    
    # 综合判断FP8可用性
    config['fp8_enabled'] = fp8_hardware_support and fp8_software_support
    
    # RTX 30系列优化
    if 'RTX 3060' in gpu_name:
        config.update({
            'workspace_size': 1 << 32,  # 4GB (RTX 3060显存较小)
            'gpu_suffix': '3060',
            'best_precision': 'fp16'
        })
    elif 'RTX 3070' in gpu_name:
        config.update({
            'workspace_size': 1 << 33,  # 8GB
            'gpu_suffix': '3070',
            'best_precision': 'fp16'
        })
    elif 'RTX 3080' in gpu_name:
        config.update({
            'workspace_size': 1 << 34,  # 16GB
            'gpu_suffix': '3080',
            'best_precision': 'fp16'
        })
    elif 'RTX 3090' in gpu_name:
        config.update({
            'workspace_size': 1 << 34,  # 16GB
            'gpu_suffix': '3090',
            'best_precision': 'fp16'
        })
    
    # RTX 40系列优化 - 支持FP8
    elif 'RTX 4060' in gpu_name:
        config.update({
            'workspace_size': 1 << 32,  # 4GB
            'gpu_suffix': '4060',
            'best_precision': 'fp8' if config['fp8_enabled'] else 'fp16'
        })
    elif 'RTX 4070' in gpu_name:
        config.update({
            'workspace_size': 1 << 33,  # 8GB
            'gpu_suffix': '4070',
            'best_precision': 'fp8' if config['fp8_enabled'] else 'fp16'
        })
    elif 'RTX 4080' in gpu_name:
        config.update({
            'workspace_size': 1 << 34,  # 16GB
            'gpu_suffix': '4080',
            'best_precision': 'fp8' if config['fp8_enabled'] else 'fp16'
        })
    elif 'RTX 4090' in gpu_name:
        config.update({
            'workspace_size': 1 << 35,  # 32GB
            'gpu_suffix': '4090',
            'best_precision': 'fp8' if config['fp8_enabled'] else 'fp16'
        })
    
    # Tesla/专业卡优化 - H100等支持FP8
    elif 'Tesla' in gpu_name or 'A100' in gpu_name or 'V100' in gpu_name or 'H100' in gpu_name or 'A800' in gpu_name:
        config.update({
            'workspace_size': 1 << 35,  # 32GB
            'int8_enabled': True,
            'gpu_suffix': 'tesla'
        })
        # H100/A800等高端卡优先使用FP8
        if 'H100' in gpu_name or 'A800' in gpu_name:
            config['best_precision'] = 'fp8' if config['fp8_enabled'] else 'fp16'
        else:
            config['best_precision'] = 'fp16'
    
    # 根据显存调整workspace - 修复RTX 3070的问题
    if memory_mb < 6144:  # <6GB
        config['workspace_size'] = min(config['workspace_size'], 1 << 31)  # 最大2GB
    elif memory_mb < 8192:  # <8GB 但不包括8192MB（RTX 3070正好8GB）
        config['workspace_size'] = min(config['workspace_size'], 1 << 32)  # 最大4GB
    elif memory_mb < 16384:  # <16GB
        config['workspace_size'] = min(config['workspace_size'], 1 << 33)  # 最大8GB
    
    logging.info(f"GPU: {gpu_name}, 显存: {memory_mb}MB, 计算能力: {compute_cap}")
    logging.info(f"FP8硬件支持: {fp8_hardware_support}, FP8软件支持: {fp8_software_support}")
    logging.info(f"推荐精度: {config['best_precision'].upper()}")
    logging.info(f"优化配置: workspace={config['workspace_size']//1024//1024//1024}GB, FP16={config['fp16_enabled']}")
    
    return config

def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, precision='auto', gpu_config=None):
    import tensorrt as trt
    logging.info("Converting onnx to trt...")
    
    # 获取GPU配置
    if gpu_config is None:
        gpu_name, memory_mb, compute_cap = get_gpu_info()
        gpu_config = get_gpu_optimized_config(gpu_name, memory_mb, compute_cap)
    else:
        gpu_name = gpu_config.get('gpu_name', 'Unknown GPU')
    
    # 自动选择最佳精度
    if precision == 'auto':
        precision = gpu_config['best_precision']
        logging.info(f"自动选择精度: {precision.upper()}")
    
    # 验证精度支持
    if precision == 'fp8' and not gpu_config['fp8_enabled']:
        logging.warning(f"GPU不支持FP8，回退到FP16")
        precision = 'fp16'
    elif precision == 'int8' and not gpu_config['int8_enabled']:
        logging.warning(f"GPU配置不推荐INT8，回退到FP16")
        precision = 'fp16'
    
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # 使用GPU优化的配置
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, gpu_config['workspace_size'])
    
    # RTX 3070特别优化
    if 'RTX 3070' in gpu_config.get('gpu_name', ''):
        # 启用更激进的优化策略
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)  # 严格类型转换
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)  # 偏好精度约束
        # 针对RTX 3070的SM86架构优化
        config.default_device_type = trt.DeviceType.GPU
        logging.info("启用RTX 3070特定优化")
    
    # 设置精度标志
    if precision == 'fp16' and gpu_config['fp16_enabled']:
        config.set_flag(trt.BuilderFlag.FP16)
        logging.info("启用FP16优化")
    elif precision == 'fp8' and gpu_config['fp8_enabled']:
        config.set_flag(trt.BuilderFlag.FP8)
        logging.info("启用FP8优化")
    elif precision == 'int8' and gpu_config['int8_enabled']:
        config.set_flag(trt.BuilderFlag.INT8)
        logging.info("启用INT8优化")
    elif precision == 'fp32':
        logging.info("使用FP32精度")
    else:
        logging.warning(f"不支持的精度 {precision}，使用FP32")
        precision = 'fp32'
    
    # 启用更多优化选项
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)  # 禁用timing cache以获得最新优化
    
    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('failed to parse {}'.format(onnx_model))
    
    # Check if engine_bytes is None before proceeding
    if len(trt_kwargs['input_names']) != len(trt_kwargs['min_shape']):
        raise ValueError(f"Mismatch: {len(trt_kwargs['input_names'])} input names vs {len(trt_kwargs['min_shape'])} shapes")
    
    # set input shapes
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(trt_kwargs['input_names'][i], trt_kwargs['min_shape'][i], trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    
    # 根据精度设置数据类型
    if precision == 'fp8':
        try:
            tensor_dtype = trt.DataType.FP8
        except AttributeError:
            logging.warning("当前TensorRT版本不支持FP8数据类型，回退到FP16")
            tensor_dtype = trt.DataType.HALF
    elif precision == 'fp16':
        tensor_dtype = trt.DataType.HALF
    elif precision == 'int8':
        tensor_dtype = trt.DataType.INT8
    else:  # fp32
        tensor_dtype = trt.DataType.FLOAT
    
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    config.add_optimization_profile(profile)
    
    # 添加GPU配置到返回值中以便调试
    gpu_config['gpu_name'] = gpu_name if 'gpu_name' not in gpu_config else gpu_config['gpu_name']
    
    engine_bytes = builder.build_serialized_network(network, config)
    
    # Check if build was successful
    if engine_bytes is None:
        raise RuntimeError("Failed to build TensorRT engine. Check the input configuration and ONNX model compatibility.")
    
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Successfully convert onnx to trt...")
    
    return gpu_config

def get_trt_kwargs_cache_enabled():
    """Configuration for cache-enabled ONNX model (CosyVoice2)"""
    # Based on actual ONNX model inspection:
    # Input 0: x, Shape: [2, 80, 'seq_len']
    # Input 1: mask, Shape: [2, 1, 'seq_len']  
    # Input 2: mu, Shape: [2, 80, 'seq_len']
    # Input 3: t, Shape: [2]
    # Input 4: spks, Shape: [2, 80]
    # Input 5: cond, Shape: [2, 80, 'seq_len']
    # Input 6: down_blocks_conv_cache, Shape: [1, 2, 832, 2]
    # Input 7: down_blocks_kv_cache, Shape: [1, 4, 2, 'cache_in_len', 512, 2]
    # Input 8: mid_blocks_conv_cache, Shape: [12, 2, 512, 2]
    # Input 9: mid_blocks_kv_cache, Shape: [12, 4, 2, 'cache_in_len', 512, 2]
    # Input 10: up_blocks_conv_cache, Shape: [1, 2, 1024, 2]
    # Input 11: up_blocks_kv_cache, Shape: [1, 4, 2, 'cache_in_len', 512, 2]
    # Input 12: final_blocks_conv_cache, Shape: [2, 256, 2]
    
    min_shape = [
        (2, 80, 2),           # x
        (2, 1, 2),            # mask
        (2, 80, 2),           # mu  
        (2,),                 # t
        (2, 80),              # spks
        (2, 80, 2),           # cond
        (1, 2, 832, 2),       # down_blocks_conv_cache (static)
        (1, 4, 2, 2, 512, 2), # down_blocks_kv_cache
        (12, 2, 512, 2),      # mid_blocks_conv_cache (static)
        (12, 4, 2, 2, 512, 2), # mid_blocks_kv_cache
        (1, 2, 1024, 2),      # up_blocks_conv_cache (static)
        (1, 4, 2, 2, 512, 2), # up_blocks_kv_cache
        (2, 256, 2),          # final_blocks_conv_cache (static)
    ]
    
    opt_shape = [
        (2, 80, 200),           # x
        (2, 1, 200),            # mask
        (2, 80, 200),           # mu
        (2,),                   # t
        (2, 80),                # spks
        (2, 80, 200),           # cond
        (1, 2, 832, 2),         # down_blocks_conv_cache (static)
        (1, 4, 2, 50, 512, 2), # down_blocks_kv_cache
        (12, 2, 512, 2),        # mid_blocks_conv_cache (static)
        (12, 4, 2, 50, 512, 2), # mid_blocks_kv_cache
        (1, 2, 1024, 2),        # up_blocks_conv_cache (static)
        (1, 4, 2, 50, 512, 2), # up_blocks_kv_cache
        (2, 256, 2),            # final_blocks_conv_cache (static)
    ]
    
    max_shape = [
        (2, 80, 1000),           # x
        (2, 1, 1000),            # mask
        (2, 80, 1000),           # mu
        (2,),                    # t
        (2, 80),                 # spks
        (2, 80, 1000),           # cond
        (1, 2, 832, 2),          # down_blocks_conv_cache (static)
        (1, 4, 2, 50, 512, 2), # down_blocks_kv_cache
        (12, 2, 512, 2),         # mid_blocks_conv_cache (static)
        (12, 4, 2, 50, 512, 2), # mid_blocks_kv_cache
        (1, 2, 1024, 2),         # up_blocks_conv_cache (static)
        (1, 4, 2, 50, 512, 2), # up_blocks_kv_cache
        (2, 256, 2),             # final_blocks_conv_cache (static)
    ]
    
    input_names = [
        "x", "mask", "mu", "t", "spks", "cond", 
        "down_blocks_conv_cache", "down_blocks_kv_cache", 
        "mid_blocks_conv_cache", "mid_blocks_kv_cache",
        "up_blocks_conv_cache", "up_blocks_kv_cache", 
        "final_blocks_conv_cache"
    ]
    
    return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

def get_trt_kwargs():
    """Configuration for standard ONNX model (CosyVoice)"""
    min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
    opt_shape = [(2, 80, 200), (2, 1, 200), (2, 80, 200), (2, 80, 200)]
    max_shape = [(2, 80, 4000), (2, 1, 4000), (2, 80, 4000), (2, 80, 4000)]
    input_names = ["x", "mask", "mu", "cond"]
    return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

if __name__ == "__main__":
    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description='ONNX to TensorRT conversion with automatic precision selection')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'fp16', 'fp8', 'int8'], 
                       default='auto', help='目标精度 (默认: auto - 自动选择最佳)')
    parser.add_argument('--check-gpu', action='store_true', help='仅检查GPU信息并显示推荐精度')
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 获取GPU信息
    gpu_name, memory_mb, compute_cap = get_gpu_info()
    gpu_config = get_gpu_optimized_config(gpu_name, memory_mb, compute_cap)
    
    # 如果只是检查GPU信息
    if args.check_gpu:
        print("\n" + "="*70)
        print("🎮 GPU 信息与精度推荐")
        print("="*70)
        print(f"GPU型号: {gpu_name}")
        print(f"显存大小: {memory_mb} MB ({memory_mb/1024:.1f} GB)")
        print(f"计算能力: {compute_cap}")
        print(f"FP8硬件支持: {'✅ 支持' if compute_cap >= 8.9 else '❌ 不支持 (需要8.9+)'}")
        print(f"FP8软件支持: {'✅ 支持' if gpu_config['fp8_enabled'] else '❌ 不支持'}")
        print(f"推荐精度: {gpu_config['best_precision'].upper()} ⭐")
        
        print(f"\n💡 精度说明:")
        print(f"  • FP8: 最小模型体积，需要RTX 40系列/H100等")
        print(f"  • FP16: 平衡性能与精度，推荐选择")
        print(f"  • FP32: 最高精度，模型体积较大")
        print(f"  • INT8: 需要校准数据集")
        print("="*70)
        exit(0)
    
    model_dir = "/home/yangzy/nas/pretrained_models/ca_cosyvoice"
    onnx_model = os.path.join(model_dir, "flow.decoder.estimator.fp32.onnx")
    
    # 确定最终使用的精度
    if args.precision == 'auto':
        final_precision = gpu_config['best_precision']
        logging.info(f"🤖 自动选择精度: {final_precision.upper()}")
    else:
        final_precision = args.precision
        logging.info(f"🎯 手动指定精度: {final_precision.upper()}")
    
    # 验证精度可用性
    if final_precision == 'fp8' and not gpu_config['fp8_enabled']:
        logging.error("❌ 当前环境不支持FP8转换")
        if compute_cap < 8.9:
            logging.info("💡 您的GPU计算能力为 %.1f，需要 8.9+ 才支持FP8", compute_cap)
        else:
            logging.info("💡 请检查TensorRT版本是否 >= 8.6")
        logging.info("🔄 自动回退到FP16精度")
        final_precision = 'fp16'
    
    # 根据最终精度生成TRT文件名
    trt_filename = f"flow.decoder.estimator.{final_precision}.{gpu_config['gpu_suffix']}.plan"
    trt_model = os.path.join(model_dir, trt_filename)
    
    logging.info(f"📁 输入ONNX模型: {onnx_model}")
    logging.info(f"📁 输出TRT模型: {trt_model}")
    logging.info(f"🎯 使用精度: {final_precision.upper()}")
    
    # 使用cache-enabled配置进行转换
    try:
        final_config = convert_onnx_to_trt(trt_model, get_trt_kwargs_cache_enabled(), onnx_model, final_precision, gpu_config)
        
        logging.info(f"✅ TensorRT转换完成！")
        logging.info(f"💾 模型已保存为: {trt_model}")
        
        # 显示文件大小和性能预期
        if os.path.exists(trt_model):
            file_size = os.path.getsize(trt_model) / 1024 / 1024  # MB
            logging.info(f"📊 TRT模型文件大小: {file_size:.1f} MB")
            
            # 性能预期说明
            if final_precision == 'fp8':
                logging.info("🚀 FP8模型: 最高推理速度，最小显存占用")
            elif final_precision == 'fp16':
                logging.info("⚡ FP16模型: 良好的速度与精度平衡")
            elif final_precision == 'fp32':
                logging.info("🎯 FP32模型: 最高精度，较大显存占用")
                
    except Exception as e:
        logging.error(f"❌ 转换失败: {e}")
        logging.info("💡 建议:")
        logging.info("  1. 检查ONNX模型是否存在且有效")
        logging.info("  2. 确保有足够的显存空间")
        logging.info("  3. 尝试降低精度或减小batch size")
        raise