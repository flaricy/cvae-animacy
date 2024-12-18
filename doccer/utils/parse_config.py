import importlib.util
import os

def load_config(file_path):
    # 获取文件的绝对路径
    abs_path = os.path.abspath(file_path)
    
    # 确保文件存在
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    # 动态加载模块
    module_name = os.path.basename(file_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # 返回 config 字典
    return getattr(config_module, 'config', None)

