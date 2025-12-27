# apply_fixes.py
import os

# 修复1: 创建符号链接，让程序能找到模型文件
print("修复模型路径问题...")
if os.path.exists("results/models") and not os.path.exists("models"):
    # 在Windows上创建目录链接
    os.system('mklink /J models results/models')
    print("已创建目录链接: models -> results/models")

# 修复2: 检查JSON编码器是否已添加
print("\n检查main.py中的JSON编码器...")
with open("main.py", "r", encoding="utf-8") as f:
    content = f.read()
    
if "class NumpyEncoder" not in content:
    print("需要添加NumpyEncoder类到main.py")
    
    # 在import部分之后添加
    import_section_end = content.find("\n\n", content.find("import "))
    if import_section_end == -1:
        import_section_end = content.find("\n", 100)
    
    encoder_code = '''
class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理NumPy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NumpyEncoder, self).default(obj)
'''
    
    new_content = content[:import_section_end] + encoder_code + content[import_section_end:]
    
    # 保存修改
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("已添加NumpyEncoder类")
else:
    print("NumpyEncoder类已存在")

print("\n修复完成!")