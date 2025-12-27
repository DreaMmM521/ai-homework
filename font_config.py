# font_config.py
import matplotlib
import matplotlib.pyplot as plt
import os

def setup_chinese_font():
    """设置中文字体"""
    try:
        # 方法1: 使用系统字体
        font_names = [
            'SimHei',           # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'KaiTi',            # 楷体
            'FangSong',         # 仿宋
            'STXihei',          # 华文细黑
            'STKaiti',          # 华文楷体
            'STSong',           # 华文宋体
            'STFangsong',       # 华文仿宋
            'Arial Unicode MS', # Mac/Linux
            'DejaVu Sans'       # Linux
        ]
        
        # 检查哪些字体可用
        available_fonts = matplotlib.font_manager.fontManager.get_font_names()
        for font in font_names:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 使用中文字体: {font}")
                return True
        
        # 方法2: 如果上述字体都不可用，尝试使用字体文件
        if os.name == 'nt':  # Windows
            font_paths = [
                'C:/Windows/Fonts/simhei.ttf',
                'C:/Windows/Fonts/msyh.ttc',
                'C:/Windows/Fonts/simkai.ttf'
            ]
        else:  # Linux/Mac
            font_paths = [
                '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
                '/System/Library/Fonts/PingFang.ttc'
            ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                matplotlib.font_manager.fontManager.addfont(font_path)
                font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 从文件加载中文字体: {font_name}")
                return True
        
        print("⚠ 未找到合适的中文字体，使用默认字体")
        return False
        
    except Exception as e:
        print(f"⚠ 设置中文字体时出错: {e}")
        return False

# 在程序开始时调用
if __name__ == "__main__":
    setup_chinese_font()
    print("中文字体设置完成")