"""
image_util
加载、转base64...
"""
import base64
import os.path

import requests
from PIL import Image
from PIL.ImagePath import Path


class ImageUtil:
    """
    图片工具类
    """
    @staticmethod
    def load_image(image_path: str, base_dir: Path) -> bytes | None:
        """
        加载图片
        :param image_path: 图片路径
        :param base_dir: 图片所在目录
        :return: 图片字节流
        """
        try:
            # 处理url
            if image_path.startswith(('http://', 'https://')):
                response = requests.request('get', image_path)
                # 检查响应状态码
                response.raise_for_status()
                return response.content

            # 处理base64（data:image/[格式,e.g. png];base64,[base64编码]）
            if image_path.startswith('data:image'):
                base64_data = image_path.split(',')[1]
                return base64.b64decode(base64_data)

            # 处理本地图片
            # 处理 file:/// 协议的本地路径
            if image_path.startswith('file:///'):
                # 移除 file:/// 前缀
                full_path = image_path[8:]  # 从第8个字符开始
            else:
                # 处理相对路径
                full_path = os.path.join(base_dir, image_path)

            if not os.path.exists(full_path):
                # 尝试其他常见路径（修复原代码逻辑错误）
                found = False
                for suffix in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    test_path = full_path + suffix
                    if os.path.exists(test_path):
                        full_path = test_path
                        found = True
                        break
                if not found:
                    raise FileNotFoundError(f"图片文件不存在: {full_path}")

            image = Image.open(full_path)
            # img_bytes = image.tobytes()
            # 修复：使用正确的字节流转换方法
            from io import BytesIO
            img_buffer = BytesIO()
            image.save(img_buffer, format=image.format)
            return img_buffer.getvalue()
        except Exception as e:
            print(f"处理图片失败 {image_path}: {e}")
            return None
