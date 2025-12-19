# image_util.py
import base64
import io
import os.path
from typing import Optional, Union, Tuple

import requests
from PIL import Image
from pathlib import Path


class ImageUtil:
    """图片工具类（修复版）"""

    @staticmethod
    def load_image(
            image_input: str,
            base_dir: Optional[Path] = None,
            mode: Optional[str] = None
    ) -> Optional[Image.Image]:
        """
        加载图片，返回PIL.Image
        :param mode: 可选的颜色模式（如 "RGB"）
        :param image_input: 本地路径/URL/Base64字符串
        :param base_dir: 相对路径的基础目录
        :return: PIL.Image，失败返回None
        """
        try:
            print(f"\n=== 开始加载图片: {image_input[:100]}... ===")

            # 1. 处理URL图片
            if image_input.startswith(('http://', 'https://')):
                print(f"类型：URL，开始下载...")
                response = requests.get(
                    image_input,
                    timeout=15,
                    stream=True,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                response.raise_for_status()
                img = Image.open(response.raw)
                img.load()

            # 2. 处理Base64图片
            elif image_input.startswith('data:image'):
                print(f"类型：Base64，开始解码...")
                base64_data = image_input.split(',')[1]
                img_bytes = base64.b64decode(base64_data)
                img = Image.open(io.BytesIO(img_bytes))
                img.load()

            # 3. 处理本地图片
            else:
                print(f"类型：本地路径，开始处理...")
                if image_input.startswith('file:///'):
                    full_path = image_input[8:]
                elif base_dir:
                    full_path = os.path.join(base_dir, image_input)
                else:
                    full_path = image_input

                full_path = os.path.abspath(full_path)
                print(f"标准化后路径：{full_path}")

                if not os.path.exists(full_path):
                    print(f"⚠️ 路径不存在，尝试添加常见后缀...")
                    found = False
                    if not os.path.splitext(full_path)[1]:
                        for suffix in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']:
                            test_path = full_path + suffix
                            if os.path.exists(test_path):
                                full_path = test_path
                                found = True
                                break
                    if not found:
                        print(f"❌ 文件不存在：{full_path}")
                        return None

                print(f"✅ 路径存在，文件大小：{os.path.getsize(full_path) / 1024:.1f}KB")
                img = Image.open(full_path)
                img.load()

            # 转换颜色模式
            if mode:
                img = img.convert(mode)

            print(f"✅ 图片加载成功，格式：{img.format}，模式：{img.mode}，尺寸：{img.size}")
            return img

        except Exception as e:
            print(f"❌ 图片加载失败：{type(e).__name__}: {e}")
            return None

    @staticmethod
    def image_to_bytes(image: Image.Image, format: str = "PNG", quality: int = 95) -> Tuple[bytes, str]:
        """
        🔥 核心修复：将PIL.Image转换为图片文件字节流（PNG/JPEG格式）

        :param image: PIL.Image对象
        :param format: 输出格式 "PNG" 或 "JPEG"
        :param quality: JPEG质量（1-100）
        :return: (图片字节流, 实际格式)
        """
        buffer = io.BytesIO()

        # 处理RGBA模式（PNG支持透明，JPEG不支持）
        actual_format = format.upper()

        if actual_format == "JPEG" and image.mode in ('RGBA', 'LA', 'P'):
            # JPEG不支持透明通道，转换为RGB
            print(f"⚠️ JPEG不支持{image.mode}模式，转换为RGB...")
            if image.mode == 'RGBA':
                # 创建白色背景
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            else:
                image = image.convert('RGB')

        # 保存到字节流
        if actual_format == "JPEG":
            image.save(buffer, format="JPEG", quality=quality)
        else:
            image.save(buffer, format="PNG")

        result_bytes = buffer.getvalue()
        print(f"✅ 图片转换成功：{actual_format}格式，{len(result_bytes) / 1024:.1f}KB")

        return result_bytes, actual_format.lower()

    @staticmethod
    def image_to_base64(image: Image.Image, format: str = "PNG") -> Tuple[str, str]:
        """
        将PIL.Image转换为Base64字符串

        :return: (base64字符串, 格式)
        """
        img_bytes, actual_format = ImageUtil.image_to_bytes(image, format)
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        return base64_str, actual_format

    @classmethod
    def to_pil_image(cls, image: Union[str, bytes, Image.Image]) -> Image.Image:
        """统一转换为RGB格式PIL.Image"""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, bytes):
            with Image.open(io.BytesIO(image)) as img:
                return img.convert("RGB")
        elif isinstance(image, str):
            pil_img = cls.load_image(image, mode="RGB")
            if pil_img is None:
                raise ValueError(f"无法加载图片：{image}")
            return pil_img
        else:
            raise ValueError(f"不支持的输入类型：{type(image)}")