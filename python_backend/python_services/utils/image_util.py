import base64
import io
import os.path
from typing import Optional, Union

import requests
from PIL import Image
from PIL.ImagePath import Path


class ImageUtil:
    """图片工具类（增强本地图片加载+详细日志）"""

    @staticmethod
    def load_image(image_input: str, base_dir: Optional[Path] = None) -> Optional[Image.Image]:
        """
        加载图片，直接返回RGB模式的PIL.Image（避免字节流转换错误）
        :param image_input: 本地路径/URL/Base64字符串
        :param base_dir: 相对路径的基础目录
        :return: RGB模式PIL.Image，失败返回None（带详细日志）
        """
        try:
            print(f"\n=== 开始加载图片: {image_input} ===")
            # 1. 处理URL图片
            if image_input.startswith(('http://', 'https://')):
                print(f"类型：URL，开始下载...")
                response = requests.get(
                    image_input, 
                    timeout=15, 
                    stream=True,
                    headers={"User-Agent": "Mozilla/5.0"}  # 模拟浏览器，避免部分URL拒绝访问
                )
                response.raise_for_status()  # 抛出HTTP错误（404/500等）
                with Image.open(response.raw) as img:
                    rgb_img = img.convert("RGB")
                    print(f"URL图片加载成功，格式：{img.format}")
                    return rgb_img

            # 2. 处理Base64图片
            if image_input.startswith('data:image'):
                print(f"类型：Base64，开始解码...")
                base64_data = image_input.split(',')[1]
                img_bytes = base64.b64decode(base64_data)
                with Image.open(io.BytesIO(img_bytes)) as img:
                    rgb_img = img.convert("RGB")
                    print(f"Base64图片解码成功，格式：{img.format}")
                    return rgb_img

            # 3. 处理本地图片（重点修复）
            print(f"类型：本地路径，开始处理...")
            # 处理file:///协议
            if image_input.startswith('file:///'):
                full_path = image_input[8:]
            elif base_dir:
                full_path = os.path.join(base_dir, image_input)
            else:
                full_path = image_input

            # 标准化路径（处理Windows反斜杠、相对路径→绝对路径）
            full_path = os.path.abspath(full_path)
            print(f"标准化后路径：{full_path}")

            # 检查路径是否存在（核心验证）
            if not os.path.exists(full_path):
                print(f"⚠️  路径不存在，尝试添加常见后缀...")
                found = False
                # 仅当原路径无后缀时才尝试添加（避免重复后缀，如.png→.png.png）
                if not os.path.splitext(full_path)[1]:
                    for suffix in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']:
                        test_path = full_path + suffix
                        if os.path.exists(test_path):
                            full_path = test_path
                            found = True
                            break
                if not found:
                    print(f"❌ 所有后缀尝试失败，文件不存在：{full_path}")
                    return None
            else:
                print(f"✅ 路径存在，文件大小：{os.path.getsize(full_path) / 1024:.1f}KB")

            # 打开图片并强制转为RGB（解决RGBA/P模式不兼容问题）
            with Image.open(full_path) as img:
                print(f"图片打开成功，原格式：{img.format}，原模式：{img.mode}")
                rgb_img = img.convert("RGB")  # 强制转RGB，适配CLIP模型
                print(f"✅ 图片加载完成（已转为RGB模式）")
                return rgb_img

        # 细分异常类型，精准定位问题
        except requests.exceptions.RequestException as e:
            print(f"❌ URL加载失败：{str(e)}")
        except base64.binascii.Error as e:
            print(f"❌ Base64解码失败：{str(e)}")
        except PermissionError:
            print(f"❌ 权限不足：无法访问文件 {full_path}（请以管理员身份运行）")
        except FileNotFoundError:
            print(f"❌ 文件不存在：{full_path}")
        except Image.UnidentifiedImageError:
            print(f"❌ 图片损坏或格式不支持：{full_path}（PIL无法识别）")
        except Exception as e:
            print(f"❌ 未知错误：{str(e)}，错误类型：{type(e).__name__}")
        return None

    @classmethod
    def to_pil_image(cls, image: Union[str, bytes, Image.Image]) -> Image.Image:
        """统一转换为RGB格式PIL.Image（兼容多种输入类型）"""
        if isinstance(image, Image.Image):
            print(f"输入类型：PIL.Image，直接转为RGB模式")
            return image.convert("RGB")
        elif isinstance(image, bytes):
            print(f"输入类型：字节流，转换为PIL.Image")
            with Image.open(io.BytesIO(image)) as img:
                return img.convert("RGB")
        elif isinstance(image, str):
            # 调用load_image加载（支持本地路径/URL/Base64）
            pil_img = cls.load_image(image)
            if pil_img is None:
                raise ValueError(f"无法加载图片：{image}（详细错误见上文日志）")
            return pil_img
        else:
            raise ValueError(f"不支持的输入类型：{type(image)}（请传入路径/字节流/PIL.Image）")