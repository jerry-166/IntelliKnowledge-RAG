"""clip模型负责图片和文字的联合嵌入"""
import numpy as np
import torch
from PIL import Image
from typing import Union

from python_services.utils.image_util import ImageUtil


class ClipEmbeddings(object):
    """CLIP向量嵌入模型（支持文本/图片跨模态检索）"""
    def __init__(
            self,
            model_name: str = "openai/clip-vit-base-patch32",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._init_clip_model()

    def _init_clip_model(self):
        from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device).eval()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文本"""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_embeddings.extend(self._embed_texts(batch))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """嵌入单个文本查询"""
        return self._embed_texts([query])[0]

    def embed_images(self, images: list[Union[str, bytes, Image.Image]]) -> list[list[float]]:
        """批量嵌入图片（过滤无效图片）"""
        all_embeddings = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            # 转换为PIL.Image并过滤无效图片（添加异常捕获）
            valid_images = []
            for img in batch:
                if img is None:
                    continue
                try:
                    pil_img = ImageUtil.to_pil_image(img)
                    valid_images.append(pil_img)
                except Exception as e:
                    print(f"⚠️  跳过无效图片 {img}，错误：{str(e)}")
                    continue
            if not valid_images:
                print(f"⚠️  当前batch无有效图片，跳过")
                continue
            # 嵌入有效图片
            embeddings = self._embed_images(valid_images)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_image(self, image: Union[str, bytes, Image.Image]) -> list[float]:
        """嵌入单个图片"""
        return self.embed_images([image])[0]

    def _embed_texts(self, documents: list[str]) -> list[list[float]]:
        """批量文本嵌入核心逻辑"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text=documents,
                padding=True,
                truncation=True,
                max_length=77,  # CLIP模型固定最大文本长度
                return_tensors="pt",
            ).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # L2归一化
            return text_features.cpu().numpy().tolist()

    def _embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        """批量图片嵌入核心逻辑（适配RGB模式PIL.Image）"""
        with torch.no_grad():
            # 直接传入RGB模式的PIL.Image列表，processor自动处理尺寸
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,    # 批量处理时自动填充
                truncation=False,
                do_resize=True,  # 强制缩放为模型要求的尺寸（224x224）
                do_center_crop=True,  # 中心裁剪（CLIP模型要求）
            ).to(self.device)
            image_features = self.model.get_image_features(** inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # L2归一化
            return image_features.cpu().numpy().tolist()

    def compute_similarity(
            self,
            text: str,
            image: Union[str, bytes, Image.Image],
    ) -> float:
        """计算文本-图片相似度（点积，范围[-1,1]）"""
        text_embed = np.array(self.embed_query(text)).squeeze()
        image_embed = np.array(self.embed_image(image)).squeeze()
        return float(np.dot(text_embed, image_embed))