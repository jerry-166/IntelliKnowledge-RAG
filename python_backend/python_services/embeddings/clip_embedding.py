"""clip模型负责图片和文字的联合嵌入"""
import logging

import numpy as np
import torch
from PIL import Image
from typing import Union

from langchain_core.embeddings import Embeddings

from python_services.utils.image_util import ImageUtil

logger = logging.getLogger(__name__)


class CLIPEmbeddings(Embeddings):
    """
    CLIP向量嵌入模型

    支持模型：
    - openai/clip-vit-base-patch32：英文原版
    - OFA-Sys/chinese-clip-vit-base-patch16：中文CLIP（推荐）
    - OFA-Sys/chinese-clip-vit-large-patch14：中文CLIP大模型
    - BAAI/AltCLIP：多语言CLIP
    """

    MODEL_TYPES = {
        "openai/clip-vit-base-patch32": "openai",
        "openai/clip-vit-large-patch14": "openai",
        "OFA-Sys/chinese-clip-vit-base-patch16": "chinese-clip",
        "OFA-Sys/chinese-clip-vit-large-patch14": "chinese-clip",
        "OFA-Sys/chinese-clip-vit-large-patch14-336px": "chinese-clip",
        "BAAI/AltCLIP": "altclip",
        "M-CLIP/XLM-Roberta-Large-Vit-B-32": "mclip",
    }

    def __init__(
            self,
            model_name: str = "BAAI/AltCLIP",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model_type = self._detect_model_type(model_name)
        self._init_clip_model()

    def _detect_model_type(self, model_name: str) -> str:
        for key, value in self.MODEL_TYPES.items():
            if key in model_name or model_name in key:
                return value
        if "chinese" in model_name.lower():
            return "chinese-clip"
        elif "altclip" in model_name.lower():
            return "altclip"
        else:
            return "openai"

    def _init_clip_model(self):
        logger.info(f"正在加载类型为：{self.model_type} 的模型：{self.model_name}")
        if self.model_type == "chinese-clip":
            self._init_chinese_clip_model()
        elif self.model_type == "altclip":
            self._init_altclip_model()
        else:
            self._init_openai_clip_model()
        logger.info("✅️模型加载完成")

    def _init_chinese_clip_model(self):
        from transformers import ChineseCLIPProcessor, ChineseCLIPModel
        self.model = ChineseCLIPModel.from_pretrained(self.model_name, output_hidden_states=False)
        self.processor = ChineseCLIPProcessor.from_pretrained(self.model_name)
        self.tokenizer = self.processor.tokenizer  # ← 关键：保存 tokenizer 引用
        self.model.to(self.device).eval()

    def _init_altclip_model(self):
        from transformers import AltCLIPModel, AltCLIPProcessor
        self.model = AltCLIPModel.from_pretrained(self.model_name)
        self.processor = AltCLIPProcessor.from_pretrained(self.model_name)
        self.tokenizer = self.processor.tokenizer  # ← 关键：保存 tokenizer 引用
        self.model.to(self.device).eval()

    def _init_openai_clip_model(self):
        from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self.model.to(self.device).eval()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_embeddings.extend(self._embed_texts(batch))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        return self._embed_texts([query])[0]

    def embed_images(self, images: list[Union[str, bytes, Image.Image]]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            embeddings = self._embed_images(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_image(self, image: Union[str, bytes, Image.Image]) -> list[float]:
        return self._embed_images([image])[0]

    def _embed_texts(self, documents: list[str]) -> list[list[float]]:
        """批量文本嵌入核心逻辑 - 统一使用 tokenizer"""
        max_length = 77
        with torch.no_grad():
            # 统一使用 tokenizer 处理文本
            inputs = self.tokenizer(
                documents,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                # 获取文本特征
                text_outputs = self.model.text_model(**inputs)
                pooled_output = text_outputs.pooler_output

                # 检查 pooled_output 是否为 None
                if pooled_output is None:
                    # 尝试从 last_hidden_state 手动池化
                    last_hidden_state = text_outputs.last_hidden_state
                    # 使用 [CLS] token 的嵌入（第一个token）
                    pooled_output = last_hidden_state[:, 0, :]

                # 应用文本投影层
                text_features = self.model.text_projection(pooled_output)

            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy().tolist()

    def _embed_images(self, images: list[Union[str, bytes, Image.Image]]) -> list[list[float]]:
        """批量图片嵌入核心逻辑"""
        with torch.no_grad():
            images = [ImageUtil.to_pil_image(img) for img in images if img is not None]
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu().numpy().tolist()

    def compute_similarity(
            self,
            text: str,
            images: list[Union[str, bytes, Image.Image]]
    ):
        images = [ImageUtil.to_pil_image(img) for img in images]
        with torch.no_grad():
            inputs = self.processor(images=images, text=text, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            # probs = torch.sigmoid(logits_per_image).cpu().numpy().squeeze().tolist()
            # # 确保返回列表
            # return [probs] if isinstance(probs, float) else probs

            logits_np = logits_per_image.cpu().numpy().squeeze()  # 转为numpy数组

            # 核心修复：用softmax做相对概率归一化（总和=1，能区分匹配度）
            # 先缩放logits（可选，增强区分度）
            scaled_logits = logits_np
            probs = torch.softmax(torch.tensor(scaled_logits), dim=0).numpy()

            # 打印调试信息：原生logit + 相对概率
            for logit, prob in zip(logits_np, probs):
                logger.info(f"原生logit：{logit:.4f} | 相对相似度：{prob:.4f}")

            # 保留4位小数，避免显示1.0
            return [round(float(p), 4) for p in probs]
