"""
See original implementation (quite far from this one)
at https://github.com/jakesnell/prototypical-networks
"""
from random import random

from torch import Tensor
from abc import abstractmethod
from typing import Optional

from easyfsl.methods.utils import power_transform
import torch
from torch import Tensor, nn

from easyfsl.methods.utils import compute_prototypes
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

from .few_shot_classifier import FewShotClassifier

from transformers import BertTokenizer, BertModel
class VectorDecoder(nn.Module):
    # 向量解码器类，用于将输入向量解码为BERT嵌入表示。
    def __init__(self, input_dim=768, bert_dim=768):
        super(VectorDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, bert_dim)
        # 加载预训练的BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, x):
        # Encode input vector
        encoded_input = self.fc(x).unsqueeze(1)
        with torch.no_grad():
            bert_output = self.bert_model(inputs_embeds=encoded_input)
        return bert_output.last_hidden_state.squeeze(1)



class S2P(nn.Module):

    def __init__(
            self,
            backbone: Optional[nn.Module] = None,
            use_softmax: bool = False,
            feature_centering: Optional[Tensor] = None,
            feature_normalization: Optional[float] = None,

    ):
        super().__init__()

        self.backbone = backbone if backbone is not None else nn.Identity()
        self.use_softmax = use_softmax

        self.prototypes = torch.tensor(())
        self.support_features = torch.tensor(())
        self.support_labels = torch.tensor(())

        self.feature_centering = (
            feature_centering if feature_centering is not None else torch.tensor(0)
        )
        self.feature_normalization = feature_normalization
        self.text_decoder = VectorDecoder()

    def rectify_prototypes(
        self, query_features: Tensor
    ):  # pylint: disable=not-callable
        """
        Updates prototypes with label propagation and feature shifting.
        Args:
            query_features: query features of shape (n_query, feature_dimension)
        """
        n_classes = self.support_labels.unique().size(0)
        one_hot_support_labels = nn.functional.one_hot(self.support_labels, n_classes)
        # 计算支持集和查询集的平均特征差
        average_support_query_shift = self.support_features.mean(
            0, keepdim=True
        ) - query_features.mean(0, keepdim=True)
        query_features = query_features + average_support_query_shift       # F'
        # 计算支持集和查询集到原型的余弦距离
        support_logits = self.cosine_distance_to_prototypes(self.support_features).exp()
        query_logits = self.cosine_distance_to_prototypes(query_features).exp()
        # 构建归一化向量
        one_hot_query_prediction = nn.functional.one_hot(
            query_logits.argmax(-1), n_classes
        )

        normalization_vector = (
            (one_hot_support_labels * support_logits).sum(0)
            + (one_hot_query_prediction * query_logits).sum(0)
        ).unsqueeze(
            0
        )  # [1, n_classes]
        support_reweighting = (
            one_hot_support_labels * support_logits
        ) / normalization_vector  # [n_support, n_classes]
        query_reweighting = (
            one_hot_query_prediction * query_logits
        ) / normalization_vector  # [n_query, n_classes]

        self.prototypes = (support_reweighting * one_hot_support_labels).t().matmul(
            self.support_features
        ) + (query_reweighting * one_hot_query_prediction).t().matmul(query_features)

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:

        # Extract the features of query images
        query_features = self.compute_features(query_images)
        # 原型修正策略
        self.rectify_prototypes(query_features=query_features)

        # 语义引导增强
        query_features = self.text_decoder(query_features).to('cuda') + query_features.to('cuda')

        self._raise_error_if_features_are_multi_dimensional(query_features)
        # 欧式距离计算分数
        scores = self.l2_distance_to_prototypes(query_features)
        return self.softmax_if_specified(scores)

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):

        self.compute_prototypes_and_store_support_set(support_images, support_labels)

    @staticmethod
    def is_transductive() -> bool:
        return False

    def compute_features(self, images: Tensor) -> Tensor:
        original_features = self.backbone(images)
        centered_features = original_features - self.feature_centering
        if self.feature_normalization is not None:
            return nn.functional.normalize(
                centered_features, p=self.feature_normalization, dim=1
            )
        return centered_features

    def softmax_if_specified(self, output: Tensor, temperature: float = 1.0) -> Tensor:

        return (temperature * output).softmax(-1) if self.use_softmax else output


    # 欧几里得距离
    def l2_distance_to_prototypes(self, samples: Tensor) -> Tensor:

        return -torch.cdist(samples, self.prototypes)

    # 余弦距离
    def cosine_distance_to_prototypes(self, samples) -> Tensor:

        return (
            nn.functional.normalize(samples, dim=1)
            @ nn.functional.normalize(self.prototypes, dim=1).T
        )

    def compute_prototypes_and_store_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        self.support_labels = support_labels
        self.support_features = self.compute_features(support_images)
        self._raise_error_if_features_are_multi_dimensional(self.support_features)
        self.prototypes = compute_prototypes(self.support_features, support_labels)
    @staticmethod
    def _raise_error_if_features_are_multi_dimensional(features: Tensor):
        if len(features.shape) != 2:
            raise ValueError(
                "Illegal backbone or feature shape. "
                "Expected output for an image is a 1-dim tensor."
            )

