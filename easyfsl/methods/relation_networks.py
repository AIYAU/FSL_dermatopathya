"""
See original implementation at
https://github.com/floodsung/LearningToCompare_FSL
"""

from typing import Optional
import torch
from torch import Tensor, nn
from easyfsl.modules.predesigned_modules import default_relation_module
from .few_shot_classifier import FewShotClassifier
from .utils import compute_prototypes


class RelationNetworks(FewShotClassifier):
    """
    通过计算查询样本与支持集原型之间的关系得分来进行分类。
    """

    def __init__(
        self,
        *args,
        feature_dimension: int,
        relation_module: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        构建Relation Networks模型。

        Args:
            feature_dimension: 从支持集中提取的特征图的第一维度大小。
            relation_module: 用于计算关系得分的模块，它接收查询特征向量和原型的拼接作为输入。
                             如果没有指定，默认使用原始论文中的默认关系模块。
        """
        super().__init__(*args, **kwargs)

        self.feature_dimension = feature_dimension

        # 构建关系模块，用于输出每对查询和原型之间的关系得分。
        self.relation_module = (
            relation_module
            if relation_module
            else default_relation_module(self.feature_dimension)
        )

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        从支持集中提取特征图并存储类原型。
        Args:
            support_images: 支持集图像。
            support_labels: 支持集标签。
        """

        support_features = self.compute_features(support_images)
        # 验证特征图的形状是否符合预期
        self._validate_features_shape(support_features)
        # 计算并存储类原型
        self.prototypes = compute_prototypes(support_features, support_labels)

    def forward(self, query_images: Tensor) -> Tensor:
        """
        Overrides method forward in FewShotClassifier.
        通过将查询图像的特征图与每个类原型拼接，并将结果送入关系模块（一个输出关系得分的CNN），
        来预测查询图像的标签。最终，查询的分类向量是它与每个类原型的关系得分。
        Args:
            query_images: 查询图像。
        Returns:
            查询图像与每个类原型的关系得分。
        """
        # 提取查询图像的特征
        print(f"query_images.shape={query_images.shape}")
        query_features = self.compute_features(query_images)
        # 验证查询特征图的形状
        # print(f"这里是query_features.shape, len(query_features)",query_features.shape, len(query_features))
        self._validate_features_shape(query_features)


        # 拼接查询特征和原型特征
        query_prototype_feature_pairs = torch.cat(
            (
                # 扩展原型特征以匹配查询特征的数量
                self.prototypes.unsqueeze(dim=0).expand(
                    query_features.shape[0], -1, -1, -1, -1
                ),
                # 扩展查询特征以匹配原型的数量
                query_features.unsqueeze(dim=1).expand(
                    -1, self.prototypes.shape[0], -1, -1, -1
                ),
            ),
            dim=2,   # 沿着通道维度拼接
        ).view(-1, 2 * self.feature_dimension, *query_features.shape[2:])

        # 计算关系得分
        relation_scores = self.relation_module(query_prototype_feature_pairs).view(
            -1, self.prototypes.shape[0]
        )
        # 返回关系得分
        return self.softmax_if_specified(relation_scores)

    def _validate_features_shape(self, features):
        # 验证特征图的形状是否合法。
        if len(features.shape) != 4:
            raise ValueError(
                f"features shape={len(features.shape)}"
                "Illegal backbone for Relation Networks. "
                "Expected output for an image is a 3-dim  tensor of shape (n_channels, width, height)."
            )
        if features.shape[1] != self.feature_dimension:
            raise ValueError(
                f"Expected feature dimension is {self.feature_dimension}, but got {features.shape[1]}."
            )

    @staticmethod
    def is_transductive() -> bool:
        return False
