"""
See original implementation (quite far from this one)
at https://github.com/jakesnell/prototypical-networks
"""

from torch import Tensor

from .few_shot_classifier import FewShotClassifier


class PrototypicalNetworks(FewShotClassifier):
    """
    Prototypical networks 提取支持集和支持集图像的特征向量。然后它
    计算每个类别的支持特征的均值（称为原型），并基于查询图像与原型的欧几里得距离预测
    查询图像的分类分数。
    """
# 查询图像的分类分数
    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        根据查询图像与类原型在特征空间中的距离预测查询标签。
        """
        # 提取查询图像的特征
        query_features = self.compute_features(query_images)
        # 确保查询特征不是多维的，如果是，则抛出错误
        self._raise_error_if_features_are_multi_dimensional(query_features)

        # 计算查询到原型的欧几里得距离
        scores = self.l2_distance_to_prototypes(query_features)

        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False
