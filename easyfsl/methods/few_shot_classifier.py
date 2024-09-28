from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn

from easyfsl.methods.utils import compute_prototypes


class FewShotClassifier(nn.Module):
    """
    Abstract class providing methods usable by all few-shot classification algorithms
    一个 Few-Shot 分类器的抽象基类，定义了基本结构和方法。
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        use_softmax: bool = False,
        feature_centering: Optional[Tensor] = None,
        feature_normalization: Optional[float] = None,
    ):
        """
        Initialize the Few-Shot Classifier
        Args:
            backbone: 用于提取特征的模型。如果为 None，则初始化为 nn.Identity()。
            use_softmax: 是否返回软概率预测。
            feature_centering: 用于对所有计算得到的特征进行中心化的特征向量。
                如果为 None，则不进行中心化。
            feature_normalization: 用于对中心化后的所有计算得到的特征进行归一化的值。
                它被用作 torch.nn.functional.normalize() 中的 p 参数。
                如果为 None，则不进行归一化。
        """
        super().__init__()
        # 特征提取器，如果backbone为None，则使用恒等变换
        self.backbone = backbone if backbone is not None else nn.Identity()
        # 是否使用softmax进行概率预测
        self.use_softmax = use_softmax
        # 初始化原型、支持集特征和标签
        self.prototypes = torch.tensor(())
        self.support_features = torch.tensor(())
        self.support_labels = torch.tensor(())
        # 如果未提供特征中心化向量，则默认为0
        self.feature_centering = (
            feature_centering if feature_centering is not None else torch.tensor(0)
        )
        # 特征归一化参数
        self.feature_normalization = feature_normalization

    @abstractmethod
    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        预测查询集的分类标签。

        Args:
            query_images: 查询集图像，形状为 (n_query, **image_shape)

        Returns:
            查询图像的分类分数预测，形状为 (n_query, n_classes)
        """
        raise NotImplementedError(
            "All few-shot algorithms must implement a forward method."
        )

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        利用支持集信息，以便之后可以通过前向调用来预测查询集标签。
        默认行为是计算原型并存储支持集。

        Args:
            support_images: 支持集图像，形状为 (n_support, **image_shape)
            support_labels: 支持集图像标签，形状为 (n_support, )
        """
        self.compute_prototypes_and_store_support_set(support_images, support_labels)

    @staticmethod
    def is_transductive() -> bool:
        raise NotImplementedError(
            "All few-shot algorithms must implement a is_transductive method."
        )

    def compute_features(self, images: Tensor) -> Tensor:
        """
        从图像计算特征，并进行中心化和归一化。

        Args:
            images: 图像，形状为 (n_images, **image_shape)

        Returns:
            特征，形状为 (n_images, feature_dimension)
        """
        original_features = self.backbone(images)
        centered_features = original_features - self.feature_centering
        if self.feature_normalization is not None:
            return nn.functional.normalize(
                centered_features, p=self.feature_normalization, dim=1
            )
        return centered_features

    def softmax_if_specified(self, output: Tensor, temperature: float = 1.0) -> Tensor:
        """
        如果初始化分类器时选择了该选项，则对输出执行 softmax 以返回软概率。

        Args:
            output: forward 方法的输出，形状为 (n_query, n_classes)
            temperature: softmax 的温度

        Returns:
            原始输出或软概率输出，形状为 (n_query, n_classes)
        """
        return (temperature * output).softmax(-1) if self.use_softmax else output

    def l2_distance_to_prototypes(self, samples: Tensor) -> Tensor:
        """
        计算样本到支持集原型的欧几里得距离作为预测的对数几率。

        Args:
            samples: 待分类项的特征，形状为 (n_samples, feature_dimension)

        Returns:
            预测对数几率，形状为 (n_samples, n_classes)
        """
        return -torch.cdist(samples, self.prototypes)

    def cosine_distance_to_prototypes(self, samples) -> Tensor:
        """
        计算样本到支持集原型的余弦距离作为预测的对数几率。

        Args:
            samples: 待分类项的特征，形状为 (n_samples, feature_dimension)

        Returns:
            预测对数几率，形状为 (n_samples, n_classes)
        """
        return (
            nn.functional.normalize(samples, dim=1)
            @ nn.functional.normalize(self.prototypes, dim=1).T
        )

    def compute_prototypes_and_store_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        提取支持集特征，计算原型，并存储支持集标签、特征和原型。

        Args:
            support_images: 支持集图像，形状为 (n_support, **image_shape)
            support_labels: 支持集图像标签，形状为 (n_support, )
        """
        self.support_labels = support_labels
        self.support_features = self.compute_features(support_images)
        self._raise_error_if_features_are_multi_dimensional(self.support_features)
        self.prototypes = compute_prototypes(self.support_features, support_labels) # 保存在支持集原型表示变量里面



    @staticmethod   # 期望的图像输出是一个 1 维张量
    def _raise_error_if_features_are_multi_dimensional(features: Tensor):
        if len(features.shape) != 2:
            raise ValueError(
                "Illegal backbone or feature shape. "
                "Expected output for an image is a 1-dim tensor."
            )
