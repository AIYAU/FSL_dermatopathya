from pathlib import Path
from typing import Union

import torch
from torch import Tensor, nn

from easyfsl.modules import MultiHeadAttention
from easyfsl.modules.feat_resnet12 import feat_resnet12

from .prototypical_networks import PrototypicalNetworks
from .utils import strip_prefix


class FEAT(PrototypicalNetworks):
    """
    本方法使用经过情景式训练的注意力模块来改进原型。
    查询基于它们与原型之间的欧几里得距离进行分类，
    类似于原型网络
    This in an inductive method.

    注意力模块必须遵循在 `FEAT.__init__` 方法文档中描述的具体约束条件。
    我们提供了一个默认的注意力模块，遵循原始实现中的一个。
    可以通过调用 `FEAT.from_resnet12_checkpoint` 来将 FEAT 初始化为作者提供的默认配置。
    """

    def __init__(self, *args, attention_module: nn.Module, **kwargs):
        """
        FEAT 需要一个额外的注意力模块。
        Args:
            *args:
            attention_module: the forward method must accept 3 Tensor arguments of shape
                (1, num_classes, feature_dimension) and return a pair of Tensor, with the first
                one of shape (1, num_classes, feature_dimension).
                This follows the original implementation of https://github.com/Sha-Lab/FEAT
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.attention_module = attention_module

    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        从支持集提取原型，并使用注意力模块对其进行修正。
        Args:
            support_images: 支持图像，形状为 (n_support, **image_shape)
            support_labels: 支持标签，形状为 (n_support,)
        """
        super().process_support_set(support_images, support_labels)
        # 使用注意力模块修正原型
        self.prototypes = self.attention_module(
            self.prototypes.unsqueeze(0),
            self.prototypes.unsqueeze(0),
            self.prototypes.unsqueeze(0),
        )[0][0]

    @classmethod
    def from_resnet12_checkpoint(
        cls,
        checkpoint_path: Union[Path, str],
        device: str = "cpu",
        feature_dimension: int = 640,
        use_backbone: bool = True,
        **kwargs,
    ):
        """
        从 resnet12 的检查点加载 FEAT 模型。
        Load a FEAT model from a checkpoint of a resnet12 model as provided by the authors.
        We initialize the default ResNet12 backbone and attention module and load the weights.
        We solve some compatibility issues in the names of the parameters and ensure there
        missing keys.

        Compatible weights can be found here (availability verified 30/05/2023):
            - miniImageNet: https://drive.google.com/file/d/1ixqw1l9XVxl3lh1m5VXkctw6JssahGbQ/view
            - tieredImageNet: https://drive.google.com/file/d/1M93jdOjAn8IihICPKJg8Mb4B-eYDSZfE/view
        Args:
            checkpoint_path: path to the checkpoint
            device: device to load the model on
            feature_dimension: dimension of the features extracted by the backbone.
                Should be 640 with the default Resnet12 backbone.
            use_backbone: if False, we initialize the backbone to nn.Identity() (useful for
                working on pre-extracted features)
        Returns:
            a FEAT model with weights loaded from the checkpoint
        Raises:
            ValueError: if the checkpoint does not contain all the expected keys
                of the backbone or the attention module
        """
        state_dict = torch.load(str(checkpoint_path), map_location=device)["params"]

        if use_backbone:
            backbone = feat_resnet12().to(device)
            backbone_missing_keys, _ = backbone.load_state_dict(
                strip_prefix(state_dict, "encoder."), strict=False
            )
            if len(backbone_missing_keys) > 0:
                raise ValueError(f"Missing keys for backbone: {backbone_missing_keys}")
        else:
            backbone = nn.Identity()

        attention_module = MultiHeadAttention(
            1,
            feature_dimension,
            feature_dimension,
            feature_dimension,
        ).to(device)
        attention_missing_keys, _ = attention_module.load_state_dict(
            strip_prefix(state_dict, "slf_attn."), strict=False
        )
        if len(attention_missing_keys) > 0:
            raise ValueError(
                f"Missing keys for attention module: {attention_missing_keys}"
            )

        return cls(backbone, attention_module=attention_module, **kwargs).to(device)
