"""
See original implementation at
https://github.com/facebookresearch/low-shot-shrink-hallucinate
"""

from typing import Optional

import torch
from torch import Tensor, nn

from easyfsl.modules.predesigned_modules import (
    default_matching_networks_query_encoder,
    default_matching_networks_support_encoder,
)

from .few_shot_classifier import FewShotClassifier

class MatchingNetworks(FewShotClassifier):
    """
        Matching Networks 从支持集和支持图像中提取特征向量。
        然后利用整个支持集的上下文信息（通过 LSTM）细化这些特征。
        最后，根据查询图像与支持图像之间的余弦相似度计算查询标签。

        注意：虽然有些方法在情景训练中使用交叉熵损失，
        但 Matching Networks 输出的是对数概率，因此应该使用负对数似然损失。
    """
    def __init__(
        self,
        *args,
        feature_dimension: int,
        support_encoder: Optional[nn.Module] = None,
        query_encoder: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        Build Matching Networks by calling the constructor of FewShotClassifier.
        参数:
            feature_dimension: 由主干网络提取的特征向量的维度。
            support_encoder: 编码支持集特征的模块。如果没有指定，则使用原始论文中的默认编码器。
            query_encoder: 编码查询集特征的模块。如果没有指定，则使用原始论文中的默认编码器。
        """
        super().__init__(*args, **kwargs)

        self.feature_dimension = feature_dimension

        # 这些模块利用整个支持集的信息细化支持集和查询集的特征向量
        self.support_features_encoder = (
            support_encoder
            if support_encoder
            else default_matching_networks_support_encoder(self.feature_dimension)
        )
        self.query_features_encoding_cell = (
            query_encoder
            if query_encoder
            else default_matching_networks_query_encoder(self.feature_dimension)
        )

        self.softmax = nn.Softmax(dim=1)

        # 创建字段以便模型可以存储来自一个支持集的计算信息
        self.contextualized_support_features = torch.tensor(())
        self.one_hot_support_labels = torch.tensor(())

    '''
    支持集处理 process_support_set
    接收支持集图像 support_images 和标签 support_labels。
    使用 compute_features 方法提取特征。
    使用 encode_support_features 方法对支持集特征进行编码，使用双向LSTM来考虑整个支持集的上下文信息。
    将支持集标签转换为one-hot编码
    '''
    def process_support_set(
        self,
        support_images: Tensor,
        support_labels: Tensor,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        从支持集中提取带有完整上下文嵌入的特征。
        存储经过上下文编码的支持特征向量以及支持标签的一热编码形式。

        参数:
            support_images: 支持集的图像，形状为 (n_support, **image_shape)
            support_labels: 支持集图像的标签，形状为 (n_support, )
        """
        support_features = self.compute_features(support_images)
        self._validate_features_shape(support_features)
        self.contextualized_support_features = self.encode_support_features(
            support_features
        )

        self.one_hot_support_labels = (
            nn.functional.one_hot(  # pylint: disable=not-callable
                support_labels
            ).float()
        )
        '''
        前向传播 forward
        接收查询集图像 query_images。
        使用 compute_features 方法提取查询集特征。
        使用 encode_query_features 方法对查询集特征进行编码，使用注意力机制结合支持集的上下文信息。
        计算查询特征与支持集特征之间的余弦相似度。
        使用softmax函数获取分类概率。
        '''
    def forward(self, query_images: Tensor) -> Tensor:
        """
        Overrides method forward in FewShotClassifier.
        根据查询图像与支持集特征之间的余弦相似度预测查询标签。
        分类得分是对数概率。

        参数:
            query_images: 查询集的图像，形状为 (n_query, **image_shape)
        返回:
            对查询图像的分类得分预测，形状为 (n_query, n_classes)
        """

        # 利用整个支持集的上下文细化查询特征
        query_features = self.compute_features(query_images)
        self._validate_features_shape(query_features)
        contextualized_query_features = self.encode_query_features(query_features)

        # 计算所有查询图像与标准化支持图像之间的余弦相似度矩阵
        # 根据原始实现，我们不对查询特征进行标准化以保持softmax后的“锐利”向量（如果标准化，所有值趋于相同）
        similarity_matrix = self.softmax(
            contextualized_query_features.mm(
                nn.functional.normalize(self.contextualized_support_features).T
            )
        )

        # 根据查询实例与支持实例之间的余弦相似度及支持标签计算查询的对数概率
        log_probabilities = (
            similarity_matrix.mm(self.one_hot_support_labels) + 1e-6
        ).log()
        return self.softmax_if_specified(log_probabilities)

    '''
    支持集特征编码 encode_support_features
    使用双向LSTM对支持集特征进行编码，考虑整个支持集的上下文信息。 
    将原始特征与LSTM的隐藏状态合并，得到上下文化的支撑集特征。
    '''
    def encode_support_features(
        # 定义一个方法来对支持集特征进行编码，使用双向LSTM来捕获整个支持集的上下文信息
        self,
        support_features: Tensor,
    ) -> Tensor:
        """
        使用双向LSTM对支持集特征进行编码，考虑整个支持集的上下文信息。
        将原始特征与LSTM的隐藏状态合并，得到上下文化的支撑集特征。
        参数:
            support_features: 骨干网络的输出，形状为 (n_support, feature_dimension)
        返回:
            上下文化的支持集特征，与输入特征形状相同
    """
        # 隐藏状态形状为 [number_of_support_images, 2 * feature_dimension]，因为LSTM是双向的
        hidden_state = self.support_features_encoder(support_features.unsqueeze(0))[
            0
        ].squeeze(0)

        # 根据论文，上下文化的特征是通过将原始特征和双向LSTM的隐藏状态相加得到的
        contextualized_support_features = (
            support_features
            + hidden_state[:, : self.feature_dimension]
            + hidden_state[:, self.feature_dimension :]
        )

        return contextualized_support_features

    def encode_query_features(self, query_features: Tensor) -> Tensor:
        # 定义一个方法来对查询集特征进行编码，使用注意力机制和LSTM来考虑整个支持集的上下文信息
        """
        使用注意力机制和LSTM对查询集特征进行编码，考虑整个支持集的上下文信息。
        通过迭代，每个查询实例都通过LSTM更新其特征表示。
        参数:
            query_features: 骨干网络的输出，形状为 (n_query, feature_dimension)
        返回:
            上下文化的查询集特征，与输入特征形状相同
        """
        hidden_state = query_features
        cell_state = torch.zeros_like(query_features)

        # 遍历支持集特征，利用注意力机制更新查询集特征
        for _ in range(len(self.contextualized_support_features)):
            # 计算查询集特征与支持集特征之间的注意力权
            attention = self.softmax(
                hidden_state.mm(self.contextualized_support_features.T)
            )
            # 根据注意力权重读取支持集特征
            read_out = attention.mm(self.contextualized_support_features)
            # 将读取的特征与查询集特征拼接作为LSTM的输入
            lstm_input = torch.cat((query_features, read_out), 1)
            # 通过LSTM更新隐藏状态
            hidden_state, cell_state = self.query_features_encoding_cell(
                lstm_input, (hidden_state, cell_state)
            )
            hidden_state = hidden_state + query_features

        return hidden_state
    def _validate_features_shape(self, features: Tensor):
        # 定义一个方法来验证特征的形状是否正确
        # 确保特征不是多维的
        self._raise_error_if_features_are_multi_dimensional(features)
        # 确保特征维度与定义的feature_dimension匹配
        if features.shape[1] != self.feature_dimension:
            raise ValueError(
                f"Expected feature dimension is {self.feature_dimension}, but got {features.shape[1]}."
            )
    '''
    传递性 is_transductive
    
    返回 False 表示模型是非传递性的，即它不会在训练过程中改变其参数。
    '''
    @staticmethod
    def is_transductive() -> bool:
        return False
