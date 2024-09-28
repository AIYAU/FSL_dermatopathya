from torch import Tensor, nn

from .few_shot_classifier import FewShotClassifier


class BDCSPN(FewShotClassifier):
    """
    通过标签传播和特征偏移修正原型。
    基于查询与原型的余弦距离进行分类。

    This is a transductive method.
    """

    def rectify_prototypes(
        self, query_features: Tensor
    ):  # pylint: disable=not-callable
        """
        Updates prototypes with label propagation and feature shifting.
        Args:
            query_features: query features of shape (n_query, feature_dimension)
        """
        n_classes = self.support_labels.unique().size(0)    # 支持集类别数量
        one_hot_support_labels = nn.functional.one_hot(self.support_labels, n_classes)  # 标签转换为one hot
        # 计算支持集和查询集特征的平均偏移量
        average_support_query_shift = self.support_features.mean(
            0, keepdim=True
        ) - query_features.mean(0, keepdim=True)
        # 应用到查询特征上
        query_features = query_features + average_support_query_shift
        # 计算支持集和查询集到当前原型的余弦距离，并转换为概率分布
        support_logits = self.cosine_distance_to_prototypes(self.support_features).exp()
        query_logits = self.cosine_distance_to_prototypes(query_features).exp()

        # 对查询集特征进行预测，并进行独热编码
        one_hot_query_prediction = nn.functional.one_hot(
            query_logits.argmax(-1), n_classes
        )
        # 计算归一化向量
        normalization_vector = (
            (one_hot_support_labels * support_logits).sum(0)
            + (one_hot_query_prediction * query_logits).sum(0)
        ).unsqueeze(
            0
        )  # [1, n_classes]

        # 计算支持集和查询集的重加权矩阵
        support_reweighting = (
            one_hot_support_labels * support_logits
        ) / normalization_vector  # [n_support, n_classes]
        query_reweighting = (
            one_hot_query_prediction * query_logits
        ) / normalization_vector  # [n_query, n_classes]

        # 更新原型
        self.prototypes = (support_reweighting * one_hot_support_labels).t().matmul(
            self.support_features
        ) + (query_reweighting * one_hot_query_prediction).t().matmul(query_features)

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        使用查询图像更新原型，
        然后基于查询图像与更新后原型的余弦距离对查询图像进行分类。
        步骤:
        1. 计算查询图像的特征。
        2. 更新原型，使用查询图像特征。
        3. 计算查询图像特征与更新后的原型之间的余弦距离。
        4. 应用softmax函数（如果指定）并返回结果。
        """
        query_features = self.compute_features(query_images)

        self.rectify_prototypes(
            query_features=query_features,
        )
        return self.softmax_if_specified(
            self.cosine_distance_to_prototypes(query_features)
        )

    @staticmethod
    def is_transductive() -> bool:
        return True
