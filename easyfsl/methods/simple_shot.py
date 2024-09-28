from torch import Tensor

from .few_shot_classifier import FewShotClassifier


class SimpleShot(FewShotClassifier):
    """
    它使用余弦距离进行最近邻分类，而不是欧几里得距离。
    Almost exactly Prototypical Classification, but with cosine distance instead of euclidean distance.
    """

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        使用模型预测查询集图像的分类标签。
        参数:
            query_images: 查询集的图像，形状为(n_query, **image_shape)
        返回:
            查询图像的分类得分，形状为(n_query, n_classes)
        """
        # 计算查询集图像的特征
        query_features = self.compute_features(query_images)
        # 确保特征不是多维的，只支持一维特征
        self._raise_error_if_features_are_multi_dimensional(query_features)
        # 计算查询集图像特征与原型之间的余弦距离，得到分类得分
        scores = self.cosine_distance_to_prototypes(query_features)

        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False
