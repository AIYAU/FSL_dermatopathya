import torch
from torch import Tensor

from .bd_cspn import BDCSPN
from .utils import k_nearest_neighbours


class LaplacianShot(BDCSPN):
    """
    LaplacianShot 使用拉普拉斯正则化更新软分配，以提高相邻查询点之间分配的一致性。
    默认超参数针对 miniImageNet 上的 5-way 5-shot 分类进行了优化

    LaplianShot is a transductive method.
    """

    def __init__(
        self,
        *args,
        inference_steps: int = 20,
        knn: int = 3,
        lambda_regularization: float = 0.7,
        **kwargs,
    ):
        """
        初始化 LaplacianShot 对象，并设置特定的超参数。
        参数:
            inference_steps (int): 推理更新步骤数。
            knn (int): 计算邻近矩阵时考虑的最近邻居数量。
            lambda_regularization (float): 拉普拉斯项的正则化参数。
        """
        super().__init__(*args, **kwargs)
        self.knn = knn
        self.inference_steps = inference_steps
        self.lambda_regularization = lambda_regularization

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        对查询图像执行前向传播。
        参数:
            query_images (Tensor): 查询图像张量。
        返回:
            Tensor: 每个查询图像的预测软分配
        """
        query_features = self.compute_features(query_images)
        self.rectify_prototypes(query_features=query_features)
        # 计算查询与原型的欧几里得距离
        features_to_prototypes_distances = (
            torch.cdist(query_features, self.prototypes) ** 2
        )
        # 据特征计算邻近矩阵。（用knn）
        pairwise_affinities = self.compute_pairwise_affinities(query_features)
        # 使用边界更新算法计算软分配。
        predictions = self.bound_updates(
            initial_scores=features_to_prototypes_distances, kernel=pairwise_affinities
        )

        return predictions

    def compute_pairwise_affinities(self, features: Tensor) -> Tensor:
        """
        根据特征计算邻近矩阵。
        参数:
            features (Tensor): 形状为 (n_features, feature_dimension) 的特征张量。
        返回:
            Tensor: 形状为 (n_features, n_features) 的邻近矩阵。
        """
        # Compute the k-nearest neighbours of each feature vector.
        # Each row is the indices of the k nearest neighbours of the corresponding feature, not including itself
        nearest_neighbours = k_nearest_neighbours(features, self.knn)
        affinity_matrix = torch.zeros((len(features), len(features))).to(
            nearest_neighbours.device
        )
        for vector_index, vector_nearest_neighbours in enumerate(nearest_neighbours):
            affinity_matrix[vector_index].index_fill_(0, vector_nearest_neighbours, 1)

        return affinity_matrix

    def compute_upper_bound(
        self, soft_assignments: Tensor, initial_scores: Tensor, kernel: Tensor
    ) -> float:
        """
        计算软分配的上界目标值。
        参数:
            soft_assignments (Tensor): 形状为 (n_query, n_classes) 的软分配。
            initial_scores (Tensor): 初始得分（到原型的距离）。
            kernel (Tensor): 邻近矩阵。
        返回:
            float: 上界目标值。
        """
        pairwise = kernel.matmul(soft_assignments)
        temp = (initial_scores * soft_assignments) + (
            -self.lambda_regularization * pairwise * soft_assignments
        )
        upper_bound = (soft_assignments * (soft_assignments + 1e-12).log() + temp).sum()

        return upper_bound.item()

    def bound_updates(self, initial_scores: Tensor, kernel: Tensor) -> Tensor:
        """
        使用边界更新算法计算软分配。
        参数:
            initial_scores (Tensor): 初始得分（到原型的距离）。
            kernel (Tensor): 邻近矩阵。
        返回:
            Tensor: 最终软分配。
        """
        old_upper_bound = float("inf")
        soft_assignments = (-initial_scores).softmax(dim=1)
        for i in range(self.inference_steps):
            additive = -initial_scores
            mul_kernel = kernel.matmul(soft_assignments)
            soft_assignments = -self.lambda_regularization * mul_kernel
            additive = additive - soft_assignments
            soft_assignments = additive.softmax(dim=1)
            upper_bound = self.compute_upper_bound(
                soft_assignments, initial_scores, kernel
            )

            if i > 1 and (
                abs(upper_bound - old_upper_bound) <= 1e-6 * abs(old_upper_bound)
            ):
                break

            old_upper_bound = upper_bound

        return soft_assignments

    @staticmethod
    def is_transductive() -> bool:
        return True
