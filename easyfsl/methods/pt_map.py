import torch
from torch import Tensor, nn

from easyfsl.methods.utils import power_transform

from .few_shot_classifier import FewShotClassifier
# 最大Sinkhorn迭代次数，用于优化传输计划
MAXIMUM_SINKHORN_ITERATIONS = 1000

class PTMAP(FewShotClassifier):
    """
    该方法计算查询到支持类别原型的最优传输计划作为软分配。
    在每次迭代中，根据软分配对原型进行微调。
    This is a transductive method.
    """

    def __init__(
        self,
        *args,
        fine_tuning_steps: int = 10,
        fine_tuning_lr: float = 0.2,
        lambda_regularization: float = 10.0,
        power_factor: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.lambda_regularization = lambda_regularization
        self.power_factor = power_factor

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        预测查询软分配。
        """
        query_features = self.compute_features(query_images)

        support_assignments = nn.functional.one_hot(  # pylint: disable=not-callable
            self.support_labels, len(self.prototypes)
        )
        for _ in range(self.fine_tuning_steps):
            # 计算查询到原型的软分配。（通过欧几里得和sinkhorn算法迭代）
            query_soft_assignments = self.compute_soft_assignments(query_features)
            all_features = torch.cat([self.support_features, query_features], 0)
            all_assignments = torch.cat(
                [support_assignments, query_soft_assignments], dim=0
            )
            # 更新原型
            self.update_prototypes(all_features, all_assignments)
        # 计算查询到原型的软分配。
        return self.compute_soft_assignments(query_features)

    def compute_features(self, images: Tensor) -> Tensor:
        """
        应用幂变换到特征。
        Apply power transform on features following Equation (1) in the paper.
        Args:
            images: images of shape (n_images, **image_shape)
        Returns:
            features of shape (n_images, feature_dimension) with power-transform.
        """
        features = super().compute_features(images)
        return power_transform(features, self.power_factor)

    def compute_soft_assignments(self, query_features: Tensor) -> Tensor:
        """
        计算查询到原型的软分配。

        Compute soft assignments from queries to prototypes, following Equation (3) of the paper.
        Args:
            query_features: query features, of shape (n_queries, feature_dim)

        Returns:
            soft assignments from queries to prototypes, of shape (n_queries, n_classes)
        """

        distances_to_prototypes = (
            torch.cdist(query_features, self.prototypes) ** 2
        )  # [Nq, K]
        # 使用Sinkhorn - Knopp算法计算查询到原型的最优传输计划。
        soft_assignments = self.compute_optimal_transport(
            distances_to_prototypes, epsilon=1e-6
        )

        return soft_assignments

    def compute_optimal_transport(
        self, cost_matrix: Tensor, epsilon: float = 1e-6
    ) -> Tensor:
        """
        使用Sinkhorn-Knopp算法计算查询到原型的最优传输计划。
        Args:
            cost_matrix: 形状为(n_queries, n_classes)的查询到原型的欧几里得距离
            epsilon: 收敛参数。当更新值小于epsilon时停止。
        Returns:
            形状为(n_queries, n_classes)的查询到原型的传输计划
        """
        # 根据成本矩阵的行和列数计算实例乘数因子
        instance_multiplication_factor = cost_matrix.shape[0] // cost_matrix.shape[1]

        # 初始化运输计划，使用负的正则化成本矩阵进行指数运算
        transport_plan = torch.exp(-self.lambda_regularization * cost_matrix)
        # 对运输计划进行归一化，使其在所有维度上的和为1
        transport_plan /= transport_plan.sum(dim=(0, 1), keepdim=True)
        # 进行Sinkhorn迭代，直到满足收敛条件或达到最大迭代次数
        for _ in range(MAXIMUM_SINKHORN_ITERATIONS):
            # 计算每行的和，用于后续的归一化
            per_class_sums = transport_plan.sum(1)
            # 对运输计划进行行归一化
            transport_plan *= (1 / (per_class_sums + 1e-10)).unsqueeze(1)
            # 对运输计划进行列归一化，考虑实例乘数因子
            transport_plan *= (
                instance_multiplication_factor / (transport_plan.sum(0) + 1e-10)
            ).unsqueeze(0)
            # 检查收敛性，如果每行和的变化小于给定的epsilon，则停止迭代
            if torch.max(torch.abs(per_class_sums - transport_plan.sum(1))) < epsilon:
                break

        return transport_plan

    def update_prototypes(self, all_features, all_assignments) -> None:
        """
        Update prototypes by weigh-averaging the features with their soft assignments,
            following Equation (6) of the paper.
        Args:
            all_features: concatenation of support and query features,
                of shape (n_support + n_query, feature_dim)
            all_assignments: concatenation of support and query soft assignments,
                of shape (n_support + n_query, n_classes)-
        """
        new_prototypes = (all_assignments.T @ all_features) / all_assignments.sum(
            0
        ).unsqueeze(1)
        delta = new_prototypes - self.prototypes
        self.prototypes += self.fine_tuning_lr * delta

    @staticmethod
    def is_transductive() -> bool:
        return True
