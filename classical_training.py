# 导包
import copy
from pathlib import Path
import random
from statistics import mean
import numpy as np
import torch
import torchvision
from torch import nn
from tqdm import tqdm

# 设置随机种子
random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from easyfsl.datasets import CUB
from torch.utils.data import DataLoader

batch_size = 128
n_workers = 0

train_set = CUB(split="train", training=True,image_size=128)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=n_workers,
    pin_memory=True,
    shuffle=True,
)

from easyfsl.modules import resnet12

DEVICE = "cuda"

model = resnet12( # 定义主干网络
    use_fc=True,
    num_classes=len(set(train_set.get_labels())),
).to(DEVICE)

from easyfsl.methods import PrototypicalNetworks, FewShotClassifier, PTMAP
from easyfsl.samplers import TaskSampler
from easyfsl.utils import evaluate

n_way = 3
n_shot = 5
n_query = 5
n_validation_tasks = 100

val_set = CUB(split="val", training=False, image_size=224)
val_sampler = TaskSampler(
    val_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_validation_tasks
)

val_loader = DataLoader(
    val_set,
    batch_sampler=val_sampler,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=val_sampler.episodic_collate_fn,
)
convolutional_network = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
# convolutional_network.fc = nn.Linear(in_features=512, out_features=768, bias=True)
# print(convolutional_network)
# 定义 Few-shot 分类器
few_shot_classifier = PTMAP(backbone=convolutional_network).to(DEVICE)


from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter


LOSS_FUNCTION = nn.CrossEntropyLoss()
# 定义训练参数
n_epochs = 50
scheduler_milestones = [1, 5, 10]
scheduler_gamma = 0.1
learning_rate = 1e-5
tb_logs_dir = Path("./logs")

# train_optimizer = SGD(
#     few_shot_classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
# )
train_optimizer = Adam(
    few_shot_classifier.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4
)
train_scheduler = MultiStepLR(
    train_optimizer,
    milestones=scheduler_milestones,
    gamma=scheduler_gamma,
)

tb_writer = SummaryWriter(log_dir=str(tb_logs_dir))

# 训练一个epoch，这里是训练主干网络及特征提取器
def training_epoch(model_: nn.Module, data_loader: DataLoader, optimizer: Optimizer):
    all_loss = []
    model_.train()
    with tqdm(data_loader, total=len(data_loader), desc="Training") as tqdm_train: # 遍历数据集
        for images, labels in tqdm_train:
            optimizer.zero_grad()

            loss = LOSS_FUNCTION(model_(images.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()

            all_loss.append(loss.item())

            tqdm_train.set_postfix(loss=mean(all_loss))

    return mean(all_loss)


from easyfsl.utils import evaluate

# 初始化最佳模型状态和最佳验证准确率
best_state = model.state_dict()
best_validation_accuracy = 0.0
validation_frequency = 5
# 开始训练
for epoch in range(n_epochs):
    print(f"Epoch {epoch}")
    average_loss = training_epoch(model, train_loader, train_optimizer) # 传入定义的模型和数据集优化器配置等

    if epoch % validation_frequency == validation_frequency - 1: # 如果达到预设的验证要求就验证模型

        # We use this very convenient method from EasyFSL's ResNet to specify
        # that the model shouldn't use its last fully connected layer during validation.
        model.set_use_fc(False)
        validation_accuracy = evaluate(
            few_shot_classifier, val_loader, device=DEVICE, tqdm_prefix="Validation"
        )
        model.set_use_fc(True)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = copy.deepcopy(few_shot_classifier.state_dict())
            # state_dict() returns a reference to the still evolving model's state so we deepcopy
            # https://pytorch.org/tutorials/beginner/saving_loading_models
            print("Ding ding ding! We found a new best model!")

        tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)

    tb_writer.add_scalar("Train/loss", average_loss, epoch)

    # Warn the scheduler that we did an epoch
    # so it knows when to decrease the learning rate
    train_scheduler.step()

# 加载最优权重
model.load_state_dict(best_state, strict=False)
# 定义测试集数据加载器
n_test_tasks = 1000

test_set = CUB(split="test", training=False, image_size=128)

test_sampler = TaskSampler(
    test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
)
test_loader = DataLoader(
    test_set,
    batch_sampler=test_sampler,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

model.set_use_fc(False)
# 在测试集上评估模型
accuracy = evaluate(few_shot_classifier, test_loader, device=DEVICE)
print(f"Average accuracy : {(100 * accuracy):.2f} %")