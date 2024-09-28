import json
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple, Union

from PIL import Image

from .default_configs import DEFAULT_IMAGE_FORMATS, default_transform
from .few_shot_dataset import FewShotDataset


class EasySet(FewShotDataset):
    """
    一个即用型数据集。适用于图像按类别分组在目录中的任何数据集。它期望一个JSON文件来定义
    类别及其位置。JSON文件必须具有以下结构:
        {
            "class_names": [
                "class_1",
                "class_2"
            ],
            "class_roots": [
                "path/to/class_1_folder",
                "path/to/class_2_folder"
            ]
        }
    """

    def __init__(
        self,
        specs_file: Union[Path, str],
        image_size: int = 84,
        transform: Optional[Callable] = None,
        training: bool = False,
        supported_formats: Optional[Set[str]] = None,
    ):
        """
        初始化EasySet数据集。

        参数:
            specs_file: JSON文件的路径。
            image_size: 数据集返回的图像将是给定大小的正方形图像。
            transform: 应用于图像的torchvision变换。如果未提供，
                我们将使用一些标准变换，包括ImageNet标准化。
                这些默认变换取决于"training"参数。
            training: 预处理稍微不同，添加一个随机裁剪和一个随机水平翻转。
                只在transforms = None时使用。
            supported_formats: 允许的文件格式集合。当列出数据实例时，EasySet
                只考虑这些文件。如果未提供，我们将使用默认的图像格式集合。
        """
        specs = self.load_specs(Path(specs_file))

        self.images, self.labels = self.list_data_instances(
            specs["class_roots"], supported_formats=supported_formats
        )

        self.class_names = specs["class_names"]

        self.transform = (
            transform if transform else default_transform(image_size, training)
        )

    @staticmethod
    def load_specs(specs_file: Path) -> dict:
        """
        从JSON文件加载规格。

        参数:
            specs_file: JSON文件的路径。

        返回:
            JSON文件中包含的字典。

        异常:
            ValueError: 如果specs_file不是JSON文件，或者如果是JSON文件且内容不符合预期结构。
        """

        if specs_file.suffix != ".json":
            raise ValueError("EasySet requires specs in a JSON file.")

        with open(specs_file, "r", encoding="utf-8") as file:
            specs = json.load(file)

        if "class_names" not in specs.keys() or "class_roots" not in specs.keys():
            raise ValueError(
                "EasySet requires specs in a JSON file with the keys class_names and class_roots."
            )

        if len(specs["class_names"]) != len(specs["class_roots"]):
            raise ValueError(
                "Number of class names does not match the number of class root directories."
            )

        return specs

    @staticmethod
    def list_data_instances(
        class_roots: List[str], supported_formats: Optional[Set[str]] = None
    ) -> Tuple[List[str], List[int]]:
        """
        探索class_roots中指定的目录，以找到所有的数据实例。

        参数:
            class_roots: 每个元素是包含一个类的元素的目录的路径
            supported_formats: 允许的文件格式集合。当列出数据实例时，EasySet
                只考虑这些文件。如果未提供，我们将使用默认的图像格式集合。

        返回:
            图像路径的列表和每个图像的整数标签列表
        """
        if supported_formats is None:
            supported_formats = DEFAULT_IMAGE_FORMATS

        images = []
        labels = []
        for class_id, class_root in enumerate(class_roots):
            class_images = [
                str(image_path)
                for image_path in sorted(Path(class_root).glob("*"))
                if image_path.is_file()
                & (image_path.suffix.lower() in supported_formats)
            ]

            images += class_images
            labels += len(class_images) * [class_id]

        if len(images) == 0:
            warnings.warn(
                UserWarning(
                    "No images found in the specified directories. The dataset will be empty."
                )
            )

        return images, labels

    def __getitem__(self, item: int):
        """"
        根据其整数id获取一个数据样本。

        参数:
            item: 样本的整数id

        返回:
            数据样本，以元组形式 (image, label)，其中label是整数。
            图像对象的类型取决于self.transform的输出类型。默认情况下
            它是torch.Tensor，然而你可以自由定义任何函数作为self.transform，
            因此图像的类型可以是任意的。例如，如果self.transform = lambda x: x，
            那么输出图像将是PIL.Image.Image类型。
        """
        # 一些ILSVRC2015的图像是灰度图像，所以我们转换所有图像以保持一致性。
        # 如果你想处理灰度图像，可以在你的变换管道中使用torch.transforms.Grayscale
        # transformation pipeline.
        img = self.transform(Image.open(self.images[item]).convert("RGB"))
        label = self.labels[item]

        return img, label

    def __len__(self) -> int:
        # 返回数据集的样本数量
        return len(self.labels)

    def get_labels(self) -> List[int]:
        # 返回数据集中所有样本的标签列表
        return self.labels

    def number_of_classes(self):
        # 返回数据集中类的数量
        return len(self.class_names)
