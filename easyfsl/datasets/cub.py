from pathlib import Path

from .easy_set import EasySet

CUB_SPECS_DIR = Path("data/CUB")


class CUB(EasySet):
    def __init__(self, split: str, **kwargs):
        """
        CUB类用于构建特定分割的CUB数据集。

        继承自EasySet类，初始化时根据提供的分割类型（如训练、验证或测试）加载相应的数据集规范。
        """
        specs_file = CUB_SPECS_DIR / f"{split}.json"
        if not specs_file.is_file():
            raise ValueError(
                f"Could not find specs file {specs_file.name} in {CUB_SPECS_DIR}"
            )
        super().__init__(specs_file=specs_file, **kwargs)
