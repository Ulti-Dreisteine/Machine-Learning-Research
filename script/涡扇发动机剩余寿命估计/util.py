import pandas as pd
import numpy as np


def load_batch_test_data(label: str = "1") -> pd.DataFrame:
    """
    载入批次试验数据
    
    Params:
    -------
    label: 批次标签
    """
    
    cols=["unit_nb", "time_cycle"] + ["set_1", "set_2", "set_3"] + [f"s_{i}" for i in range(1, 22)]
    data = pd.read_csv(
        f"data/train_FD00{label}.txt", sep="\\s+", names=cols, header=None, index_col=False)
    return data


def load_unit_test_data(data: pd.DataFrame, unit_nb: int, x_col: str) -> np.ndarray:
    """
    载入单台设备上的试验数据
    
    Params:
    -------
    data: 批次试验数据
    unit_nb: 设备编号
    x_col: 所采集的信号名
    """
    
    arr = data[data["unit_nb"] == unit_nb][x_col].values  # type: ignore
    return arr