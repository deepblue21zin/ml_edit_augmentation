import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


CATEGORY_LIST = ["circle", "diagonal_left", "diagonal_right", "horizontal", "vertical"]


def load_data(data_folder: str) -> pd.DataFrame:
    """
    input:
        - data_folder: ex) data1, data2 ...

    output:
        - df: dataframe
    """
    all_datas = []
    for category in CATEGORY_LIST:  # 카테고리별로 데이터 수집
        folder_path = os.path.join(data_folder, category)

        # 포즈 하나의 데이터 추출 (ex. circle>1.txt, circle>2.txt)
        for data_id in os.listdir(folder_path):
            data_file_path = os.path.join(folder_path, data_id)
            pose_data = _get_pose_data(data_file_path)  # 동작 하나의 데이터
            data_id = data_id.split(".")[0]
            pose_data = [
                [category, int(data_id), time_step, line[0], line[1], line[2]]
                for time_step, line in enumerate(pose_data)
            ]
            all_datas.extend(pose_data)
    # print(f"{len(all_datas)} x {len(all_datas[0])}") 

    return pd.DataFrame(
        all_datas, columns=["category", "data_id", "time_step", "x", "y", "z"]
    )


def _get_pose_data(file_path: str) -> list[tuple[int]]:
    """
    하나의 포즈 데이터를 시각화하는 함수 (ex. 1.txt, 2.txt, 3.txt ...)
    input:
        - file_path: ex) data1/circle/1.txt

    output:
        - data_array
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data_list = f.readlines()

    data_array = []  # 하나의 시계열 데이터
    for line in data_list:
        row = line.split(",")
        # print(f"하나의 행을 인코딩합니다. {row}")
        if len(row) < 7:
            continue
        x, y, z = map(int, row[6].split("/"))
        data_array.append((x, y, z))

    return data_array


def write_data(df, category: str, data_id: int) -> plt.figure:
    """
    input:
        - df: Data Frame
        - category: ["circle", "diagonal_left", "diagonal_right", "horizontal", "vertical"]
        - data_id: 1 ~
    """
    series = df[(df["category"] == category) & (df["data_id"] == data_id)]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(series["x"], series["y"], series["z"])
    return fig
