# -*- coding: utf-8 -*-
import shutil
import os
import tyro
import numpy as np
from tqdm import tqdm # 用于显示进度条

# 导入必要的 LeRobot 组件
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
# 不再需要导入 lerobot_as_path

# ----- 用户配置 -----
# !! 请根据你的数据进行修改 !!

# 你想要读取的 *现有* LeRobot 数据集的名称/ID
# 这应该与 LEROBOT_HOME 内的目录名或你的 HF 仓库 ID 匹配
# 例如: "IPEC-COMMUNITY/libero_goal_no_noops_lerobot"
# 或本地: "libero_goal_no_noops_lerobot"
INPUT_REPO_ID = "IPEC-COMMUNITY/libero_goal_no_noops_lerobot" # <-- 修改这里！指定你的输入数据集 ID

# 你想要创建的 *新* LeRobot 数据集的名称/ID
# 一个以此命名的、新的目录将在 LEROBOT_HOME 中被创建
OUTPUT_REPO_ID = "your_hf_username/libero_goal_frame_by_frame_zh_v2" # <-- 修改这里！指定你想要的输出名称

# 为 *新* 数据集定义特征 (Features)
# 我们将把输入数据集的特征数据映射到这些新特征上
# 这里的键 (key) 必须与下面 `frame_data` 字典中使用的键匹配
OUTPUT_DATASET_FEATURES = {
    # 主相机图像
    "image": {
        "dtype": "image",
        # !! 请根据你的输入数据集 info.json 中的实际形状或期望的输出形状调整 !!
        "shape": (128, 128, 3),
        "names": ["height", "width", "channel"],
    },
    # 手腕相机图像
    "wrist_image": {
        "dtype": "image",
        # !! 请根据你的输入数据集 info.json 中的实际形状或期望的输出形状调整 !!
        "shape": (84, 84, 3),
        "names": ["height", "width", "channel"],
    },
    # 状态信息
    "state": {
        "dtype": "float32",
        # !! 请根据你的输入数据集 info.json 中的实际形状调整 !!
        "shape": (8,),
        "names": ["state"],
    },
    # 动作信息
    "actions": {
        "dtype": "float32",
         # !! 请根据你的输入数据集 info.json 中的实际形状调整 !!
        "shape": (7,),
        "names": ["actions"],
    },
    # 如果你还想复制其他特征，请在此添加
}

# 从 *输入* 数据集特征名到 *输出* 数据集特征名的映射
# 键 (Keys): 输入数据集中 info.json / 数据键中出现的特征名
# 值 (Values): 在 OUTPUT_DATASET_FEATURES 中定义的输出特征名
FEATURE_MAP = {
    "observation.images.image": "image",
    "observation.images.wrist_image": "wrist_image",
    "observation.state": "state",
    "action": "actions", # 注意输入是 'action'，输出我们习惯用 'actions'
    # 如果需要，添加其他映射
}

# 为 *新* 数据集定义元数据 (可以从输入数据集复制)
# !! 请根据你的输入数据集 info.json 中的实际值调整 !!
ROBOT_TYPE = "panda"
FPS = 10

# ----- 用户配置结束 -----


def main(input_repo_id: str = INPUT_REPO_ID,
         output_repo_id: str = OUTPUT_REPO_ID,
         *,
         push_to_hub: bool = False):
    """
    将一个现有的 LeRobot 数据集（包含视频块）转换为一个新的 LeRobot 数据集，
    新数据集使用 add_frame 进行逐帧图像存储。
    (此版本不使用 lerobot_as_path 函数)

    Args:
        input_repo_id: 要读取的 LeRobot 数据集的仓库 ID (本地路径或 HF Hub ID)。
                       LeRobotDataset 类将自行处理查找。
        output_repo_id: 要创建的新数据集的仓库 ID (本地路径)。
        push_to_hub: 是否将新创建的数据集推送到 Hub。
    """

    # --- 1. 设置路径并加载输入数据集 ---
    # 输出路径总是先在本地创建
    output_path = LEROBOT_HOME / output_repo_id

    # 直接打印用户提供的输入 ID 和计算出的输出路径
    print(f"输入数据集 ID: {input_repo_id}")
    print(f"输出数据集路径: {output_path}")

    # 检查输出路径是否存在，如果存在则删除
    if output_path.exists():
        print(f"警告: 输出目录 {output_path} 已存在。将删除它。")
        shutil.rmtree(output_path)

    print(f"尝试加载输入数据集: {input_repo_id}")
    try:
        # 加载现有数据集。 LeRobotDataset 类会处理查找逻辑，
        # 包括检查本地缓存 ($LEROBOT_HOME, ~/.cache/huggingface/lerobot)
        # 以及在需要时从 Hugging Face Hub 下载。
        input_ds = LeRobotDataset(input_repo_id)
        print(f"输入数据集加载成功。包含 {len(input_ds)} 个 episodes。")
    except Exception as e:
        # 捕获加载时可能发生的任何错误（例如，找不到数据集）
        print(f"\n错误: 加载输入数据集 '{input_repo_id}' 失败: {e}")
        print("\n请确认以下几点:")
        print(f"1. 如果 '{input_repo_id}' 是本地路径，请确保它相对于 $LEROBOT_HOME 或 ~/.cache/huggingface/lerobot/ 存在且有效。")
        print(f"2. 如果 '{input_repo_id}' 是 Hugging Face Hub ID，请确保 ID 正确且你有权访问（如果是私有仓库）。")
        print("3. 检查你的网络连接（如果需要从 Hub 下载）。")
        return # 加载失败则退出

    # --- 2. 获取元数据并创建输出数据集 ---
    print("正在从输入数据集中提取元数据（或使用用户配置）...")
    # 注意：你可能想根据加载的 input_ds.info 字典来精确设置 FPS, ROBOT_TYPE,
    # 以及特征的 shapes/dtypes，而不是完全依赖上面的手动配置。
    # 为简单起见，我们暂时使用上面手动配置的值。

    print(f"正在创建新的输出数据集: {output_repo_id}")
    output_ds = LeRobotDataset.create(
        repo_id=output_repo_id, # 使用输出仓库 ID
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features=OUTPUT_DATASET_FEATURES, # 使用为输出定义的特征
        image_writer_threads=10, # 根据你的系统资源调整
        image_writer_processes=5,  # 根据你的系统资源调整
    )
    print("输出数据集实例创建完成。")

    # --- 3. 迭代、读取和写入数据 ---
    num_episodes = len(input_ds)
    print(f"开始转换 {num_episodes} 个 episodes...")
    total_steps_processed = 0

    # 检查必要的属性是否存在
    if not hasattr(input_ds, 'episode_data_index') or \
       'from' not in input_ds.episode_data_index or \
       'to' not in input_ds.episode_data_index:
        print("错误: 输入数据集中未找到 'episode_data_index' 属性或其必要的 'from'/'to' 键。无法按 episode 处理。")
        return
    if not hasattr(input_ds, 'episodes'):
        print("错误: 输入数据集中未找到 'episodes' 属性。无法获取任务描述。")
        # 或者可以设置一个默认任务描述，如果任务描述不重要的话
        # use_default_task = True
        return # 或者根据情况决定是否继续

    # 通过索引迭代 episodes
    # ... (前面的代码) ...

    # 通过索引迭代 episodes
    for episode_index in tqdm(range(num_episodes), desc="处理 Episodes"):
        try:
            # --- 修改开始 ---
            start_index = input_ds.episode_data_index['from'][episode_index]
            end_index = input_ds.episode_data_index['to'][episode_index]
            episode_data_slice = input_ds[start_index:end_index]

            # 获取任务描述
            task_description = f"Episode {episode_index} 的默认任务 (信息缺失或格式错误)" # 先设置默认值
            if input_ds.episodes is not None and episode_index in input_ds.episodes:
                 episode_meta = input_ds.episodes[episode_index]
                 # 检查 "tasks" 键是否存在，并且其值是一个非空列表
                 if "tasks" in episode_meta and isinstance(episode_meta["tasks"], list) and len(episode_meta["tasks"]) > 0:
                     # 从 "tasks" 列表中获取第一个元素作为任务描述
                     task_description = str(episode_meta["tasks"][0]) # 确保是字符串
                 else:
                     # 如果 "tasks" 键不存在或是空列表，打印警告并使用默认值
                     print(f"  警告: Episode {episode_index} 在 episodes.jsonl 中的 'tasks' 键缺失或为空列表。")

            else:
                 # 如果 episodes 为 None 或当前 episode_index 不在里面，使用默认任务描述 (已在上面设置)
                 print(f"  警告: Episode {episode_index} 的元数据在 episodes.jsonl 中未找到。")


            episode_data = episode_data_slice
            # --- 修改结束 ---

            # ... (后续处理 episode_data 的代码) ...

            # 检查提取的任务描述是否是有效字符串
            if not isinstance(task_description, str) or not task_description:
                 print(f"  严重警告: Episode {episode_index} 的任务描述处理后无效 ('{task_description}')。将使用最终默认值。")
                 task_description = f"Episode {episode_index}_invalid_task"


            # 保存已完成的 episode 到输出数据集中
            # 在调用 save_episode 前确保 task_description 是字符串
            output_ds.save_episode(task=str(task_description))
            total_steps_processed += num_steps

        # ... (except 块) ...

        # except OutOfMemory# Error:
        #      print(f"\n错误: 处理 Episode {episode_i# ndex} 时发生内存不足错误！")
        #      print("这个 episode 可能太# 长或分辨率太高，无法一次性加载所有帧。")
        #      print("跳过此 episode。考虑增加可用内存或修改脚本以分块处理。")
        #      continue#  # 继续处理下一个 episode
        # except KeyError as e:
        #      print(f"\n错误: 处理 Episode {episode_in# dex} 时发生键错误: {e}")
        #      print("这可能意味着 FEATURE_MAP 中的键在输入数据集的 episode_data_slice 中不存在，或者 input_ds# .episodes 结构不正确。")
        #      print(f"检查 FEATURE_MA# P: {FEATURE_MAP}")
        #      if 'episode_data' in locals()#  and episode_data:
        #          print(f"检查 episode_data 的键: {list(episo# de_data.keys())}")
        #      print("跳过此 episode。")
        #      continue
        except Exception as e: # <--- 让通用的 Exception 捕获所有错误
            import traceback
            print(f"\n错误: 处理 Episode {episode_index} 时发生意外错误: {e}")
            # 检查错误类型是否是 KeyError，并特别处理一下打印信息
            if isinstance(e, KeyError):
                 print(f"  错误详情 (KeyError): 试图访问键 '{e.args[0]}'") # e.args[0] 通常是缺失的键名
                 print(f"  检查 FEATURE_MAP: {FEATURE_MAP}")
                 if 'episode_data' in locals() and episode_data:
                      print(f"  检查 episode_data 的键: {list(episode_data.keys())}")
            traceback.print_exc() # 打印完整的错误追踪信息
            print("跳过此 episode。")
            continue

    # ... 脚本的其余部分 ...

    print(f"\n已处理 {num_episodes} 个 episodes，总计 {total_steps_processed} 个步骤。")

    # --- 4. 整合新数据集 ---
    print("正在整合 (Consolidating) 新的数据集...")
    # 这会最终确定数据集结构，写入索引文件等。
    output_ds.consolidate(run_compute_stats=False) # 暂时不计算统计数据
    print("数据集整合完成。")
    print(f"新的数据集已保存在本地: {output_path}")

    # --- 5. 可选：推送到 Hub ---
    if push_to_hub:
        print(f"准备将数据集 '{output_repo_id}' 推送到 Hugging Face Hub...")
        try:
            # 确保你已经通过 `huggingface-cli login` 登录
            output_ds.push_to_hub(
                tags=[ROBOT_TYPE, "frame-by-frame", "libero", "converted"],
                private=False,
                push_videos=False, # 已经是逐帧，通常不需要再生成预览视频
                license="apache-2.0",
            )
            print("推送完成。")
        except Exception as e:
            print(f"错误: 推送到 Hub 时出错: {e}")
            print("请确保你已登录 (`huggingface-cli login`) 并且拥有正确的权限。")


if __name__ == "__main__":
    # 确保安装了必要的库
    # pip install lerobot tyro tqdm numpy pyarrow torch av huggingface_hub
    # uv pip install lerobot tyro tqdm numpy pyarrow torch av huggingface_hub
    print("运行数据集转换脚本...")
    tyro.cli(main)
    print("脚本执行完毕。")
























# """
# Minimal example script for converting a dataset to LeRobot format.

# We use the Libero dataset (stored in RLDS) for this example, but it can be easily
# modified for any other data you have saved in a custom format.

# Usage:
# uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

# If you want to push your dataset to the Hugging Face Hub, you can use the following command:
# uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

# Note: to run the script, you need to install tensorflow_datasets:
# `uv pip install tensorflow tensorflow_datasets`

# You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
# The resulting dataset will get saved to the $LEROBOT_HOME directory.
# Running this conversion script will take approximately 30 minutes.
# """

# import shutil

# from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# import tensorflow_datasets as tfds
# import tyro

# REPO_NAME = "your_hf_username/libero"  # Name of the output dataset, also used for the Hugging Face Hub
# RAW_DATASET_NAMES = [
#     "libero_10_no_noops",
#     "libero_goal_no_noops",
#     "libero_object_no_noops",
#     "libero_spatial_no_noops",
# ]  # For simplicity we will combine multiple Libero datasets into one training dataset


# def main(data_dir: str, *, push_to_hub: bool = False):
#     # Clean up any existing dataset in the output directory
#     output_path = LEROBOT_HOME / REPO_NAME
#     if output_path.exists():
#         shutil.rmtree(output_path)

#     # Create LeRobot dataset, define features to store
#     # OpenPi assumes that proprio is stored in `state` and actions in `action`
#     # LeRobot assumes that dtype of image data is `image`
#     dataset = LeRobotDataset.create(
#         repo_id=REPO_NAME,
#         robot_type="panda",
#         fps=10,
#         features={
#             "image": {
#                 "dtype": "image",
#                 "shape": (256, 256, 3),
#                 "names": ["height", "width", "channel"],
#             },
#             "wrist_image": {
#                 "dtype": "image",
#                 "shape": (256, 256, 3),
#                 "names": ["height", "width", "channel"],
#             },
#             "state": {
#                 "dtype": "float32",
#                 "shape": (8,),
#                 "names": ["state"],
#             },
#             "actions": {
#                 "dtype": "float32",
#                 "shape": (7,),
#                 "names": ["actions"],
#             },
#         },
#         image_writer_threads=10,
#         image_writer_processes=5,
#     )

#     # Loop over raw Libero datasets and write episodes to the LeRobot dataset
#     # You can modify this for your own data format
#     for raw_dataset_name in RAW_DATASET_NAMES:
#         raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
#         for episode in raw_dataset:
#             for step in episode["steps"].as_numpy_iterator():
#                 dataset.add_frame(
#                     {
#                         "image": step["observation"]["image"],
#                         "wrist_image": step["observation"]["wrist_image"],
#                         "state": step["observation"]["state"],
#                         "actions": step["action"],
#                     }
#                 )
#             dataset.save_episode(task=step["language_instruction"].decode())

#     # Consolidate the dataset, skip computing stats since we will do that later
#     dataset.consolidate(run_compute_stats=False)

#     # Optionally push to the Hugging Face Hub
#     if push_to_hub:
#         dataset.push_to_hub(
#             tags=["libero", "panda", "rlds"],
#             private=False,
#             push_videos=True,
#             license="apache-2.0",
#         )


# if __name__ == "__main__":
#     tyro.cli(main)
