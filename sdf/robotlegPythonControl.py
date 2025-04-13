import pandas as pd
import time
import numpy as np
from gz.transport13 import Node
from gz.msgs10.joint_trajectory_pb2 import JointTrajectory
from gz.msgs10.duration_pb2 import Duration  # 修改這一行，使用 Gazebo 的 Duration 類型

def read_excel_data(file_path):
    # 讀取Excel檔案
    df = pd.read_excel(file_path, usecols=['t','R1 (rad)', 'R2 (rad)', 'R3 (rad)', 'R4 (rad)', 'R5 (rad)', 'R6 (rad)', 'L1 (rad)', 'L2 (rad)', 'L3 (rad)',
       'L4 (rad)', 'L5 (rad)', 'L6 (rad)'])
    # 將列重新命名為 time, R1-R6, L1-L6
    df.columns = ['time', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6']
    return df

def publish_trajectory(node, topic, times, positions):
    # 初始化JointTrajectory訊息
    traj_msg = JointTrajectory()
    traj_msg.joint_names.extend([
        "left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_joint6",
        "right_joint1", "right_joint2", "right_joint3", "right_joint4", "right_joint5", "right_joint6"
    ])

    # 創建軌跡點
    for t, pos in zip(times, positions):
        point = traj_msg.points.add()  # 添加新的Point訊息
        # 設置time_from_start
        duration = Duration()
        duration.sec = int(t)  # 使用 Gazebo 的 sec 屬性
        duration.nsec = int((t - int(t)) * 1e9)  # 使用 Gazebo 的 nsec 屬性
        point.time_from_start.CopyFrom(duration)
        # 設置關節角度
        point.positions.extend(pos)

    # 創建Publisher
    pub = node.advertise(topic, JointTrajectory)

    # 確保Publisher已初始化
    time.sleep(1)

    # 發送軌跡訊息
    success = pub.publish(traj_msg)
    if success:
        print("軌跡發送成功")
    else:
        print("軌跡發送失敗")

def main():
    # Excel檔案路徑
    excel_file = "/home/tingyi/Downloads/step.xlsx"
    
    # 讀取數據
    df = read_excel_data(excel_file)
    times = df['time'].to_numpy()
    # 將關節角度按順序排列：L1-L6, R1-R6
    positions = df[['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']].to_numpy()

    # 初始化gz-transport節點
    node = Node()
    topic = "/model/robot/joint_trajectory"

    # 發送軌跡
    publish_trajectory(node, topic, times, positions)

    # 等待軌跡執行完成
    total_duration = times[-1]
    print(f"等待軌跡執行完成 ({total_duration} 秒)...")
    time.sleep(total_duration)

if __name__ == "__main__":
    main()