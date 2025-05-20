#!/usr/bin/env python3

#subscribe to /joint_states and /clock topics
#plot the joint angles in real-time using matplotlib
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rosgraph_msgs.msg import Clock
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

class JointAnglePlotter(Node):
    def __init__(self):
        super().__init__('joint_angle_plotter')
        self.get_logger().info('Joint Angle Plotter Node Started')

        # 儲存關節數據與時間（使用deque限制數據長度）
        self.max_points = 1000  # 最多儲存1000個數據點
        self.joint_data = {}  # 儲存各關節的角度數據
        self.time_data = deque(maxlen=self.max_points)  # 儲存時間戳
        self.sim_time = 0.0  # 模擬時間

        # 訂閱 /joint_states topic
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # 訂閱 /clock topic 以獲取模擬時間
        self.clock_sub = self.create_subscription(
            Clock,
            '/clock',
            self.clock_callback,
            10
        )

        # 設定matplotlib即時繪圖
        plt.ion()  # 啟用互動模式
        self.fig, self.ax = plt.subplots()
        self.lines = {}  # 儲存各關節的繪圖線條
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Joint Angle (rad)')
        self.ax.set_title('Real-time Joint Angles')
        self.ax.grid(True)

    def joint_state_callback(self, msg):
        self.get_logger().info(f'Received joint states: names={msg.name}, positions={msg.position}')
        # 處理關節狀態數據
        for name, position in zip(msg.name, msg.position):
            if name not in self.joint_data:
                # 初始化新關節的數據儲存與繪圖線條
                self.joint_data[name] = deque(maxlen=self.max_points)
                self.lines[name], = self.ax.plot([], [], label=name)
                self.ax.legend()
                self.get_logger().info(f'Added joint: {name}')
            self.joint_data[name].append(position)

        # 記錄當前模擬時間
        self.time_data.append(self.sim_time)

        # 更新繪圖
        self.update_plot()

    def clock_callback(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9
        self.get_logger().info(f'Received sim time: {self.sim_time}')
        # 更新模擬時間（單位：秒）
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def update_plot(self):
        self.get_logger().info('Plot updated')
        # 更新每條關節角度曲線
        for name in self.joint_data:
            self.lines[name].set_xdata(list(self.time_data))
            self.lines[name].set_ydata(list(self.joint_data[name]))

        # 動態調整X軸與Y軸範圍
        if self.time_data:
            self.ax.set_xlim(max(0, self.time_data[-1] - 10), self.time_data[-1] + 1)
            all_angles = [angle for data in self.joint_data.values() for angle in data]
            if all_angles:
                self.ax.set_ylim(min(all_angles) - 0.5, max(all_angles) + 0.5)

        # 重新繪製圖表
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main(args=None):
    rclpy.init(args=args)
    plotter = JointAnglePlotter()
    try:
        rclpy.spin(plotter)
    except KeyboardInterrupt:
        plotter.get_logger().info('Shutting down Joint Angle Plotter')
    finally:
        plt.close('all')
        plotter.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()