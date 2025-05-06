import time
from gz.transport13 import Node #Node 是 Gazebo 中的「通訊節點」，用來發送（publish）或接收（subscribe）訊息，類似 ROS 中的 rospy.Node
from gz.msgs10.joint_trajectory_pb2 import JointTrajectory
from google.protobuf.duration_pb2 import Duration

def publish_joint_trajectory():
    # 初始化gz-transport節點
    node = Node()

    # 定義topic
    topic = "/model/RR_position_control/joint_trajectory"

    # 創建JointTrajectory訊息
    msg = JointTrajectory()

    # 設置關節名稱
    msg.joint_names.extend(["RR_position_control_joint1", "RR_position_control_joint2"])

    # 定義軌跡點（與你的終端命令一致）
    # Point 1: t=0.25s, joint1=-0.7854, joint2=1.5708
    point1 = msg.points.add()
    point1.positions.extend([-0.7854, 1.5708])
    point1.time_from_start.sec = 0
    point1.time_from_start.nsec = 250000000

    # Point 2: t=0.5s, joint1=-1.5708, joint2=0
    point2 = msg.points.add()
    point2.positions.extend([-1.5708, 0.0])
    point2.time_from_start.sec = 0
    point2.time_from_start.nsec = 500000000

    # Point 3: t=1.0s, joint1=-1.5708, joint2=-1.5708
    point3 = msg.points.add()
    point3.positions.extend([-1.5708, -1.5708])
    point3.time_from_start.sec = 1
    point3.time_from_start.nsec = 0

    # 創建發布者
    pub = node.advertise(topic, JointTrajectory)

    # 確保發布者初始化
    time.sleep(1)

    # 發布訊息
    print(f"Publishing joint trajectory to {topic}")
    if pub.publish(msg):
        print("Trajectory sent successfully!")
    else:
        print("Failed to send trajectory.")

    # 保持運行以觀察效果
    time.sleep(2)

if __name__ == "__main__":
    try:
        publish_joint_trajectory()
    except KeyboardInterrupt:
        print("Program terminated by user.")