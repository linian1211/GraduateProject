from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class LinkNode:
    # 保持與你的原始定義一致（省略重複部分）
    j: int  # Self ID
    sister: int = None  # Sister ID
    child: int = None  # Child ID
    mother: int = None  # Parent ID

    p: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Position in World Coordinates
    R: np.ndarray = field(default_factory=lambda: np.eye(3))  # Attitude in World Coordinates

    v: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Linear Velocity
    w: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Angular Velocity

    q: float = 0.0  # Joint Angle
    dq: float = 0.0  # Joint Velocity
    ddq: float = 0.0  # Joint Acceleration
    
    a: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Joint Axis Vector (Relative to Parent)
    b: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Joint Relative Position (Relative to Parent)

    vertex: list = field(default_factory=list)  # Shape (Vertex Information, Link Local)
    face: list = field(default_factory=list)  # Shape (Point Connections)

    m: float = 0.0  # Mass
    c: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Center of Mass (Link Local)
    I: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # Moment of Inertia (Link Local)

def create_joint_motors(base_node: LinkNode, num_dofs: int, parent_id: int, child_ids: list, base_offset=np.zeros(3)):
    """
    創建指定數量的馬達（自由度）作為連續的 LinkNode。
    第一個馬達相對於母節點有偏移，其他馬達相對於前一個馬達偏移為 0。
    關節軸 \(a\) 按照右手定則，順時針為正方向（螺絲鎖緊方向）。
    """
    motors = []
    current_parent = base_node
    
    joint_axes = [
        np.array([1, 0, 0]),  # X 軸旋轉（俯仰，順時針為正）
        np.array([0, 1, 0]),  # Y 軸旋轉（偏航，順時針為正）
        np.array([0, 0, 1])   # Z 軸旋轉（滾轉，順時針為正）
    ]
    
    # 創建第一個馬達，相對於母節點有偏移
    if num_dofs > 0:
        first_motor = LinkNode(
            j=child_ids[0],
            mother=parent_id,
            child=child_ids[1] if num_dofs > 1 else None,
            p=current_parent.p + base_offset,  # 第一個馬達的位置包括偏移
            R=np.eye(3),
            m=0.0,
            c=np.zeros(3),
            I=np.zeros((3, 3)),
            a=joint_axes[0],
            b=base_offset  # 第一個馬達的 b 向量為 base_offset
        )
        motors.append(first_motor)
        print(f"Created Node {first_motor.j} - Position: {first_motor.p}, Axis: {first_motor.a}, Offset: {first_motor.b}")
    
    # 創建其餘馬達，相對於前一個馬達偏移為 0
    for i in range(1, num_dofs):
        if i >= len(joint_axes):
            raise ValueError("Not enough predefined joint axes. Please extend joint_axes list.")
        
        previous_motor = motors[i-1]
        motor = LinkNode(
            j=child_ids[i],
            mother=child_ids[i-1],
            child=child_ids[i+1] if i < num_dofs - 1 else None,
            p=previous_motor.p.copy(),  # 共享相同的位置（無偏移）
            R=np.eye(3),
            m=0.0,
            c=np.zeros(3),
            I=np.zeros((3, 3)),
            a=joint_axes[i],
            b=np.zeros(3)  # 後續馬達相對於前一個馬達的偏移為 0
        )
        motors.append(motor)
        print(f"Created Node {motor.j} - Position: {motor.p}, Axis: {motor.a}, Offset: {motor.b}")
    
    return motors

def create_leg(base_position=np.array([0, 0, 100]), direction="down_z", num_joints=2, joint_length=30, hip_dofs=3, ankle_dofs=2):
    """
    創建一隻腿的連桿鏈，包括髖部三自由度的馬達、單一膝蓋關節（沿 Y+）、小腿、腳和腳踝兩自由度（沿 X+ 和 Y+）。
    """
    links = []
    
    # 1. 創建髖部（基座），位置在 [0, 0, 100]
    hip = LinkNode(
        j=0,
        p=base_position,
        R=np.eye(3),
        m=10.0,
        a=np.array([0, 0, 1]),  # 初始關節軸（可以由馬達覆蓋）
        b=np.array([0, 0, 0])
    )
    links.append(hip)
    print(f"Created Hip (Node 0) - Position: {hip.p}, Axis: {hip.a}, Offset: {hip.b}")
    
    # 2. 定義髖部馬達的基點位置 [0, -15, 100]，並設置相對偏移
    hip_motor_offset = np.array([0, -15, 0])  # 從基座到第一個馬達的偏移
    
    # 創建髖部三自由度的馬達（沿 X+, Y+, Z+，順時針為正）
    hip_motors = create_joint_motors(
        hip,  # 直接使用基座作為母節點
        hip_dofs,
        parent_id=0,
        child_ids=[1, 2, 3],
        base_offset=hip_motor_offset
    )
    links.extend(hip_motors)
    
    # 3. 創建腿的其餘部分（從大腿開始）
    current_position = hip_motors[0].p.copy()  # 從第一個馬達的位置開始
    current_parent_id = 3  # 最後一個馬達的 ID 作為母節點
    
    # 根據方向計算每個關節的位置和相對位置
    if direction == "down_z":
        direction_vector = np.array([0, 0, -1])
        b_relative = np.array([0, 0, -joint_length])
    elif direction == "down_y":
        direction_vector = np.array([0, -1, 0])
        b_relative = np.array([0, -joint_length, 0])
    else:
        raise ValueError("Unsupported direction. Use 'down_z' or 'down_y'.")
    
    # 大腿（母節點為最後一個馬達）
    thigh = LinkNode(
        j=current_parent_id + 1,
        mother=current_parent_id,
        child=current_parent_id + 2,
        p=current_position + (direction_vector * joint_length),
        R=np.eye(3),
        m=5.0,
        a=np.array([0, 1, 0]),  # 膝蓋關節沿 Y+ 方向
        b=b_relative
    )
    links.append(thigh)
    print(f"Created Thigh (Node 4) - Position: {thigh.p}, Axis: {thigh.a}, Offset: {thigh.b}")
    
    # 小腿（母節點為大腿）
    shank = LinkNode(
        j=current_parent_id + 2,
        mother=current_parent_id + 1,
        child=current_parent_id + 3,
        p=thigh.p + (direction_vector * joint_length),
        R=np.eye(3),
        m=3.0,
        a=np.array([0, 0, 0]),  # 小腿無獨立關節軸（腳踝關節由馬達處理）
        b=b_relative
    )
    links.append(shank)
    print(f"Created Shank (Node 5) - Position: {shank.p}, Axis: {shank.a}, Offset: {shank.b}")
    
    # 腳（母節點為小腿，長度縮短至 5 單位）
    foot = LinkNode(
        j=current_parent_id + 3,
        mother=current_parent_id + 2,
        child=None,
        p=shank.p + (direction_vector * 5),  # 腳長度縮短至 5 單位
        R=np.eye(3),
        m=2.0,
        a=np.array([0, 0, 0]),  # 腳初始無關節軸（由馬達覆蓋）
        b=(direction_vector * 5)  # 相對小腿的向下位置，長度為 5
    )
    links.append(foot)
    print(f"Created Foot (Node 6) - Position: {foot.p}, Axis: {foot.a}, Offset: {foot.b}")
    
    # 4. 創建腳踝的兩個自由度馬達（沿 X+ 和 Y+，順時針為正）
    ankle_motors = create_joint_motors(
        foot,
        ankle_dofs,
        parent_id=foot.j,
        child_ids=[current_parent_id + 4, current_parent_id + 5]  # 腳踝馬達的 ID
    )
    links.extend(ankle_motors)
    
    return links

def forward_kinematics(links: list, joint_angles: list):
    """
    執行正運動學，根據給定的關節角度更新所有 LinkNode 的位置和姿態。
    添加調試信息以檢查角度應用情況。
    
    Args:
        links: LinkNode 列表，表示連桿鏈
        joint_angles: 關節角度列表（弧度），按節點 ID 順序對應每個可動關節
    
    Returns:
        updated_links: 更新後的 LinkNode 列表，包含新的位置 p 和姿態 R
    """
    updated_links = [LinkNode(**link.__dict__) for link in links]  # 深拷貝
    
    root = next(link for link in updated_links if link.mother is None)
    root.p = root.p.copy()
    root.R = np.eye(3)
    
    angle_idx = 0
    
    def update_node(node: LinkNode, parent: LinkNode = None):
        nonlocal angle_idx
        
        if node.mother is not None:
            parent = next(l for l in updated_links if l.j == node.mother)
            
            # 打印當前節點的狀態以調試
            print(f"Processing Node {node.j} - Mother: {node.mother}, Axis: {node.a}, Current Angle Index: {angle_idx}")
            
            # 嘗試應用角度，即使 a 可能為零
            if angle_idx < len(joint_angles):
                joint_angle = joint_angles[angle_idx]
                node.q = joint_angle
                
                print(f"Attempting to apply angle {np.degrees(joint_angle)} degrees to Node {node.j} with axis {node.a}")
                
                # 打印父節點的旋轉以調試
                print(f"Parent Node {parent.j} Rotation:")
                print(parent.R)
                
                a_world = parent.R @ node.a
                if np.linalg.norm(a_world) == 0:
                    print(f"Warning: Zero axis for Node {node.j}, forcing default axis [0, 0, 1]")
                    a_world = np.array([0, 0, 1])  # 臨時使用 Z 軸作為預設
                a_world = a_world / np.linalg.norm(a_world) if np.linalg.norm(a_world) > 0 else np.array([0, 0, 1])
                
                print(f"Axis in world coordinates (a_world): {a_world}")
                
                K = np.array([
                    [0, -a_world[2], a_world[1]],
                    [a_world[2], 0, -a_world[0]],
                    [-a_world[1], a_world[0], 0]
                ])
                R_joint = np.eye(3) + np.sin(joint_angle) * K + (1 - np.cos(joint_angle)) * (K @ K)
                
                # 打印旋轉矩陣以調試
                print(f"Rotation matrix for Node {node.j}:")
                print(R_joint)
                
                node.R = parent.R @ R_joint
            else:
                node.R = parent.R.copy()
            
            # 打印更新前的位置和旋轉
            print(f"Node {node.j} before position update - Position: {node.p}, Rotation:")
            print(node.R)
            
            b_world = parent.R @ node.b
            print(f"b_world for Node {node.j}: {b_world}")
            
            node.p = parent.p + b_world
            
            # 打印更新後的位置
            print(f"Node {node.j} after position update - Position: {node.p}, Rotation:")
            print(node.R)
            
            # 每次處理節點後增加 angle_idx（確保順序正確）
            if angle_idx < len(joint_angles):
                angle_idx += 1
        
        if node.child is not None:
            child = next(l for l in updated_links if l.j == node.child)
            update_node(child, node)
    
    update_node(root)
    
    return updated_links

    
def PlotLinkChain(links: list):
    """ 在 3D 空間中顯示連桿鏈 """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for link in links:
        # 畫節點位置
        ax.scatter(link.p[0], link.p[1], link.p[2], 
                  c='r', marker='o', label=f"Node {link.j}" if link.j == 0 else "")

        # 如果有母節點，畫出連桿（從母節點到子節點）
        if link.mother is not None:
            mother = next(l for l in links if l.j == link.mother)
            # 計算從母節點到當前節點的向量
            b_world = link.R @ link.b
            end_point = mother.p + b_world

            # 畫連桿線段
            ax.plot([mother.p[0], end_point[0]], 
                    [mother.p[1], end_point[1]], 
                    [mother.p[2], end_point[2]], 
                    'b-', label='Link' if link.j == 1 else "")

        # 畫關節軸向量 a（縮放到可視化長度，例如 10 單位）
        if np.any(link.a != 0):
            a_world = link.R @ link.a * 10
            ax.quiver(link.p[0], link.p[1], link.p[2],
                      a_world[0], a_world[1], a_world[2],
                      color='g', label='Joint Axis' if link.j == 0 else "")

    # 設定座標軸標籤
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 設定視圖範圍（擴展以顯示可能的旋轉）
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([0, 200])

    # 顯示標籤
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()

# 測試範例
if __name__ == "__main__":
    # 創建一隻腿（包含髖部三自由度、單一膝蓋沿 Y+、腳踝沿 X+ 和 Y+ 的兩個自由度）
    leg_links = create_leg(base_position=np.array([0, 0, 100]), direction="down_z", num_joints=2, joint_length=30, hip_dofs=3, ankle_dofs=2)
    
    # 定義關節角度（弧度），對應每個可動關節（3 個髖部馬達 + 1 個膝蓋 + 2 個腳踝馬達，共 6 個角度）
    joint_angles = [
        np.radians(90),  # 髖部馬達 1（X+）
        np.radians(90),  # 髖部馬達 2（Y+）
        np.radians(90),  # 髖部馬達 3（Z+）
        np.radians(45),  # 膝蓋關節（Y+）
        np.radians(30),  # 腳踝馬達 1（X+）
        np.radians(30)   # 腳踝馬達 2（Y+）
    ]
    
    # 應用正運動學
    updated_links = forward_kinematics(leg_links, joint_angles)
    
    # 可視化更新後的結構
    PlotLinkChain(updated_links)
    
    # 打印更新後的節點位置以供檢查
    for link in updated_links:
        print(f"Node {link.j} - Position: {link.p}, Rotation:")
        print(link.R)