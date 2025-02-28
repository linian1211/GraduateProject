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

def create_leg(base_position=np.array([0, 0, 120]), direction="down_z", num_joints=2, joint_length=30, hip_dofs=3, ankle_dofs=2):
    """
    創建一隻腿的連桿鏈，包括身體中心節點、髖部三自由度節點、單一膝蓋關節（沿 Y+）、小腿（無自由度）、腳踝兩自由度（沿 X+ 和 Y+）。
    """
    links = []
    
    # 1. 創建身體中心節點（根節點），位置在 [0, 0, 120]
    body_center = LinkNode(
        j=0,  # 身體中心節點
        p=base_position,  # 位置在 [0, 0, 120]
        R=np.eye(3),
        m=20.0,  # 假設身體中心的質量較大
        a=np.array([0, 0, 0]),  # 無關節軸（固定節點）
        b=np.array([0, 0, 0]),
        child=1  # 子節點為髖部
    )
    links.append(body_center)
    print(f"Created Body Center (Node 0) - Position: {body_center.p}, Axis: {body_center.a}, Offset: {body_center.b}")
    
    # 2. 創建髖部（基座），位置在 [0, 0, 100]，母節點為身體中心
    hip_position = np.array([0, 0, 100])  # 髖部位置調整為 [0, 0, 100]
    hip_offset = np.array([0, 0, -20])  # 從身體中心 [0, 0, 120] 到髖部 [0, 0, 100] 的偏移
    
    hip = LinkNode(
        j=1,  # 髖部為 j=1
        mother=0,  # 母節點為身體中心
        child=2,  # 子節點為第一個髖部自由度節點
        p=body_center.p + hip_offset,  # 位置在 [0, 0, 100]
        R=np.eye(3),
        m=10.0,
        a=np.array([0, 0, 0]),  # 髖部基座無關節軸
        b=hip_offset  # 相對於身體中心的偏移
    )
    links.append(hip)
    print(f"Created Hip (Node 1) - Position: {hip.p}, Axis: {hip.a}, Offset: {hip.b}")
    
    # 3. 創建髖部三自由度節點（替代原來的馬達），位置在 [0, -15, 100]，沿 X+、Y+、Z+
    hip_joint_position = np.array([0, -15, 100])  # 大腿上方髖部關節位置
    hip_joint_offset = np.array([0, -15, 0])  # 從髖部 [0, 0, 100] 到 [0, -15, 100] 的偏移
    
    # 第一個髖部自由度節點（沿 X+）
    hip1 = LinkNode(
        j=2,
        mother=1,  # 母節點為髖部
        child=3,  # 子節點為下一個自由度節點
        p=hip.p + hip_joint_offset,  # 位置在 [0, -15, 100]
        R=np.eye(3),
        m=0.0,  # 無質量（僅表示旋轉關節）
        a=np.array([1, 0, 0]),  # 沿 X+ 軸旋轉
        b=hip_joint_offset  # 相對於髖部的偏移
    )
    links.append(hip1)
    print(f"Created Hip Joint 1 (Node 2) - Position: {hip1.p}, Axis: {hip1.a}, Offset: {hip1.b}")
    
    # 第二個髖部自由度節點（沿 Y+）
    hip2 = LinkNode(
        j=3,
        mother=2,  # 母節點為前一個自由度節點
        child=4,  # 子節點為下一個自由度節點
        p=hip1.p,  # 共享相同位置 [0, -15, 100]
        R=np.eye(3),
        m=0.0,
        a=np.array([0, 1, 0]),  # 沿 Y+ 軸旋轉
        b=np.zeros(3)  # 相對於前一個節點無偏移
    )
    links.append(hip2)
    print(f"Created Hip Joint 2 (Node 3) - Position: {hip2.p}, Axis: {hip2.a}, Offset: {hip2.b}")
    
    # 第三個髖部自由度節點（沿 Z+）
    hip3 = LinkNode(
        j=4,
        mother=3,  # 母節點為前一個自由度節點
        child=5,  # 子節點為大腿
        p=hip2.p,  # 共享相同位置 [0, -15, 100]
        R=np.eye(3),
        m=0.0,
        a=np.array([0, 0, 1]),  # 沿 Z+ 軸旋轉
        b=np.zeros(3)  # 相對於前一個節點無偏移
    )
    links.append(hip3)
    print(f"Created Hip Joint 3 (Node 4) - Position: {hip3.p}, Axis: {hip3.a}, Offset: {hip3.b}")
    
    # 4. 創建腿的其餘部分（從大腿開始）
    current_position = hip3.p.copy()  # 從最後一個髖部自由度節點的位置開始
    current_parent_id = 4  # 最後一個髖部自由度節點的 ID 作為母節點
    
    # 根據方向計算每個關節的位置和相對位置
    if direction == "down_z":
        direction_vector = np.array([0, 0, -1])
        b_relative = np.array([0, 0, -joint_length])
    elif direction == "down_y":
        direction_vector = np.array([0, -1, 0])
        b_relative = np.array([0, -joint_length, 0])
    else:
        raise ValueError("Unsupported direction. Use 'down_z' or 'down_y'.")
    
    # 大腿（母節點為最後一個髖部自由度節點）
    thigh = LinkNode(
        j=current_parent_id + 1,  # 現在是大腿，j=5
        mother=current_parent_id,
        child=current_parent_id + 2,  # 子節點為小腿
        p=current_position + (direction_vector * joint_length),
        R=np.eye(3),
        m=5.0,
        a=np.array([0, 1, 0]),  # 膝蓋關節沿 Y+ 方向
        b=b_relative
    )
    links.append(thigh)
    print(f"Created Thigh (Node 5) - Position: {thigh.p}, Axis: {thigh.a}, Offset: {thigh.b}")
    
    # 小腿（母節點為大腿，無自由度）
    shank = LinkNode(
        j=current_parent_id + 2,  # 現在是小腿，j=6
        mother=current_parent_id + 1,
        child=current_parent_id + 3,  # 子節點為腳踝 1
        p=thigh.p + (direction_vector * joint_length),
        R=np.eye(3),
        m=3.0,
        a=np.array([0, 0, 0]),  # 小腿無自由度，固定節點
        b=b_relative
    )
    links.append(shank)
    print(f"Created Shank (Node 6) - Position: {shank.p}, Axis: {shank.a}, Offset: {shank.b}")
    
    # 5. 創建腳踝的兩個自由度節點（沿 X+ 和 Y+，順時針為正），母節點為小腿
    # 第一個腳踝自由度節點（沿 X+，節點 7）
    ankle1 = LinkNode(
        j=current_parent_id + 3,  # 現在是腳踝 1，j=7
        mother=current_parent_id + 2,  # 母節點為小腿
        child=current_parent_id + 4,  # 子節點為第二個腳踝自由度節點
        p=shank.p,  # 共享小腿的位置 [0, -15, 40]
        R=np.eye(3),
        m=0.0,  # 無質量（僅表示旋轉關節）
        a=np.array([1, 0, 0]),  # 沿 X+ 軸旋轉
        b=np.zeros(3)  # 相對於小腿無偏移
    )
    links.append(ankle1)
    print(f"Created Ankle Joint 1 (Node 7) - Position: {ankle1.p}, Axis: {ankle1.a}, Offset: {ankle1.b}")
    
    # 第二個腳踝自由度節點（沿 Y+，節點 8）
    ankle2 = LinkNode(
        j=current_parent_id + 4,  # 現在是腳踝 2，j=8
        mother=current_parent_id + 3,  # 母節點為前一個腳踝自由度節點
        child=None,
        p=ankle1.p,  # 共享相同位置 [0, -15, 40]
        R=np.eye(3),
        m=0.0,
        a=np.array([0, 1, 0]),  # 沿 Y+ 軸旋轉
        b=np.zeros(3)  # 相對於前一個節點無偏移
    )
    links.append(ankle2)
    print(f"Created Ankle Joint 2 (Node 8) - Position: {ankle2.p}, Axis: {ankle2.a}, Offset: {ankle2.b}")
    
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
            
            # 嘗試應用角度，僅對具有非零 a 的節點應用
            if angle_idx < len(joint_angles) and np.any(node.a != 0):
                joint_angle = joint_angles[angle_idx]
                node.q = joint_angle
                
                print(f"Attempting to apply angle {np.degrees(joint_angle)} degrees to Node {node.j} with axis {node.a}")
                
                # 打印父節點的旋轉以調試
                print(f"Parent Node {parent.j} Position: {parent.p}, Rotation:")
                print(parent.R)
                
                a_world = parent.R @ node.a
                if np.linalg.norm(a_world) == 0:
                    print(f"Warning: Zero axis for Node {node.j}, forcing default axis [0, 0, 1]")
                    a_world = np.array([0, 0, 1])  # 臨時使用 Z 軸作為預設
                a_world = a_world / np.linalg.norm(a_world) if np.linalg.norm(a_world) > 0 else np.array([0, 0, 1])
                
                print(f"Axis in world coordinates (a_world): {a_world}")
                
                # 修正 K 矩陣方向，確保順時針旋轉為正（與 a 向量一致）
                K = np.array([
                    [0, -a_world[2], a_world[1]],
                    [a_world[2], 0, -a_world[0]],
                    [-a_world[1], a_world[0], 0]
                ])
                # 確保 joint_angle = 0 時 R_joint 為單位矩陣
                if abs(joint_angle) < 1e-10:  # 避免數值誤差
                    R_joint = np.eye(3)
                else:
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
            
            # 修正位置更新，確保關節圍繞自身位置旋轉
            print(f"Node {node.j} initial offset b: {node.b}")
            print(f"Parent Node {parent.j} initial position: {parent.p}")
            b_world = node.R @ node.b  # 使用當前節點的旋轉計算 b_world
            print(f"b_world for Node {node.j} (using node.R): {b_world}")
            
            # 確保位置更新基於正確的父節點位置和當前節點的相對偏移
            node.p = parent.p + b_world
            
            # 打印更新後的位置
            print(f"Node {node.j} after position update - Position: {node.p}, Rotation:")
            print(node.R)
            
            # 每次處理可動關節後增加 angle_idx
            if angle_idx < len(joint_angles) and np.any(node.a != 0):
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

        # 畫關節軸向量 a，從當前可動關節位置開始（僅對具有非零 a 向量的可動關節）
        # 限制僅顯示特定可動關節（節點 2-4、5、7-8）
        movable_joints = [2, 3, 4, 5, 7, 8]  # 更新為新結構（節點 2-4 為髖部，5 為膝蓋，7-8 為腳踝）
        if link.j in movable_joints and np.any(link.a != 0):
            # 使用當前節點的位置作為起點，並計算 a 在世界座標系中的方向
            a_world = link.R @ link.a * 10  # 縮放後的關節軸向量
            ax.quiver(link.p[0], link.p[1], link.p[2],  # 從當前節點位置開始
                      a_world[0], a_world[1], a_world[2],
                      color='g', label='Joint Axis' if link.j == 2 else "")

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
    # 創建一隻腿（包含身體中心、髖部三自由度、單一膝蓋沿 Y+、小腿（無自由度）、腳踝沿 X+ 和 Y+ 的兩個自由度）
    leg_links = create_leg(base_position=np.array([0, 0, 120]), direction="down_z", num_joints=2, joint_length=30, hip_dofs=3, ankle_dofs=2)
    
    # 定義關節角度（弧度），對應每個可動關節（3 個髖部自由度 + 1 個膝蓋 + 2 個腳踝自由度，共 6 個角度）
    # 順序：節點 2（X+，髖部 1）、3（Y+，髖部 2）、4（Z+，髖部 3）、5（Y+，膝蓋）、7（X+，腳踝 1）、8（Y+，腳踝 2）
    joint_angles = [
        np.radians(0),  # 髖部自由度 1（X+，節點 2）
        np.radians(0),   # 髖部自由度 2（Y+，節點 3，設為 0 度）
        np.radians(90),   # 髖部自由度 3（Z+，節點 4，設為 0 度）
        np.radians(0),  # 膝蓋關節（Y+，節點 5）
        np.radians(0),  # 腳踝自由度 1（X+，節點 7）
        np.radians(0)   # 腳踝自由度 2（Y+，節點 8）
    ]
    
    # 應用正運動學
    updated_links = forward_kinematics(leg_links, joint_angles)
    
    # 可視化更新後的結構
    PlotLinkChain(updated_links)
    
    # 打印更新後的節點位置以供檢查
    for link in updated_links:
        print(f"Node {link.j} - Position: {link.p}, Rotation:")
        print(link.R)