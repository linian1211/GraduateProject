from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class Link:
    j: int  # 自身ID
    child: int = None  # 子節點ID
    mother: int = None  # 母節點ID

    p: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 靠近身體那端節點的世界座標位置
    R: np.ndarray = field(default_factory=lambda: np.eye(3))  # 靠近身體那端節點的世界姿態

    v: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 線速度
    w: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 角速度

    q: float = 0.0  # 繞著a相對於前一節link旋轉的角度
    dq: float = 0.0  # 關節速度
    ddq: float = 0.0  # 關節加速度
    
    a: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 本link在前一節座標中相對於前一節link的轉軸向量
    b: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 本link末端在旋轉過後的相對座標(本link座標)中的向量

    vertex: list = field(default_factory=list)  # 形狀（頂點信息，鏈接本地）
    face: list = field(default_factory=list)  # 形狀（點連接）

    m: float = 0.0  # 質量
    c: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 質心（鏈接本地）
    I: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # 慣性矩（鏈接本地）

def create_simple_model():
    links = []
    
    # 1. 創建基座（節點 0）
    base = Link(
        j=0,
        p=np.array([0, 0, 0]), 
        R=np.eye(3), 
        m=10.0,
        a=np.zeros(3), 
        b=np.array([0, 0, -15]),  # 基座末端偏移（向下 15 單位）
        child=1  #
    )
    links.append(base)
    
    # 2. （節點 1），相對於基座偏移 [0, -15, 0]，沿 X+ 軸旋轉
    motor = Link(
        j=1,
        mother=0,  # 母節點為基座
        child=None,
        p=links[0].p + links[0].R @ links[0].b,  # 馬達靠近身體端的位置（初始與基座相同，後由 b 決定末端）
        R=np.eye(3),  # 初始旋轉為單位矩陣
        m=0.0,
        a=np.array([1, 0, 0]),  # 沿 X+ 軸旋轉
        b=np.array([0, -15, 0]),  # 馬達末端相對於靠近身體端的偏移（旋轉後）
        q=0
    )
    links.append(motor)
    
    return links

def forward_kinematics(links: list, joint_angles: list):
    # 創建深拷貝
    updated_links = [Link(**link.__dict__) for link in links]
    
    # 查找根節點（無母節點的節點）
    root_candidates = [link for link in updated_links if link.mother is None]
    if not root_candidates:
        print("Error: No root node found!")
        return updated_links
    root = root_candidates[0]

    # 初始化根節點
    root.p = root.p.copy()
    root.R = np.eye(3)
    
    angle_idx = 0
    
    def update_node(node: Link, parent: Link = None):
        nonlocal angle_idx
        
        if node.mother is not None:
            # 查找母節點
            parent = next(l for l in updated_links if l.j == node.mother)

            # 嘗試應用角度（僅對具有非零 a 的節點）
            if angle_idx < len(joint_angles) and np.any(node.a != 0):
                joint_angle = joint_angles[angle_idx]
                node.q = joint_angle
                
                # 計算 a 在世界座標系中的方向
                a_world = parent.R @ node.a
                # 讓a_world長度為1
                a_world = a_world / np.linalg.norm(a_world) if np.linalg.norm(a_world) > 0 else np.array([1, 0, 0])
                
                # 計算 K 矩陣（用於羅德里格斯旋轉公式）
                K = np.array([
                    [0, -a_world[2], a_world[1]],
                    [a_world[2], 0, -a_world[0]],
                    [-a_world[1], a_world[0], 0]
                ])
                # 計算並打印 R_joint（旋轉矩陣）
                R_joint = np.eye(3) + np.sin(joint_angle) * K + (1 - np.cos(joint_angle)) * (K @ K)                
                # 更新節點的姿態（靠近身體端的世界姿態）
                node.R = R_joint @ parent.R
            
            else:
                # 如果無角度或 a 為零，繼承父節點的姿態
                node.R = parent.R.copy()
            
            # 更新 p 為母節點 p 加上 b（已應用旋轉）
            node.p = parent.p + (parent.R @ parent.b)  # 使用當前節點的 R 應用到 b
            
            # 每次處理可動關節後增加 angle_idx
            if angle_idx < len(joint_angles) and np.any(node.a != 0):
                angle_idx += 1
        
        # 遞歸處理子節點
        if node.child is not None:
            child = next(l for l in updated_links if l.j == node.child)
            update_node(child, node)

    return updated_links

def PlotLinkChain(links: list):
    """ 在 3D 空間中顯示連桿鏈 """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for link in links:
        # 畫節點位置（靠近身體端的位置 p）
        ax.scatter(link.p[0], link.p[1], link.p[2], 
                  c='r', marker='o', label=f"Node" if link.j == 0 else "")

        # 畫出連桿
        # 計算連桿末端位置（當前節點的 p + R @ b）
        end_point = link.p + (link.R @ link.b)

        # 畫連桿線段
        ax.plot([link.p[0], end_point[0]], 
                [link.p[1], end_point[1]], 
                [link.p[2], end_point[2]], 
                'b-', label='Link' if link.j == 1 else "")

        # 畫關節軸向量 a，從當前節點位置開始（僅對具有非零 a 向量的可動關節）
        if np.any(link.a != 0):
            # 使用當前節點的位置作為起點，並計算 a 在世界座標系中的方向
            a_world = link.R @ link.a * 10  # 縮放後的關節軸向量
            ax.quiver(link.p[0], link.p[1], link.p[2],  # 從當前節點位置開始
                      a_world[0], a_world[1], a_world[2],
                      color='g', label='Joint Axis' if link.j == 1 else "")

    # 設定座標軸標籤
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 設定視圖範圍（擴展以顯示可能的旋轉）
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([-50, 50])

    # 顯示標籤
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.show()

# 測試範例
if __name__ == "__main__":
    # 創建簡單模型（基座 + 單一馬達）
    simple_links = create_simple_model()
    
    # 可視化初始結構
    PlotLinkChain(simple_links)
    
    # 定義關節角度（弧度），對應單一可動關節（馬達）
    joint_angles = [np.radians(90)]  # 沿 X+ 軸旋轉 90 度
    
    # 應用正運動學
    updated_links = forward_kinematics(simple_links, joint_angles)
    
    # 可視化更新後的結構
    PlotLinkChain(updated_links)
    
    # 打印更新後的節點位置以供檢查
    for link in updated_links:
        print(f"Node {link.j} - Position (p): {link.p}, Rotation (R):")
        print(link.R)

        修改