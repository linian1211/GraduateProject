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
        p=np.array([0, 0, 95]),  # 基座位置
        R=np.eye(3),  # 初始旋轉為單位矩陣
        m=10.0,
        a=np.zeros(3),  # 基座無關節軸
        b=np.array([0, 0, -15]),  # 基座末端偏移（向下 15 單位）
        child=1  # 子節點為髖部到髖部馬達的 link（節點 1）
    )
    links.append(base)
    
    # 2. 創建髖部到髖部馬達間的 link（節點 1），長度為 0
    hip_to_thigh = Link(
        j=1,
        mother=0,  # 母節點為基座
        child=2,  # 子節點為第一個髖部自由度（節點 2）
        p=base.p + (base.R @ base.b),  # 靠近身體端的位置（基於基座末端）
        R=np.eye(3),  # 初始旋轉為單位矩陣
        m=0.0,
        a=np.zeros(3),  # 無自由度
        b=np.array([0, -15, 0]),  # 末端偏移到髖部自由度位置（長度為 0）
        q=0
    )
    links.append(hip_to_thigh)
    
    # 3. 創建髖部三個自由度關節（節點 2、3、4），長度為 0
    # 第一個髖部自由度（沿 X+，節點 2）
    hip1 = Link(
        j=2,
        mother=1,  # 母節點為髖部到馬達的 link
        child=3,  # 子節點為第二個髖部自由度
        p=hip_to_thigh.p + (hip_to_thigh.R @ hip_to_thigh.b),  # 靠近身體端的位置
        R=np.eye(3),  # 初始旋轉為單位矩陣
        m=0.0,
        a=np.array([0, 0, 1]),  # 沿 X+ 軸旋轉
        b=np.zeros(3),  # 長度為 0（無偏移）
        q=0
    )
    links.append(hip1)
    
    # 第二個髖部自由度（沿 Y+，節點 3）
    hip2 = Link(
        j=3,
        mother=2,  # 母節點為第一個髖部自由度
        child=4,  # 子節點為第三個髖部自由度
        p=hip1.p + (hip1.R @ hip1.b),  # 靠近身體端的位置
        R=np.eye(3),  # 初始旋轉為單位矩陣
        m=0.0,
        a=np.array([1, 0, 0]),  # 沿 Y+ 軸旋轉
        b=np.zeros(3),  # 長度為 0
        q=0
    )
    links.append(hip2)
    
    # 第三個髖部自由度（沿 Z+，節點 4）
    hip3 = Link(
        j=4,
        mother=3,  # 母節點為第二個髖部自由度
        child=5,  # 子節點為膝蓋
        p=hip2.p + (hip2.R @ hip2.b),  # 靠近身體端的位置
        R=np.eye(3),  # 初始旋轉為單位矩陣
        m=0.0,
        a=np.array([0, 1, 0]),  # 沿 Z+ 軸旋轉
        b=np.array([0, 0, -40]), 
        q=0
    )
    links.append(hip3)
    
    # 4. 創建膝蓋一個自由度（節點 5）
    knee = Link(
        j=5,
        mother=4,  # 母節點為第三個髖部自由度
        child=6,  # 子節點為第一個腳踝自由度
        p=hip3.p + (hip3.R @ hip3.b),  # 靠近身體端的位置
        R=np.eye(3),  # 初始旋轉為單位矩陣
        m=0.0,
        a=np.array([0, 1, 0]),  # 沿 Y+ 軸旋轉
        b=np.array([0, 0, -25]), 
        q=0
    )
    links.append(knee)
    
    # 5. 創建腳踝兩個自由度（節點 6、7），長度為 0
    # 第一個腳踝自由度（沿 X+，節點 6）
    ankle1 = Link(
        j=6,
        mother=5,  # 母節點為膝蓋
        child=7,  # 子節點為第二個腳踝自由度
        p=knee.p + (knee.R @ knee.b),  # 靠近身體端的位置
        R=np.eye(3),  # 初始旋轉為單位矩陣
        m=0.0,
        a=np.array([0, 1, 0]),
        b=np.array([0, 0, 0]),
        q=0
    )
    links.append(ankle1)
    
    # 第二個腳踝自由度（沿 Y+，節點 7）
    ankle2 = Link(
        j=7,
        mother=6,  # 母節點為第一個腳踝自由度
        child=None,
        p=ankle1.p + (ankle1.R @ ankle1.b),  # 靠近身體端的位置
        R=np.eye(3),  # 初始旋轉為單位矩陣
        m=0.0,
        a=np.array([1, 0, 0]),  # 沿 Y+ 軸旋轉
        b=np.array([0, 0, -15]),
        q=0
    )
    links.append(ankle2)
    
    print("Created Model:")
    for link in links:
        print(f"Node {link.j} - Mother: {link.mother}, Child: {link.child}, Position (p): {link.p}, "
              f"Rotation (R):\n{link.R}\nAxis (a): {link.a}, Offset (b): {link.b}")
    
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
                # 若a為零(不旋轉的軸)，則使用預設軸向量
                if np.linalg.norm(a_world) == 0:
                    print(f"Warning: Zero axis for Node {node.j}, forcing default axis [1, 0, 0]")
                    a_world = np.array([1, 0, 0])  # 臨時使用 X 軸作為預設
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
            node.p = parent.p + (parent.R @ parent.b)
            
            # 每次處理可動關節後增加 angle_idx
            if angle_idx < len(joint_angles) and np.any(node.a != 0):
                angle_idx += 1

        # 遞歸處理子節點
        if node.child is not None:
            child = next(l for l in updated_links if l.j == node.child)
            update_node(child, node)

    # 調用 update_node 從根節點開始
    update_node(root)
    
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
    
    # 定義關節角度（弧度），對應單一可動關節（馬達）
    joint_angles = [
        np.radians(0),  # 髖部自由度 1（Z+，節點 2）
        np.radians(0),   # 髖部自由度 2（X+，節點 3，設為 0 度）
        np.radians(90),   # 髖部自由度 3（Y+，節點 4，設為 0 度）
        np.radians(0),  # 膝蓋關節（Y+，節點 5）
        np.radians(0),  # 腳踝自由度 1（Y+，節點 6）
        np.radians(0)   # 腳踝自由度 2（X+，節點 7）
    ]
    # 應用正運動學
    updated_links = forward_kinematics(simple_links, joint_angles)
    
    # 可視化更新後的結構
    PlotLinkChain(updated_links)
    
    # 打印更新後的節點位置以供檢查
    for link in updated_links:
        print(f"Node {link.j} - Position (p): {link.p}, Rotation (R):")
        print(link.R)