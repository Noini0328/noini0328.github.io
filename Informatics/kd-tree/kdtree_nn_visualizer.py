"""
kd-tree による最近傍探索の可視化
================================

2次元点群から kd-tree を構築し、クエリ点に対する最近傍探索の過程を
ステップごとにアニメーション表示します。

表示内容:
  - 左: 空間分割 (分割線) と探索の様子
      * 訪問中のノード (オレンジ)
      * 現在の最良候補 (緑) と、その距離を半径とする探索円
      * 枝刈りされた領域 (グレーの網掛け)
  - 右: kd-tree の木構造 (訪問済み / 枝刈り / 現在ノードを色分け)
  - 下: 各ステップの説明テキスト

使い方:
  python kdtree_nn_visualizer.py                  # ウィンドウでアニメーション表示
  python kdtree_nn_visualizer.py --save out.gif   # GIF として保存
  python kdtree_nn_visualizer.py -n 30 --seed 3 --query 0.3 0.7

必要ライブラリ: numpy, matplotlib (GIF 保存時は pillow)
"""

import argparse
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

# 日本語フォントがあれば使う (無ければ英語表示でも動作します)
from matplotlib import font_manager as _fm

_available = {f.name for f in _fm.fontManager.ttflist}
for _f in ["Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "Hiragino Sans", "Yu Gothic",
           "Meiryo", "MS Gothic", "Noto Sans CJK SC", "Noto Sans CJK TC"]:
    if _f in _available:
        plt.rcParams["font.family"] = _f
        break


# ----------------------------------------------------------------------
# kd-tree
# ----------------------------------------------------------------------
@dataclass
class Node:
    point: np.ndarray
    axis: int
    bounds: tuple            # (xmin, xmax, ymin, ymax) このノードが担当する領域
    left: "Node" = None
    right: "Node" = None
    id: int = 0
    depth: int = 0
    # 木の描画用座標
    tx: float = 0.0
    ty: float = 0.0


def build_kdtree(points, depth=0, bounds=(0, 1, 0, 1), counter=None):
    """中央値で分割する標準的な kd-tree 構築"""
    if counter is None:
        counter = [0]
    if len(points) == 0:
        return None
    axis = depth % 2
    pts = points[points[:, axis].argsort()]
    mid = len(pts) // 2
    node = Node(point=pts[mid], axis=axis, bounds=bounds, id=counter[0], depth=depth)
    counter[0] += 1

    xmin, xmax, ymin, ymax = bounds
    s = pts[mid][axis]
    if axis == 0:
        lb, rb = (xmin, s, ymin, ymax), (s, xmax, ymin, ymax)
    else:
        lb, rb = (xmin, xmax, ymin, s), (xmin, xmax, s, ymax)
    node.left = build_kdtree(pts[:mid], depth + 1, lb, counter)
    node.right = build_kdtree(pts[mid + 1:], depth + 1, rb, counter)
    return node


def iter_nodes(node):
    if node is None:
        return
    yield node
    yield from iter_nodes(node.left)
    yield from iter_nodes(node.right)


def layout_tree(root):
    """木を描画するための座標 (in-order で x、深さで y) を割り当てる"""
    order = [0]

    def rec(n):
        if n is None:
            return
        rec(n.left)
        n.tx = order[0]
        order[0] += 1
        n.ty = -n.depth
        rec(n.right)

    rec(root)


def rect_dist(q, bounds):
    """クエリ点から矩形領域までの最短距離 (枝刈り判定の補助表示用)"""
    xmin, xmax, ymin, ymax = bounds
    dx = max(xmin - q[0], 0, q[0] - xmax)
    dy = max(ymin - q[1], 0, q[1] - ymax)
    return np.hypot(dx, dy)


# ----------------------------------------------------------------------
# 最近傍探索 (各ステップを記録)
# ----------------------------------------------------------------------
@dataclass
class Step:
    current: Node
    best: Node
    best_dist: float
    visited: set
    pruned: list          # 枝刈りされたノード (部分木の根) のリスト
    message: str


def nn_search_with_trace(root, q):
    steps = []
    state = {"best": None, "dist": np.inf}
    visited, pruned = set(), []

    def snap(cur, msg):
        steps.append(Step(cur, state["best"], state["dist"],
                          set(visited), list(pruned), msg))

    def search(node):
        if node is None:
            return
        visited.add(node.id)
        d = np.linalg.norm(node.point - q)
        ax = "x" if node.axis == 0 else "y"
        snap(node, f"ノード {node.id} ({node.point[0]:.2f}, {node.point[1]:.2f}) を訪問"
                   f"  距離 = {d:.3f}  [分割軸: {ax}]")
        if d < state["dist"]:
            state["best"], state["dist"] = node, d
            snap(node, f"最良候補を更新 → ノード {node.id}  (最短距離 {d:.3f})")

        diff = q[node.axis] - node.point[node.axis]
        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
        side = "左/下" if diff < 0 else "右/上"
        if near is not None:
            snap(node, f"クエリは分割線の{side}側 → 近い側の部分木へ降りる")
        search(near)

        if far is None:
            return
        if abs(diff) < state["dist"]:
            snap(node, f"戻り: 分割線までの距離 {abs(diff):.3f} < 最短距離 {state['dist']:.3f}"
                       f" → 反対側も探索")
            search(far)
        else:
            pruned.append(far)
            snap(node, f"戻り: 分割線までの距離 {abs(diff):.3f} ≥ 最短距離 {state['dist']:.3f}"
                       f" → 反対側を枝刈り")

    search(root)
    snap(None, f"探索終了: 最近傍はノード {state['best'].id} "
               f"({state['best'].point[0]:.2f}, {state['best'].point[1]:.2f})、"
               f"距離 {state['dist']:.3f}  / 訪問 {len(visited)} ノード")
    return steps


# ----------------------------------------------------------------------
# 可視化
# ----------------------------------------------------------------------
C_POINT = "#4a6fa5"
C_SPLIT = "#9aa5b1"
C_CUR = "#f28e2b"
C_BEST = "#2ca02c"
C_QUERY = "#d62728"
C_PRUNE = "#bbbbbb"


def draw_static(ax_space, ax_tree, root, q, all_nodes):
    # --- 空間: 分割線 ---
    for n in all_nodes:
        xmin, xmax, ymin, ymax = n.bounds
        if n.axis == 0:
            ax_space.plot([n.point[0]] * 2, [ymin, ymax], color=C_SPLIT, lw=1, zorder=1)
        else:
            ax_space.plot([xmin, xmax], [n.point[1]] * 2, color=C_SPLIT, lw=1, zorder=1)
    pts = np.array([n.point for n in all_nodes])
    ax_space.scatter(pts[:, 0], pts[:, 1], s=28, color=C_POINT, zorder=3)
    for n in all_nodes:
        ax_space.annotate(str(n.id), n.point, xytext=(3, 3), textcoords="offset points",
                          fontsize=7, color="#333")
    ax_space.scatter(*q, marker="*", s=220, color=C_QUERY, zorder=6, label="クエリ")
    ax_space.set_xlim(0, 1)
    ax_space.set_ylim(0, 1)
    ax_space.set_aspect("equal")
    ax_space.set_title("空間分割と探索")
    ax_space.set_xticks([])
    ax_space.set_yticks([])

    # --- 木構造: エッジ ---
    for n in all_nodes:
        for c in (n.left, n.right):
            if c is not None:
                ax_tree.plot([n.tx, c.tx], [n.ty, c.ty], color=C_SPLIT, lw=1, zorder=1)
    ax_tree.set_title("kd-tree")
    ax_tree.axis("off")


def make_animation(root, q, steps, interval=900):
    all_nodes = list(iter_nodes(root))
    layout_tree(root)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[12, 1], width_ratios=[1, 1.25])
    ax_space = fig.add_subplot(gs[0, 0])
    ax_tree = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")

    draw_static(ax_space, ax_tree, root, q, all_nodes)

    # 木のノード (色を毎フレーム更新)
    tree_scatter = ax_tree.scatter([n.tx for n in all_nodes], [n.ty for n in all_nodes],
                                   s=260, color="white", edgecolor="#555", zorder=3)
    for n in all_nodes:
        ax_tree.text(n.tx, n.ty, str(n.id), ha="center", va="center", fontsize=7, zorder=4)
    id_to_idx = {n.id: i for i, n in enumerate(all_nodes)}

    def subtree_ids(n):
        return {m.id for m in iter_nodes(n)}

    dynamic = []
    text = ax_text.text(0.5, 0.5, "", ha="center", va="center", fontsize=12)

    def update(i):
        for art in dynamic:
            art.remove()
        dynamic.clear()
        st = steps[i]

        # 枝刈りされた領域
        pruned_ids = set()
        for p in st.pruned:
            xmin, xmax, ymin, ymax = p.bounds
            r = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor=C_PRUNE,
                          alpha=0.35, hatch="//", edgecolor="#999", lw=0, zorder=0)
            ax_space.add_patch(r)
            dynamic.append(r)
            pruned_ids |= subtree_ids(p)

        # 現在ノードの担当領域と分割線
        if st.current is not None:
            xmin, xmax, ymin, ymax = st.current.bounds
            r = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                          edgecolor=C_CUR, lw=2, ls="--", zorder=2)
            ax_space.add_patch(r)
            dynamic.append(r)
            dynamic.extend(ax_space.plot(*st.current.point, "o", ms=12, mfc="none",
                                     mec=C_CUR, mew=2.5, zorder=5))
            dynamic.extend(ax_space.plot([q[0], st.current.point[0]], [q[1], st.current.point[1]],
                                     color=C_CUR, lw=1, ls=":", zorder=4))

        # 最良候補と探索円
        if st.best is not None:
            c = Circle(q, st.best_dist, fill=False, edgecolor=C_BEST, lw=1.8, zorder=4)
            ax_space.add_patch(c)
            dynamic.append(c)
            dynamic.extend(ax_space.plot(*st.best.point, "o", ms=9, color=C_BEST, zorder=5))
            dynamic.extend(ax_space.plot([q[0], st.best.point[0]], [q[1], st.best.point[1]],
                                     color=C_BEST, lw=1.5, zorder=4))

        # 木のノード色
        colors = []
        for n in all_nodes:
            if st.current is not None and n.id == st.current.id:
                colors.append(C_CUR)
            elif st.best is not None and n.id == st.best.id:
                colors.append(C_BEST)
            elif n.id in pruned_ids:
                colors.append(C_PRUNE)
            elif n.id in st.visited:
                colors.append("#a6c8ff")
            else:
                colors.append("white")
        tree_scatter.set_facecolor(colors)

        text.set_text(f"[{i + 1}/{len(steps)}]  {st.message}")
        return dynamic + [tree_scatter, text]

    # 凡例
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], marker="*", color=C_QUERY, ls="", ms=12, label="クエリ"),
        Line2D([], [], marker="o", mec=C_CUR, mfc="none", ls="", ms=10, mew=2, label="現在ノード"),
        Line2D([], [], marker="o", color=C_BEST, ls="", ms=8, label="最良候補 / 探索円"),
        Patch(facecolor="#a6c8ff", edgecolor="#555", label="訪問済み"),
        Patch(facecolor=C_PRUNE, edgecolor="#999", hatch="//", label="枝刈り"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    anim = FuncAnimation(fig, update, frames=len(steps), interval=interval,
                         blit=False, repeat=True)
    return fig, anim


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="kd-tree 最近傍探索の可視化")
    ap.add_argument("-n", "--num-points", type=int, default=20, help="点の数")
    ap.add_argument("--seed", type=int, default=0, help="乱数シード")
    ap.add_argument("--query", type=float, nargs=2, default=None, help="クエリ点 (x y)")
    ap.add_argument("--interval", type=int, default=900, help="フレーム間隔 [ms]")
    ap.add_argument("--save", type=str, default=None, help="GIF の保存先 (例: out.gif)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    points = rng.random((args.num_points, 2))
    q = np.array(args.query) if args.query else rng.random(2)

    root = build_kdtree(points)
    steps = nn_search_with_trace(root, q)

    # 総当たりで結果を検証
    brute = points[np.argmin(np.linalg.norm(points - q, axis=1))]
    assert np.allclose(brute, steps[-1].best.point), "kd-tree の結果が総当たりと一致しません"
    print(steps[-1].message)

    fig, anim = make_animation(root, q, steps, args.interval)
    if args.save:
        anim.save(args.save, writer=PillowWriter(fps=max(1, 1000 // args.interval)))
        print(f"保存しました: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
