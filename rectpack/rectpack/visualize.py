import random
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rectpack import PackingBin, PackingMode, SORT_AREA, newPacker

BoxSpec = Tuple[int, int, int]
BinSpec = Tuple[int, int, int, dict]


def pack_boxes(
    box_specs: Sequence[BoxSpec],
    bins: Sequence[BinSpec],
    *,
    bin_algo: int = PackingBin.BBF,
    sort_algo=SORT_AREA,
    rotation: bool = True,
) -> object:
    """
    Pack a collection of 3D boxes (width, height, depth).
    """
    packer = newPacker(
        mode=PackingMode.Offline,
        bin_algo=bin_algo,
        sort_algo=sort_algo,
        rotation=rotation,
    )

    for spec in bins:
        if len(spec) == 2:
            width, height = spec
            count, kwargs = 1, {}
        elif len(spec) == 3:
            width, height, count = spec
            kwargs = {}
        elif len(spec) == 4:
            width, height, count, kwargs = spec
        else:
            raise ValueError("Bin spec must be (w, h), (w, h, count), or (w, h, count, kwargs)")
        packer.add_bin(width, height, count=count, **kwargs)

    for idx, (width, height, depth) in enumerate(box_specs):
        packer.add_box(width, height, depth, rid=f"box-{idx}")

    packer.pack()
    return packer


def random_boxes(
    count: int,
    *,
    width_range: Tuple[int, int] = (20, 75),
    height_range: Tuple[int, int] = (20, 50),
    depth_range: Tuple[int, int] = (20, 75),
    seed: int = None,
) -> List[BoxSpec]:
    """
    Generate random (width, height, depth) tuples within the provided ranges.
    """
    if seed is not None:
        random.seed(seed)

    boxes: List[BoxSpec] = []
    for _ in range(count):
        while True:
            width = random.randint(*width_range)
            height = random.randint(*height_range)
            depth = random.randint(*depth_range)
            if min(width, height, depth) < 50:
                break
        boxes.append((width, height, depth))
    return boxes


def show_2d(packer: object) -> None:
    """
    Display the packing footprint (X/Y plane) per bin.
    """
    colors = plt.cm.tab20.colors
    for bin_index, abin in enumerate(packer):
        fig, ax = plt.subplots()
        for rect_index, rect in enumerate(abin):
            meta = rect.rid if isinstance(rect.rid, dict) else {"rid": rect.rid}
            color = colors[rect_index % len(colors)]
            ax.add_patch(
                Rectangle(
                    (rect.x, rect.y),
                    rect.width,
                    rect.height,
                    facecolor=color,
                    alpha=0.6,
                    edgecolor="black",
                )
            )
            label = str(meta.get("rid", ""))
            face = meta.get("face")
            if face is not None:
                label = f"{label} ({face})"
            ax.text(
                rect.x + rect.width / 2,
                rect.y + rect.height / 2,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
        ax.set_xlim(0, abin.width)
        ax.set_ylim(0, abin.height)
        ax.set_aspect("equal")
        ax.set_title(f"Bin {bin_index}: {abin.width}x{abin.height}")
        plt.show()


def show_3d(packer: object) -> None:
    """
    Extrude the 2D packing into 3D using the selected orientation metadata.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    max_vertical = 1

    for bin_index, abin in enumerate(packer):
        for rect_index, rect in enumerate(abin):
            meta = rect.rid if isinstance(rect.rid, dict) else {"vertical": 1, "rid": rect.rid}
            vertical = meta.get("vertical", 1)
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
            z = 0
            max_vertical = max(max_vertical, vertical)

            bottom = [(x, y, z), (x + w, y, z), (x + w, y + h, z), (x, y + h, z)]
            top = [(p[0], p[1], p[2] + vertical) for p in bottom]
            verts = [bottom, top]

            ax.add_collection3d(
                Poly3DCollection(
                    verts,
                    alpha=0.4,
                    facecolor=plt.cm.tab20(rect_index % 20),
                    edgecolor="black",
                )
            )

            label = str(meta.get("rid", ""))
            face = meta.get("face")
            if face is not None:
                label = f"{label} ({face})"
            ax.text(
                x + w / 2,
                y + h / 2,
                z + vertical / 2,
                label,
                ha="center",
                va="center",
                fontsize=8,
            )

        ax.set_xlim(0, abin.width)
        ax.set_ylim(0, abin.height)
        ax.set_title(f"Bin {bin_index}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_zlim(0, max_vertical + 5)
    plt.show()


def print_centers_2d(packer: object) -> None:
    """
    Print 2D center coordinates for each placed box.
    """
    for bin_index, abin in enumerate(packer):
        for rect in abin:
            cx = rect.x + rect.width / 2
            cy = rect.y + rect.height / 2
            rid = rect.rid if not isinstance(rect.rid, dict) else rect.rid.get("rid")
            # 2D 중심 좌표 출력
            print(f"bin {bin_index}: center=({cx:.2f}, {cy:.2f}) rid={rid}")


def describe_packing(packer: object) -> Iterable[str]:
    """
    Yield human-readable lines summarising the packing result.
    """
    for bin_index, abin in enumerate(packer):
        yield f"Bin {bin_index}: {abin.width}x{abin.height} -> {len(abin)} boxes"


if __name__ == "__main__":
    BOX_COUNT = 80
    BINS: Sequence[BinSpec] = [
        (336, 255, 4, {}),
        (400, 400, 1, {}),
    ]

    boxes = random_boxes(BOX_COUNT, seed=42)
    packer = pack_boxes(boxes, BINS, rotation=True)

    for line in describe_packing(packer):
        print(line)

    print_centers_2d(packer)
    show_2d(packer)
