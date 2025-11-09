import random
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rectpack import PackingBin, PackingMode, SORT_AREA, newPacker

BoxSpec = Tuple[int, int, int]
BinSpec = Tuple[int, int, int, dict]
DEFAULT_MARGIN: Tuple[int, int] = (25, 25)
DEFAULT_RX = 0.0
DEFAULT_RY = 1.5707
DEFAULT_RZ_X = 0.0
DEFAULT_RZ_Y = 1.5707


def pack_boxes(
    box_specs: Sequence[BoxSpec],
    bins: Sequence[BinSpec],
    *,
    bin_algo: int = PackingBin.BBF,
    sort_algo=SORT_AREA,
    rotation: bool = True,
    margin: Tuple[int, int] = DEFAULT_MARGIN,
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

    mx, my = margin

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

        inner_width = width - 2 * mx
        inner_height = height - 2 * my
        if inner_width <= 0 or inner_height <= 0:
            raise ValueError(
                f"Margin {margin} is too large for bin {width}x{height} (needs positive interior)."
            )
        packer.add_bin(inner_width, inner_height, count=count, **kwargs)

    for idx, (width, height, depth) in enumerate(box_specs):
        packer.add_box(width, height, depth, rid=f"box-{idx}")

    packer.pack()
    _apply_margin(packer, margin)
    return packer


def _apply_margin(packer: object, margin: Tuple[int, int]) -> None:
    """
    Shift every rectangle in-place so bins get uniform margins on each edge.
    """
    mx, my = margin
    if mx == 0 and my == 0:
        return

    for abin in packer:
        for rect in abin:
            rect.x += mx
            rect.y += my
        abin.width += mx * 2
        abin.height += my * 2


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


def show_2d(packer: object, *, offset: Tuple[int, int] = (0, 0)) -> None:
    """
    Display the packing footprint (X/Y plane) per bin.
    """
    ox, oy = offset
    colors = plt.cm.tab20.colors
    for bin_index, abin in enumerate(packer):
        fig, ax = plt.subplots()
        for rect_index, rect in enumerate(abin):
            meta = rect.rid if isinstance(rect.rid, dict) else {"rid": rect.rid}
            color = colors[rect_index % len(colors)]
            x = rect.x + ox
            y = rect.y + oy
            ax.add_patch(
                Rectangle(
                    (x, y),
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
                x + rect.width / 2,
                y + rect.height / 2,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
        ax.set_xlim(ox, ox + abin.width)
        ax.set_ylim(oy, oy + abin.height)
        ax.set_aspect("equal")
        ax.set_title(f"Bin {bin_index}: {abin.width}x{abin.height}")
        plt.show()


def show_3d(packer: object, *, offset: Tuple[int, int] = (0, 0)) -> None:
    """
    Extrude the 2D packing into 3D using the selected orientation metadata.
    """
    ox, oy = offset
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    max_vertical = 1

    for bin_index, abin in enumerate(packer):
        for rect_index, rect in enumerate(abin):
            meta = rect.rid if isinstance(rect.rid, dict) else {"vertical": 1, "rid": rect.rid}
            vertical = meta.get("vertical", 1)
            x, y, w, h = rect.x + ox, rect.y + oy, rect.width, rect.height
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

        ax.set_xlim(ox, ox + abin.width)
        ax.set_ylim(oy, oy + abin.height)
        ax.set_title(f"Bin {bin_index}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_zlim(0, max_vertical + 5)
    plt.show()


def print_centers_2d(packer: object, *, offset: Tuple[int, int] = (0, 0)) -> None:
    """
    Print 2D center coordinates for each placed box.
    """
    ox, oy = offset
    for bin_index, abin in enumerate(packer):
        for rect in abin:
            cx = rect.x + rect.width / 2 + ox
            cy = rect.y + rect.height / 2 + oy
            rid = rect.rid if not isinstance(rect.rid, dict) else rect.rid.get("rid")
            # 2D 중심 좌표 출력
            print(f"bin {bin_index}: center=({cx:.2f}, {cy:.2f}) rid={rid}")


def describe_packing(packer: object) -> Iterable[str]:
    """
    Yield human-readable lines summarising the packing result.
    """
    for bin_index, abin in enumerate(packer):
        yield f"Bin {bin_index}: {abin.width}x{abin.height} -> {len(abin)} boxes"


def iter_pick_poses(
    packer: object,
    *,
    base_z: float = 0.0,
    rx: float = DEFAULT_RX,
    ry: float = DEFAULT_RY,
    rz_x: float = DEFAULT_RZ_X,
    rz_y: float = DEFAULT_RZ_Y,
) -> Iterator[Dict[str, float]]:
    """
    Yield pose dictionaries (x, y, z, rx, ry, rz) for every placed rectangle.
    """
    for bin_index, abin in enumerate(packer):
        for rect_index, rect in enumerate(abin):
            meta = rect.rid if isinstance(rect.rid, dict) else {"rid": rect.rid}
            width = float(rect.width)
            height = float(rect.height)
            vertical = float(meta.get("vertical", 0.0))
            cx = rect.x + width / 2.0
            cy = rect.y + height / 2.0
            cz = base_z + vertical / 2.0
            long_on_x = width >= height
            rz = rz_x if long_on_x else rz_y
            yield {
                "bin": bin_index,
                "rect_index": rect_index,
                "rid": meta.get("rid"),
                "x": cx,
                "y": cy,
                "z": cz,
                "rx": rx,
                "ry": ry,
                "rz": rz,
                "width": width,
                "height": height,
                "vertical": vertical,
            }


def print_pick_poses(
    packer: object,
    *,
    base_z: float = 0.0,
    rx: float = DEFAULT_RX,
    ry: float = DEFAULT_RY,
    rz_x: float = DEFAULT_RZ_X,
    rz_y: float = DEFAULT_RZ_Y,
) -> None:
    """
    Print pick/place poses that a manipulator can consume directly.
    """
    for pose in iter_pick_poses(
        packer, base_z=base_z, rx=rx, ry=ry, rz_x=rz_x, rz_y=rz_y
    ):
        print(
            "bin={bin} rect={rect_index} rid={rid} "
            "pose=({x:.2f}, {y:.2f}, {z:.2f}, {rx:.4f}, {ry:.4f}, {rz:.4f}) "
            "size=({width:.2f}, {height:.2f}, {vertical:.2f})".format(**pose)
        )


def build_sequence_request(
    packer: object,
    *,
    robot_id: int,
    order_id: int,
    base_z: float = 0.0,
    rx: float = DEFAULT_RX,
    ry: float = DEFAULT_RY,
    rz_x: float = DEFAULT_RZ_X,
    rz_y: float = DEFAULT_RZ_Y,
    seq_start: int = 0,
) -> Dict[str, Any]:
    """
    Construct a request payload matching the described ROS2 service schema.

    Sequences are sorted so the rectangle with the lowest Z center is first.
    """
    poses = list(
        iter_pick_poses(
            packer,
            base_z=base_z,
            rx=rx,
            ry=ry,
            rz_x=rz_x,
            rz_y=rz_y,
        )
    )
    poses.sort(key=lambda pose: pose["z"])

    sequences: List[Dict[str, Any]] = []
    for seq_idx, pose in enumerate(poses, start=seq_start):
        rid = pose["rid"]
        rect_id = rid if isinstance(rid, int) else pose["rect_index"]
        sequences.append(
            {
                "seq": seq_idx,
                "id": rect_id,
                "x": pose["x"],
                "y": pose["y"],
                "z": pose["z"],
                "rx": pose["rx"],
                "ry": pose["ry"],
                "rz": pose["rz"],
            }
        )

    return {"robot_id": int(robot_id), "order_id": int(order_id), "sequences": sequences}


def build_sequence_response(success: bool, message: str = "") -> Dict[str, Any]:
    """
    Helper to mirror the service response portion (bool success, string message).
    """
    return {"success": bool(success), "message": str(message)}


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
    print_pick_poses(packer)
    show_2d(packer)
