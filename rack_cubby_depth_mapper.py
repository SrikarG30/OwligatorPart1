import depthai as dai
import cv2
import numpy as np
import json
import argparse
from string import ascii_uppercase

# =========================
# UI / EDIT STATE
# =========================
clicked = []               # 4 corners TL,TR,BR,BL
active_corner = -1         # 0..3 if dragging a corner
dragging_whole = False
prev_mouse = None
HIT_RADIUS = 18            # px grab radius for corner handles

# =========================
# Mouse callback
# =========================
def clamp_pt(pt, w, h):
    x, y = pt
    return (int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)))

def closest_corner_index(x, y, corners, radius=18):
    if len(corners) != 4:
        return -1
    d2 = [ (corners[i][0] - x) ** 2 + (corners[i][1] - y) ** 2 for i in range(4) ]
    i = int(np.argmin(d2))
    return i if d2[i] <= radius * radius else -1

def move_all_corners(corners, dx, dy, w, h):
    return [clamp_pt((cx + dx, cy + dy), w, h) for (cx, cy) in corners]

def draw_corner_handles(img, corners):
    if len(corners) != 4:
        return
    labels = ["TL", "TR", "BR", "BL"]
    for i, (x, y) in enumerate(corners):
        cv2.circle(img, (x, y), 8, (0, 255, 255), -1)
        cv2.putText(img, labels[i], (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def mouse_cb(event, x, y, flags, param):
    """
    Combo Option 1 + Option 2:
    - First 4 left-clicks place corners TL->TR->BR->BL
    - After that:
      * drag near a corner to adjust it
      * Shift+drag anywhere to move the whole quad
      * right-click drag also moves the whole quad (if you have a mouse)
    """
    global clicked, active_corner, dragging_whole, prev_mouse

    img_w, img_h = param if param is not None else (99999, 99999)

    # Stage 1: collect 4 corners by clicking
    if len(clicked) < 4:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
            print(f"[click] {len(clicked)}/4 -> ({x},{y})")
            if len(clicked) == 4:
                print("[edit] Drag corners to refine. Shift+drag to move whole grid.")
        return

    # Stage 2: edit mode
    if event == cv2.EVENT_LBUTTONDOWN:
        prev_mouse = (x, y)

        # Shift + left drag => move whole quad
        if (flags & cv2.EVENT_FLAG_SHIFTKEY):
            dragging_whole = True
            active_corner = -1
            return

        # Otherwise try to grab a corner
        idx = closest_corner_index(x, y, clicked, radius=HIT_RADIUS)
        if idx != -1:
            active_corner = idx
            dragging_whole = False
        else:
            # Clicked not near corner -> move whole quad (nice UX)
            dragging_whole = True
            active_corner = -1

    elif event == cv2.EVENT_RBUTTONDOWN:
        prev_mouse = (x, y)
        dragging_whole = True
        active_corner = -1

    elif event == cv2.EVENT_MOUSEMOVE:
        if prev_mouse is None:
            return

        if (active_corner != -1) or dragging_whole:
            dx = x - prev_mouse[0]
            dy = y - prev_mouse[1]

            if active_corner != -1:
                new_pt = clamp_pt(
                    (clicked[active_corner][0] + dx, clicked[active_corner][1] + dy),
                    img_w, img_h
                )
                clicked[active_corner] = new_pt
            else:
                clicked[:] = move_all_corners(clicked, dx, dy, img_w, img_h)

            prev_mouse = (x, y)

    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        active_corner = -1
        dragging_whole = False
        prev_mouse = None

# =========================
# DepthAI pipeline
# =========================
def create_pipeline():
    pipeline = dai.Pipeline()

    # ---- Color camera ----
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setPreviewSize(1280, 720)

    rgb_out = pipeline.create(dai.node.XLinkOut)
    rgb_out.setStreamName("rgb")
    cam_rgb.preview.link(rgb_out.input)

    # ---- Stereo depth ----
    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_l.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    depth_out = pipeline.create(dai.node.XLinkOut)
    depth_out.setStreamName("depth")
    stereo.depth.link(depth_out.input)

    return pipeline

# =========================
# Helpers
# =========================
def colorize_depth(depth_u16, max_depth_mm=4000):
    d = np.clip(depth_u16, 0, max_depth_mm)
    d8 = (d.astype(np.float32) / max_depth_mm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(255 - d8, cv2.COLORMAP_JET)

def median_depth_patch(depth_u16, x, y, k=4):
    """Median depth in (2k+1)x(2k+1) window, ignoring zeros."""
    h, w = depth_u16.shape[:2]
    x0, x1 = max(0, x - k), min(w - 1, x + k)
    y0, y1 = max(0, y - k), min(h - 1, y + k)
    patch = depth_u16[y0:y1 + 1, x0:x1 + 1].reshape(-1)
    patch = patch[patch > 0]
    if patch.size == 0:
        return None
    return int(np.median(patch))

def bilinear_point(TL, TR, BR, BL, u, v):
    """Interpolate inside quad. u,v in [0..1]."""
    TL = np.array(TL, np.float32)
    TR = np.array(TR, np.float32)
    BR = np.array(BR, np.float32)
    BL = np.array(BL, np.float32)
    top = TL * (1 - u) + TR * u
    bot = BL * (1 - u) + BR * u
    pt = top * (1 - v) + bot * v
    return int(pt[0]), int(pt[1])

def cubby_id(r, c):
    return f"{ascii_uppercase[r]}{c + 1}"

def compute_physical_uv_lines(rows, cols, cubby_w_in, cubby_h_in, gap_x_in, gap_y_in):
    """
    Physical math grid:
    total rack width/height includes outer gaps too.
    W = cols*cubby_w + (cols+1)*gap_x
    H = rows*cubby_h + (rows+1)*gap_y

    Grid lines are at:
    x_c = gap_x + c*(cubby_w + gap_x)   for c=0..cols
    y_r = gap_y + r*(cubby_h + gap_y)   for r=0..rows

    normalized u=x/W, v=y/H
    """
    W = cols * cubby_w_in + (cols + 1) * gap_x_in
    H = rows * cubby_h_in + (rows + 1) * gap_y_in

    u_lines = []
    for c in range(cols + 1):
        x_c = gap_x_in + c * (cubby_w_in + gap_x_in)
        u_lines.append(x_c / W)

    v_lines = []
    for r in range(rows + 1):
        y_r = gap_y_in + r * (cubby_h_in + gap_y_in)
        v_lines.append(y_r / H)

    return W, H, u_lines, v_lines

def cubby_center_uv(r, c, cols, rows, cubby_w_in, cubby_h_in, gap_x_in, gap_y_in, W, H):
    """
    Physical center of cubby (r,c) in inches -> normalized u,v.
    x_center = gap_x + c*(cubby_w+gap_x) + cubby_w/2
    y_center = gap_y + r*(cubby_h+gap_y) + cubby_h/2
    """
    x_center = gap_x_in + c * (cubby_w_in + gap_x_in) + cubby_w_in / 2.0
    y_center = gap_y_in + r * (cubby_h_in + gap_y_in) + cubby_h_in / 2.0
    u = x_center / W
    v = y_center / H
    return u, v

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)

    # occupancy depth parameters (optional)
    parser.add_argument("--depth_thresh_mm", type=int, default=90,
                        help="How much closer than rack plane counts as occupied (mm)")
    parser.add_argument("--max_depth_mm", type=int, default=4000)

    # physical rack geometry (required for physical-math grid)
    parser.add_argument("--cubby_w_in", type=float, required=True,
                        help="Clear cubby opening width in inches (e.g., 4)")
    parser.add_argument("--cubby_h_in", type=float, required=True,
                        help="Clear cubby opening height in inches (e.g., 3)")
    parser.add_argument("--gap_x_in", type=float, required=True,
                        help="Post thickness between columns in inches (e.g., 1.0)")
    parser.add_argument("--gap_y_in", type=float, required=True,
                        help="Slat/thickness between rows in inches (e.g., 0.75)")

    args = parser.parse_args()

    global clicked
    pipeline = create_pipeline()

    cv2.namedWindow("RGB")
    cv2.setMouseCallback("RGB", mouse_cb, param=(1280, 720))

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        print("\nControls:")
        print(" - Click rack corners: TL -> TR -> BR -> BL")
        print(" - After 4 clicks: drag corners to refine")
        print(" - Shift+drag anywhere to move whole grid (or right-click drag)")
        print(" - Press 's' to save cubby grid config to cubby_config.json")
        print(" - Press 'r' to reset corners")
        print(" - Press 'q' to quit\n")

        while True:
            in_rgb = q_rgb.tryGet()
            in_depth = q_depth.tryGet()

            if in_rgb is not None:
                rgb = in_rgb.getCvFrame()
            else:
                rgb = rgb if 'rgb' in locals() else None

            if in_depth is not None:
                depth = in_depth.getFrame()
            else:
                depth = depth if 'depth' in locals() else None

            if rgb is None or depth is None:
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                continue

            h, w = rgb.shape[:2]
            cv2.setMouseCallback("RGB", mouse_cb, param=(w, h))

            # Ensure same size
            if depth.shape[:2] != rgb.shape[:2]:
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

            depth_vis = colorize_depth(depth, args.max_depth_mm)

            # draw corner handles (click points / editable handles)
            draw_corner_handles(rgb, clicked)

            # If we have the quad, draw grid and compute occupancy
            if len(clicked) == 4:
                TL, TR, BR, BL = clicked

                # ---- Physical UV grid lines ----
                W_rack, H_rack, u_lines, v_lines = compute_physical_uv_lines(
                    args.rows, args.cols,
                    args.cubby_w_in, args.cubby_h_in,
                    args.gap_x_in, args.gap_y_in
                )

                # --- Estimate rack face depth baseline (Z_plane) ---
                # sample along top edge of quad (near v=0.05)
                samples = []
                for t in np.linspace(0.1, 0.9, 30):
                    x_s, y_s = bilinear_point(TL, TR, BR, BL, t, 0.05)
                    z = median_depth_patch(depth, x_s, y_s, k=4)
                    if z is not None:
                        samples.append(z)
                Z_plane = int(np.median(samples)) if samples else None

                # ---- Draw grid lines using physical u/v positions ----
                # horizontal lines
                for v in v_lines:
                    xL, yL = bilinear_point(TL, TR, BR, BL, 0.0, v)
                    xR, yR = bilinear_point(TL, TR, BR, BL, 1.0, v)
                    cv2.line(rgb, (xL, yL), (xR, yR), (255, 0, 0), 1)

                # vertical lines
                for u in u_lines:
                    xT, yT = bilinear_point(TL, TR, BR, BL, u, 0.0)
                    xB, yB = bilinear_point(TL, TR, BR, BL, u, 1.0)
                    cv2.line(rgb, (xT, yT), (xB, yB), (255, 0, 0), 1)

                # ---- Occupancy check per cubby using depth ----
                if Z_plane is not None:
                    cv2.putText(rgb, f"Rack plane ~ {Z_plane} mm",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    for r in range(args.rows):
                        for c in range(args.cols):
                            u, v = cubby_center_uv(
                                r, c,
                                args.cols, args.rows,
                                args.cubby_w_in, args.cubby_h_in,
                                args.gap_x_in, args.gap_y_in,
                                W_rack, H_rack
                            )
                            cx, cy = bilinear_point(TL, TR, BR, BL, u, v)
                            z_cell = median_depth_patch(depth, cx, cy, k=4)
                            if z_cell is None:
                                continue

                            occupied = (z_cell < (Z_plane - args.depth_thresh_mm))
                            color = (0, 0, 255) if occupied else (0, 255, 0)
                            cv2.circle(rgb, (cx, cy), 2, color, -1)
                else:
                    cv2.putText(rgb, "Rack plane depth: N/A (not enough depth samples)",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("RGB", rgb)
            cv2.imshow("Depth", depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                clicked = []
                print("[reset] cleared corners")
            if key == ord("s") and len(clicked) == 4:
                TL, TR, BR, BL = clicked
                cfg = {
                    "rows": args.rows,
                    "cols": args.cols,
                    "cubby_w_in": args.cubby_w_in,
                    "cubby_h_in": args.cubby_h_in,
                    "gap_x_in": args.gap_x_in,
                    "gap_y_in": args.gap_y_in,
                    "corners": {
                        "TL": TL,
                        "TR": TR,
                        "BR": BR,
                        "BL": BL
                    },
                    "depth_thresh_mm": args.depth_thresh_mm
                }
                with open("cubby_config.json", "w") as f:
                    json.dump(cfg, f, indent=2)
                print("[saved] cubby_config.json")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()