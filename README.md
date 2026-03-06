
# Owligator – Part 1: Cubby Detection & Grid Mapping

This module builds a physically accurate cubby grid on a wine rack using the OAK-D S2 depth camera.

It uses:
- Real cubby width/height (inches)
- Real gap spacing between cubbies
- Perspective-correct bilinear mapping
- Depth-based occupancy detection
- Interactive calibration (click + drag)

---

## What This System Does

1. User clicks the 4 outer rack corners (TL → TR → BR → BL)
2. A mathematically accurate cubby grid is generated using:
   - Cubby width
   - Cubby height
   - Gap thickness (posts/slats)
3. The grid is projected using perspective mapping
4. Depth is sampled at each cubby center
5. Cubbies are marked occupied if depth < rack plane − threshold
6. Configuration is saved to JSON

---

## 🖥️ System Requirements

- Python 3.10+
- OAK-D S2 connected via USB
- macOS / Linux supported
- numpy < 2
- DepthAI installed

---

## Setup Instructions

### 1️ Clone the Repository

```bash
git clone <your-repo-url>
cd owligator
```

### 2️ Create Virtual Environment

```bash
python3 -m venv .venv
```

### 3️ Activate Virtual Environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

You should see:

```
(.venv) yourname@machine owligator %
```

### 4️ Install Dependencies

⚠ IMPORTANT: Use numpy<2 to avoid OpenCV / DepthAI conflicts.

```bash
pip install numpy<2
pip install opencv-python
pip install depthai
```

---

## Running the Cubby Mapper

Main script:

```
rack_cubby_depth_mapper.py
```

### Required Parameters

- --rows → number of cubby rows
- --cols → number of cubby columns
- --cubby_w_in → cubby opening width (inches)
- --cubby_h_in → cubby opening height (inches)
- --gap_x_in → vertical post thickness (inches)
- --gap_y_in → horizontal slat thickness (inches)

---

## Example Command

```bash
python3 rack_cubby_depth_mapper.py   --rows 10   --cols 12   --cubby_w_in 4   --cubby_h_in 3   --gap_x_in 1.0   --gap_y_in 0.75
```

---

## 🖱️ Calibration Workflow

### Step 1: Click Rack Corners

Click in this order:
1. Top-Left
2. Top-Right
3. Bottom-Right
4. Bottom-Left

### Step 2: Refine Alignment

After 4 clicks:
- Drag a corner to adjust perspective
- Hold Shift + drag to move the entire grid
- Right-click drag (if mouse supported) to move whole grid

### Step 3: Save Configuration

Press:

```
s
```

This saves:

```
cubby_config.json
```

---

## 🔍 Depth Occupancy Logic

1. Sample depth along rack top edge → estimate rack plane
2. Compute median depth → Z_plane
3. For each cubby center:
   - Sample depth
   - If depth < (Z_plane - depth_threshold)
   - Mark occupied (red)
   - Else empty (green)

---

## 🛠 Controls

| Key | Action |
|-----|--------|
| r   | Reset corners |
| s   | Save config |
| q   | Quit program |

---

## ⚠ Common Issues

### NumPy Error

If you see:

```
AttributeError: _ARRAY_API not found
```

Fix:

```bash
pip uninstall numpy
pip install numpy<2
```

### Camera Not Detected

Unplug and replug OAK-D or check USB connection.

---

## 📁 Project Structure

```
owligator/
│
├── rack_cubby_depth_mapper.py
├── cubby_config.json
├── README.md
└── .venv/
```

## Run Command Example:
python3 rack_cubby_depth_mapper.py \
  --input_mode image \
  --image_path "/Users/srikar/Desktop/IMG_7437.jpeg" \
  --rows 6 \
  --cols 10 \
  --cubby_w_in 4 \
  --cubby_h_in 3 \
  --gap_x_in 0.875 \
  --gap_y_in 0.75 \
  --left_margin_in 0.5 \
  --right_margin_in 0.5 \
  --top_margin_in 0.5 \
  --bottom_margin_in 0.5