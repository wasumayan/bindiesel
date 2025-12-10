# YOLO OBB Detector Troubleshooting

## Issue: No Detections

If the OBB detector isn't detecting anything, try these solutions:

---

## Quick Fixes

### 1. Lower Confidence Threshold
The default confidence might be too high. Try:

```bash
python test_yolo_obb.py --conf 0.05
```

Or even lower:
```bash
python test_yolo_obb.py --conf 0.01
```

### 2. Enable Debug Mode
See what's happening internally:

```bash
python test_yolo_obb.py --debug
```

This will show:
- Model loading status
- Detection attempts
- What YOLO is returning
- Any errors

### 3. Check Model Loading
The model should show:
```
Model loaded: yolo11n-obb.pt
Model task: obb
Model classes: 1 classes
```

If you see errors, the model might not be downloading correctly.

---

## Common Issues

### Issue 1: Model Not Loading Correctly

**Symptoms:**
- Error during model initialization
- "Failed to load" messages

**Solution:**
```bash
# Delete and re-download model
rm yolo11n-obb.pt
python test_yolo_obb.py
```

### Issue 2: No OBB Results

**Symptoms:**
- Model loads but `result.obb` is None
- Shows "No OBB results found"

**Possible Causes:**
1. **Wrong model type**: Make sure you're using an OBB model (`yolo11n-obb.pt`)
2. **Model not OBB-capable**: Some models don't support OBB
3. **No objects in frame**: Try pointing at common objects (bottles, cups, etc.)

**Solution:**
```bash
# Verify model type
python -c "from ultralytics import YOLO; m = YOLO('yolo11n-obb.pt', task='obb'); print(f'Task: {m.task}')"
```

### Issue 3: Confidence Too High

**Symptoms:**
- Model loads fine
- No detections even with objects in frame

**Solution:**
```bash
# Try very low confidence
python test_yolo_obb.py --conf 0.01
```

### Issue 4: Wrong Object Types

**Symptoms:**
- Detections work but no "trash" detected

**Note**: OBB models are typically trained on specific datasets (like DOTA for aerial images). The default `yolo11n-obb.pt` might not detect common objects like bottles/cans.

**Solution:**
- OBB models are usually for specific use cases (aerial images, rotated objects)
- For trash detection, regular YOLO might work better
- Or train a custom OBB model on trash dataset

---

## Testing Steps

### Step 1: Basic Test
```bash
python test_yolo_obb.py --conf 0.1 --debug
```

**What to check:**
- ✅ Model loads without errors
- ✅ Camera starts
- ✅ FPS is shown
- ✅ Debug output appears

### Step 2: Lower Confidence
```bash
python test_yolo_obb.py --conf 0.05
```

**What to check:**
- ✅ Detections appear
- ✅ Bounding boxes drawn
- ✅ Class names shown

### Step 3: Point at Objects
- Point camera at bottles, cups, containers
- Try different angles
- Check if detections appear

### Step 4: Check Regular Detection
If OBB doesn't work, try regular YOLO:

```bash
python -c "
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2

model = YOLO('yolo11n.pt')  # Regular model
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={'size': (640, 480)}))
picam2.start()

frame = picam2.capture_array()
results = model(frame, conf=0.25)
print(f'Detections: {len(results[0].boxes)}')
for box in results[0].boxes:
    print(f'  - {model.names[int(box.cls[0])]}: {float(box.conf[0]):.2f}')
"
```

---

## Understanding OBB Models

**Important**: YOLO OBB models are typically trained on:
- **DOTA dataset**: Aerial/satellite images (planes, ships, vehicles)
- **Specialized datasets**: Rotated objects in specific contexts

**Default `yolo11n-obb.pt` might not detect:**
- Common household objects (bottles, cups)
- Trash items
- Regular everyday items

**OBB models are best for:**
- Aerial/satellite imagery
- Rotated objects in specific domains
- Custom-trained datasets

---

## Alternative: Use Regular YOLO for Trash

If OBB doesn't work for your use case, use regular YOLO:

```python
from ultralytics import YOLO

# Regular detection (works better for common objects)
model = YOLO('yolo11n.pt')  # or yolo11s.pt, yolo11m.pt
results = model(frame, conf=0.25)

# This will detect: bottles, cups, books, phones, etc.
# Which are common "trash" items
```

---

## Debug Output Explained

When running with `--debug`, you'll see:

```
DEBUG - YOLO result type: <class 'ultralytics.engine.results.OBB'>
DEBUG - Has obb: True
DEBUG - Has boxes: False
```

**What this means:**
- `Has obb: True` → OBB results available ✅
- `Has obb: False` → Model might not be OBB type ❌
- `Has boxes: True` → Regular detections available (fallback)

---

## Quick Test Commands

```bash
# Test with low confidence
python test_yolo_obb.py --conf 0.05

# Test with debug
python test_yolo_obb.py --conf 0.1 --debug

# Test with different model
python test_yolo_obb.py --model yolo11s-obb.pt --conf 0.1

# Compare with regular YOLO
python test_yolo_obb.py --compare
```

---

## Expected Behavior

**If working correctly:**
- ✅ Model loads: "Model loaded: yolo11n-obb.pt"
- ✅ Camera starts: "Camera started: 640x480"
- ✅ FPS shown: "FPS: 15.2"
- ✅ Detections appear: "Detections: 2"
- ✅ Bounding boxes drawn on frame
- ✅ Class names shown

**If not working:**
- ❌ No detections: Try `--conf 0.01`
- ❌ Model errors: Check model download
- ❌ Camera errors: Check camera permissions
- ❌ No OBB results: Model might not support OBB

---

## Next Steps

1. **Try lower confidence**: `--conf 0.05` or `--conf 0.01`
2. **Enable debug**: `--debug` to see what's happening
3. **Check model**: Verify it's an OBB model
4. **Try regular YOLO**: If OBB doesn't work for your objects
5. **Train custom model**: If you need specific object detection

---

**Most likely fix: Lower the confidence threshold!**

```bash
python test_yolo_obb.py --conf 0.05 --debug
```

