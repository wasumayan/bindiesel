# RADD Mode Clothing Detection Model Setup

RADD mode uses a YOLO clothing detection model to accurately identify dress code violations.

## Model Source

Based on: [YOLO-Clothing-Detection](https://github.com/kesimeg/YOLO-Clothing-Detection)

This model detects:
- **Clothing** (pants, shirts, etc.)
- **Shoes** (footwear)
- **Bags**
- **Accessories**

## Setup Options

### Option 1: Use Pre-trained Model (If Available)

If the repository provides a pre-trained model:

```bash
# Download the model
# (Check the repository for download links)

# Place in models directory
mkdir -p models/clothing
# Copy model to models/clothing/best.pt
```

Then update `config.py`:
```python
YOLO_CLOTHING_MODEL = 'models/clothing/best.pt'
```

### Option 2: Train Your Own Model

Follow the instructions in the [YOLO-Clothing-Detection repository](https://github.com/kesimeg/YOLO-Clothing-Detection):

1. **Prepare Dataset**:
   ```bash
   # Clone the repository
   git clone https://github.com/kesimeg/YOLO-Clothing-Detection.git
   cd YOLO-Clothing-Detection
   
   # Run data preprocessing
   python data_preprocessing.py
   ```

2. **Train Model**:
   ```bash
   # Train the model
   python model_training.py
   ```

3. **Copy Trained Model**:
   ```bash
   # Copy best.pt to your project
   cp runs/detect/train/weights/best.pt /path/to/bindiesel/models/clothing/best.pt
   ```

4. **Update Config**:
   ```python
   YOLO_CLOTHING_MODEL = 'models/clothing/best.pt'
   ```

### Option 3: Use Fallback Heuristics (Default)

If no model is provided, RADD mode will use pose-based heuristics:
- Analyzes leg keypoints to detect shorts vs pants
- Analyzes foot angles to detect open-toe shoes

**Note**: Heuristics are less accurate than a trained model.

## Enhanced Detection (Future)

To improve detection accuracy, you can:

1. **Train with Specific Classes**:
   - Add classes: "full_pants", "shorts", "skirt", "closed_toe_shoes", "sandals", "flip_flops"
   - This allows more precise violation detection

2. **Fine-tune on Your Data**:
   - Collect images of people in your environment
   - Label violations (shorts, open-toe shoes)
   - Fine-tune the model

## Testing

Test the clothing detection model:

```bash
# Test with a single image
python -c "
from radd_detector import RADDDetector
import cv2

detector = RADDDetector(model_path='models/clothing/best.pt')
frame = cv2.imread('test_image.jpg')
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

result = detector.detect_violation(frame_rgb)
print(result)
"
```

## Integration

The RADD detector automatically:
1. **Tracks persons** using YOLO pose tracking
2. **Detects violations** per tracked person
3. **Maintains violation state** across frames
4. **Follows violators** persistently until they comply

## Configuration

In `config.py`:

```python
YOLO_CLOTHING_MODEL = 'models/clothing/best.pt'  # Path to clothing model
YOLO_CONFIDENCE = 0.25  # Detection confidence threshold
```

## Usage

1. Say "radd mode" after wake word
2. System will detect and track violators
3. Car will follow violators and "yell" at them when close
4. Violators are tracked by ID until they comply or timeout

## Notes

- Model must be trained on YOLO format (Ultralytics compatible)
- Detection works best with clear view of person
- Multiple violators can be tracked simultaneously
- Violation state persists for 2 seconds after person leaves frame

