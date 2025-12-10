# Google Colab Training Guide

## Quick Start

1. **Open Google Colab**: https://colab.research.google.com/

2. **Upload the notebook**: 
   - File > Upload notebook
   - Select `train_hand_keypoints_colab.ipynb`

3. **Enable GPU**:
   - Runtime > Change runtime type
   - Hardware accelerator: **GPU** (T4 or V100)
   - Click Save

4. **Run all cells** (Runtime > Run all)

## What the Notebook Does

1. **Installs dependencies** (ultralytics)
2. **Verifies GPU** is available
3. **Trains the model** (100 epochs, ~2-4 hours)
4. **Saves model** in correct format: `models/hand_keypoints/weights/best.pt`
5. **Downloads model** to your computer

## Training Time

- **GPU (T4)**: ~2-4 hours for 100 epochs
- **GPU (V100)**: ~1-2 hours for 100 epochs
- **CPU**: ~130 hours (don't use CPU!)

## After Training

1. **Download the model** (`best.pt` file)
2. **Save to your project**:
   ```bash
   # In your project directory
   mkdir -p models/hand_keypoints/weights
   # Copy downloaded best.pt to:
   # models/hand_keypoints/weights/best.pt
   ```
3. **Update config.py**:
   ```python
   YOLO_HAND_MODEL = 'models/hand_keypoints/weights/best.pt'
   ```
4. **Commit to git**:
   ```bash
   git add models/hand_keypoints/weights/best.pt
   git commit -m 'Add trained hand-keypoints model from Colab'
   git push origin main
   ```

## Alternative: Save to Google Drive

If you prefer to save to Google Drive instead of downloading:

1. Run the "Optional: Save to Google Drive" cell
2. Authorize Google Drive access
3. Model will be saved to: `/content/drive/MyDrive/bindiesel_hand_keypoints_best.pt`
4. Download from Drive later

## Troubleshooting

### "No GPU detected"
- Go to Runtime > Change runtime type
- Set Hardware accelerator to **GPU**
- Click Save
- Restart runtime

### "Out of memory"
- Reduce batch size in training cell:
  ```python
  batch_size = 8  # Instead of 16
  ```

### "Training is slow"
- Make sure GPU is enabled (check in first cell output)
- Use T4 or V100 GPU (free tier T4 is fine)

## Model Specifications

- **Model**: yolo11n-pose.pt (nano - fastest)
- **Dataset**: hand-keypoints (26,768 images, 21 keypoints per hand)
- **Epochs**: 100 (full training)
- **Image size**: 640x640
- **Output**: ~40 MB model file

## Next Steps

After training and downloading:

1. Test the model locally:
   ```python
   from hand_gesture_controller import HandGestureController
   controller = HandGestureController(
       hand_model_path='models/hand_keypoints/weights/best.pt'
   )
   ```

2. Push to Raspberry Pi:
   ```bash
   git push origin main
   # On Pi:
   git pull origin main
   ```

## Notes

- Colab sessions timeout after ~12 hours of inactivity
- Training will continue even if you close the browser
- Check training progress in the output cells
- Model is automatically saved as "best.pt" (best validation performance)

