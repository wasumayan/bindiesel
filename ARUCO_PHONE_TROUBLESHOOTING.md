# ArUco Detection on Phone Screen - Troubleshooting Guide

## Why ArUco Tags on Phone Screens Often Fail

### 1. **Screen Refresh Rate / Flickering**
- **Problem**: Phone screens refresh at 60Hz, 90Hz, or 120Hz. Camera captures at 30fps, causing flickering.
- **Solution**: 
  - Reduce phone screen brightness
  - Use a printed tag instead
  - Try different camera frame rates

### 2. **Reflections and Glare**
- **Problem**: Screen glass reflects light, creating false edges
- **Solution**:
  - Reduce ambient light
  - Angle phone to avoid reflections
  - Use matte screen protector (if available)

### 3. **Tag Size Too Small**
- **Problem**: ArUco needs minimum pixels to detect (usually 20-30px per side)
- **Solution**:
  - Make tag image larger on phone (zoom in)
  - Move camera closer
  - Generate larger tag: `python3 generate_apriltag.py --tag-id 0 --size 400`

### 4. **Focus Issues**
- **Problem**: Camera autofocus may blur the tag
- **Solution**:
  - Tap on phone screen to focus
  - Hold phone steady
  - Ensure tag is in focus before testing

### 5. **Detection Parameters Too Strict**
- **Problem**: Current parameters may reject valid tags
- **Solution**: Already optimized in code, but can be adjusted

### 6. **Dictionary Mismatch**
- **Problem**: Tag generated with one dictionary, detected with another
- **Solution**: Ensure both use `DICT_6X6_250` (already set correctly)

### 7. **Lighting Conditions**
- **Problem**: Too bright/dark, or uneven lighting
- **Solution**:
  - Use even, moderate lighting
  - Avoid direct sunlight on screen
  - Try in different lighting conditions

## Quick Fixes to Try

### 1. Generate Larger Tag
```bash
python3 generate_apriltag.py --tag-id 0 --size 400
```
Display this larger tag on your phone.

### 2. Adjust Detection Parameters (if needed)
The code already has optimized parameters, but you can make them more lenient:

Edit `test_apriltag_detection.py` line ~60-75:
```python
# Make detection more lenient
self.aruco_params.minMarkerPerimeterRate = 0.02  # Allow even smaller markers
self.aruco_params.adaptiveThreshConstant = 5     # Lower threshold
```

### 3. Use Debug Mode
```bash
python3 test_apriltag_detection.py --webcam --no-control --tag-size 0.047 --rotate 180 --debug-detection
```

### 4. Test with Printed Tag
Print the tag instead of using phone screen - this eliminates flickering and reflection issues.

## Best Practices for Phone Testing

1. **Use printed tag** (best option)
2. **If using phone**:
   - Full brightness
   - Zoom in on tag (make it large)
   - Hold steady
   - Good lighting
   - No reflections
   - Focus on tag

3. **Camera settings**:
   - Ensure good focus
   - Adequate lighting
   - No motion blur

## Expected Behavior

When working correctly, you should see:
- Green box around detected marker
- Tag ID displayed
- Angle and distance calculated
- "CENTERED" status when tag is centered

If you see "No ArUco marker detected", try the fixes above.

