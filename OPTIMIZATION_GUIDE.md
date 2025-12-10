# Performance Optimization Guide

## Current Performance Bottlenecks

1. **Frame Processing**
   - Multiple `get_frame()` calls per loop
   - Redundant color space conversions
   - Unnecessary frame copies

2. **Model Inference**
   - YOLO inference on every frame (even when not needed)
   - No frame skipping for non-critical states
   - Multiple model calls for same frame

3. **Logging Overhead**
   - Too many log calls in hot paths
   - String formatting on every call
   - File I/O blocking main thread

4. **State Machine**
   - Visual updates called multiple times
   - No caching of detection results
   - Redundant calculations

## Optimization Strategies

### 1. Frame Caching
- Cache frames per loop iteration
- Share frames between components
- Avoid redundant `get_frame()` calls

### 2. Conditional Processing
- Skip visual updates when not needed
- Process only required components per state
- Use frame skipping for non-critical operations

### 3. Model Optimization
- Use smaller models (nano vs small/medium)
- Reduce inference frequency
- Batch process when possible

### 4. Logging Optimization
- Use lazy logging (only format when needed)
- Reduce log frequency
- Use async logging for file writes

### 5. Memory Management
- Reuse arrays instead of creating new ones
- Minimize numpy array copies
- Use in-place operations where possible

### 6. Threading/Async
- Async I/O for sensors
- Background logging
- Parallel processing where safe

## Target Performance

- **FPS**: >15 FPS on Raspberry Pi 4
- **Latency**: <100ms detection-to-action
- **CPU Usage**: <80% average
- **Memory**: <500MB total

## Implementation Priority

1. **High Priority** (Immediate impact):
   - Frame caching
   - Conditional visual updates
   - Reduce logging frequency

2. **Medium Priority** (Good impact):
   - Frame skipping
   - Model size optimization
   - Memory reuse

3. **Low Priority** (Nice to have):
   - Async logging
   - Threading
   - Advanced caching

