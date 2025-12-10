# Model Training Tips

## Running Long Training Sessions

### Option 1: Use Screen (Recommended for SSH)

If training on Raspberry Pi via SSH, use `screen` to keep training running after disconnection:

```bash
# Install screen if not already installed
sudo apt install screen

# Start a new screen session
screen -S yolo_training

# Run your training command
python train_hand_keypoints.py

# Detach from screen: Press Ctrl+A, then D
# (You can now close your laptop/SSH connection)

# Reattach later:
screen -r yolo_training
```

### Option 2: Use tmux (Alternative)

```bash
# Install tmux
sudo apt install tmux

# Start new session
tmux new -s yolo_training

# Run training
python train_hand_keypoints.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t yolo_training
```

### Option 3: Use nohup (Simple Background Process)

```bash
# Run with nohup (no hang up)
nohup python train_hand_keypoints.py > training.log 2>&1 &

# Check progress
tail -f training.log

# Check if still running
ps aux | grep train_hand_keypoints
```

### Option 4: If Training on MacBook

**Problem**: Closing laptop lid puts it to sleep, pausing training.

**Solutions**:

1. **Prevent Sleep When Lid Closed** (macOS):
   ```bash
   # Install caffeinate to prevent sleep
   caffeinate -d python train_hand_keypoints.py
   ```

2. **Use SSH to Raspberry Pi Instead**:
   - Train on Pi (better for long sessions)
   - SSH in and use screen/tmux
   - Can disconnect safely

3. **Keep Laptop Plugged In and Open**:
   - System Settings → Energy Saver
   - Disable "Put hard disks to sleep"
   - Keep lid open or use external monitor

### Checking Training Progress

```bash
# If using screen
screen -r yolo_training

# If using nohup
tail -f training.log

# Check GPU/CPU usage
htop  # or top
```

### Stopping Training Safely

```bash
# Find process ID
ps aux | grep train

# Stop gracefully (saves checkpoint)
kill -SIGINT <PID>

# Or if in screen/tmux:
# Press Ctrl+C in the session
```

## Best Practices

1. **Always train on Raspberry Pi** if possible (dedicated hardware)
2. **Use screen/tmux** for SSH sessions
3. **Save checkpoints frequently** (configure in training script)
4. **Monitor system resources** (CPU, memory, temperature)
5. **Keep logs** for debugging and progress tracking

## Current Training Status

If training is already running:
- **On Pi via SSH**: Use `screen -r` or `tmux attach` to check
- **On MacBook**: Keep laptop open and plugged in, or transfer to Pi

