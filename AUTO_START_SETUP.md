# Auto-Start Conda Environment on Boot

## Quick Setup (Recommended)

### Step 1: Find Your Paths

```bash
# Activate conda environment
conda activate bindiesel

# Get Python path
which python
# Example output: /home/pi/miniforge3/envs/bindiesel/bin/python

# Get project directory
pwd
# Example output: /home/pi/Desktop/angleestimationbindiesel

# Get your username
whoami
# Example output: pi
```

### Step 2: Update Service File

```bash
# Copy service file
sudo cp bin-diesel.service /etc/systemd/system/

# Edit with your paths
sudo nano /etc/systemd/system/bin-diesel.service
```

**Update these lines with YOUR paths:**
- `User=pi` → Your username
- `WorkingDirectory=/home/pi/Desktop/angleestimationbindiesel` → Your project path
- `ExecStart=/home/pi/miniforge3/envs/bindiesel/bin/python` → Your Python path
- `Environment="PATH=..."` → Your conda bin path

### Step 3: Enable Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable bin-diesel.service
sudo systemctl start bin-diesel.service
sudo systemctl status bin-diesel.service
```

**Done!** The system will now start automatically on boot.

## Method 1: Systemd Service (Recommended - Runs on Boot)

This creates a system service that starts automatically on boot.

### Step 1: Create Service File

```bash
# Copy the service file to systemd directory
sudo cp bin-diesel.service /etc/systemd/system/

# Edit the service file to match your paths
sudo nano /etc/systemd/system/bin-diesel.service
```

**Update these paths in the service file:**
- `WorkingDirectory`: Your project directory (e.g., `/home/pi/Desktop/angleestimationbindiesel`)
- `User`: Your username (e.g., `pi` or `mayanwasu`)
- `ExecStart`: Full path to Python in conda environment
- `Environment PATH`: Path to conda environment bin

### Step 2: Find Your Conda Python Path

```bash
# Activate conda environment
conda activate bindiesel

# Get Python path
which python
# Output: /home/pi/miniforge3/envs/bindiesel/bin/python

# Get project directory
pwd
# Output: /home/pi/Desktop/angleestimationbindiesel
```

### Step 3: Update Service File

Edit `/etc/systemd/system/bin-diesel.service` with your paths:

```ini
[Unit]
Description=Bin Diesel Robot Car System
After=network.target sound.target

[Service]
Type=simple
User=pi  # Change to your username
WorkingDirectory=/home/pi/Desktop/angleestimationbindiesel  # Your project path
Environment="PATH=/home/pi/miniforge3/envs/bindiesel/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/pi/miniforge3/envs/bindiesel/bin/python /home/pi/Desktop/angleestimationbindiesel/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Step 4: Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (starts on boot)
sudo systemctl enable bin-diesel.service

# Start service now
sudo systemctl start bin-diesel.service

# Check status
sudo systemctl status bin-diesel.service

# View logs
sudo journalctl -u bin-diesel.service -f
```

### Step 5: Useful Commands

```bash
# Stop service
sudo systemctl stop bin-diesel.service

# Restart service
sudo systemctl restart bin-diesel.service

# Disable auto-start
sudo systemctl disable bin-diesel.service

# View recent logs
sudo journalctl -u bin-diesel.service -n 50
```

---

## Method 2: .bashrc Auto-Activation (For Interactive Sessions)

This activates conda when you open a terminal (not for system boot).

### Add to ~/.bashrc

```bash
# Add to end of ~/.bashrc
nano ~/.bashrc

# Add these lines:
source ~/miniforge3/etc/profile.d/conda.sh
conda activate bindiesel
cd ~/Desktop/angleestimationbindiesel  # Your project directory
```

**Note**: This only works for interactive terminal sessions, not for system services.

---

## Method 3: Crontab (Alternative)

```bash
# Edit crontab
crontab -e

# Add this line (runs on boot, waits 30 seconds for system to be ready):
@reboot sleep 30 && /home/pi/miniforge3/envs/bindiesel/bin/python /home/pi/Desktop/angleestimationbindiesel/main.py
```

---

## Method 4: Desktop Autostart (For GUI Sessions)

If you're using a desktop environment:

```bash
# Create autostart file
mkdir -p ~/.config/autostart
nano ~/.config/autostart/bin-diesel.desktop
```

Add this content:
```ini
[Desktop Entry]
Type=Application
Name=Bin Diesel
Exec=/home/pi/miniforge3/envs/bindiesel/bin/python /home/pi/Desktop/angleestimationbindiesel/main.py
Terminal=true
```

---

## Recommended: Method 1 (Systemd)

Systemd service is the best option because:
- ✅ Starts automatically on boot
- ✅ Restarts if it crashes
- ✅ Runs in background
- ✅ Logs to journal
- ✅ Works without GUI/login

## Testing

After setting up, test it:

```bash
# Reboot and check if service started
sudo reboot

# After reboot, check status
sudo systemctl status bin-diesel.service
```

