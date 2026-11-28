# Camera demo: live digits from a USB webcam

`pynq/camera_demo.py` classifies digits held up in front of a USB webcam plugged into the board.
The hardware is the same MLP bitstream: the camera sits on the ARM side, which turns each frame into
an MNIST-style 28x28 image and sends it to the FPGA like any test image.

```bash
make board CORE=mlp   # once: build the MLP into hw/vivado (make camera uses that build)
make camera           # live loop on the board: ASCII preview + the FPGA's answer, Ctrl-C stops
```

**Pipeline per frame (on the ARM):**

```
camera frame 640x480 -> grayscale -> centre square (60 % of the height) -> blur
  -> polarity: dark ink on paper is inverted, chalk on a blackboard is kept (median brightness decides)
  -> background removal (70th percentile -> 0) -> tight box around the stroke -> stroke thickening
  -> longest side to 20 px -> pasted into 28x28 -> centre of mass to (14, 14)
  -> quantize_pixels -> pack (4 pixels/word) -> FPGA -> 10 scores -> argmax
```

These are the same steps MNIST itself was made with (digit fits 20x20, centred by mass), so the
network sees images that look like its training data. Raw 720p cannot be the input: 921,600 inputs
would need ~118 M weights in the first layer, far beyond the board's ~0.6 MB of BRAM.

**Output:** the 28x28 image as ASCII art, the predicted digit, the *margin* (best score minus second
best, a rough confidence measure in the network's integer units) and CYCLES / wall time per image.
The ASCII preview is the first thing to check when an answer is wrong: if it does not look like an
MNIST digit, the network has no chance.

**Getting good results:**
- Hold the digit inside the centre of the picture and let it fill most of that square.
- Paper: a thick dark marker on white paper. Blackboard: a clean (wet-wiped) board, pressed chalk,
  and an angle without glare from the ceiling lights.
- Accuracy is lower than the 96.69 % on MNIST, because camera digits differ from the training set
  (lighting, stroke width, angle). Retraining with augmentation would help and needs no hardware change.

**Board setup:** the camera must be a standard UVC webcam (`lsusb`, `ls /dev/video*` on the board),
OpenCV must be importable in the PYNQ venv (`python3 -c "import cv2"`), and the board should run on
the 12 V supply (jumper on REG), since a camera on the USB host port can brown out a USB-powered board.

The FPGA round trip takes ~2 ms per frame including Python; the camera and the preprocessing on the ARM
are the bottleneck.
