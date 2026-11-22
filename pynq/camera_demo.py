"""camera_demo.py — classify handwritten digits from a USB webcam on the FPGA.

Run on the board:  make camera   (or: sudo python3 camera_demo.py --overlay accel.bit)
Hold a digit written with a thick dark pen on white paper in the middle of the picture.
Each frame: MNIST-style 28x28 on the ARM, inference on the FPGA, ASCII preview + answer. Ctrl-C stops.
"""

import argparse
import time

import cv2
import numpy as np

from mlp_test import HERE, ascii_digit, load_network, pack, quantize_pixels, signed


def to_mnist(frame, crop=0.6):
    """Camera frame (BGR) -> 28x28 uint8 like MNIST: bright digit on black, fits 20x20, centred by mass."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    s = int(min(h, w) * crop)                                   # square in the middle, where the digit is held
    g = g[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    b = cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)       # Removes solitary ink (less than 5x5)
    inv = b if np.median(b) < 128 else 255.0 - b                # dark background (blackboard): chalk is already bright-on-dark  
    inv = np.clip(inv - np.percentile(inv, 70), 0, None)       # paper -> 0, ink stays (expect less than 30% is ink, rest is paper. Clip removes less than zero)
    if inv.max() < 30:
        return None                                             # nothing that looks like ink
    ys, xs = np.nonzero(inv > 0.25 * inv.max())                 # bounding box of the stroke
    d = inv[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    t = max(1, int(0.06 * max(d.shape)))           # stroke width relative to digit size
    d = cv2.dilate(d, np.ones((t, t), np.uint8))   # fills chalk gaps, thickens thin lines
    k = 20.0 / max(d.shape)                                     
    d = cv2.resize(d, (max(1, round(d.shape[1] * k)), max(1, round(d.shape[0] * k))), interpolation=cv2.INTER_AREA)     # longest side -> 20 px, keep aspect ratio
    out = np.zeros((28, 28), np.float32)
    y0, x0 = (28 - d.shape[0]) // 2, (28 - d.shape[1]) // 2
    out[y0:y0 + d.shape[0], x0:x0 + d.shape[1]] = d             # pastes image into 28x28 black box with 4pixel border.
    yy, xx = np.indices(out.shape)                              
    m = out.sum()
    dy, dx = 14 - (yy * out).sum() / m, 14 - (xx * out).sum() / m
    out = cv2.warpAffine(out, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))  # centre of mass -> pixel (14, 14), like MNIST + pixels shifted out are dropped
    return np.clip(out * (255.0 / out.max()), 0, 255).astype(np.uint8)      # Makes brigtesh ink 255 and exports uint8 like MNIST.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default="accel.bit")
    ap.add_argument("--camera", type=int, default=0, help="/dev/videoN")
    ap.add_argument("--every", type=float, default=0.5, help="seconds between classifications")
    a = ap.parse_args()

    from accel import Accel
    acc = Accel(a.overlay)
    m = acc.manifest
    if m["core"] != "mlp":
        raise SystemExit(f"overlay core is '{m['core']}', not 'mlp' (run make board CORE=mlp first)")
    net = load_network(HERE)
    bits, sw, ppb = net["input_bits"], m["stream_w"], m.get("pixels_per_beat", 1)
    acc.set_arg(0, 1)                                           # one image per DMA transfer

    cam = cv2.VideoCapture(a.camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)                      # 720p is not needed for a 28x28 input
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cam.isOpened():
        raise SystemExit(f"no camera at /dev/video{a.camera} (check lsusb, ls /dev/video*)")
    try:
        while True:
            for _ in range(4): cam.grab()   # throw away queued frames
            ok, frame = cam.read()
            if not ok:
                continue
            img = to_mnist(frame)
            if img is None:
                print("no digit in view", flush=True)
            else:
                x = pack(quantize_pixels(img.reshape(1, 784), bits), ppb, bits)
                t0 = time.perf_counter()
                out, cycles = acc.run(x[0])
                dt = time.perf_counter() - t0
                s = signed(out, sw)
                top2 = np.sort(s)[-2:]
                print("\033[2J\033[H" + ascii_digit(img))       # clear the terminal, draw the 28x28
                print(f"FPGA says {int(s.argmax())}   margin {int(top2[1] - top2[0])}   "
                      f"{cycles} cycles, {dt * 1e3:.2f} ms incl. Python", flush=True)
            time.sleep(a.every)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        acc.close()


if __name__ == "__main__":
    main()