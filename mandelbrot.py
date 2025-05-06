import torch

# Make use of a GPU or MPS (Apple) if one is available.
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
#device="cpu"
print(f"Using device: {device}")

from io import BytesIO

import numpy as np
import PIL.Image


# Take a NumPy array and render it as a Mandelbrot.
def render(a):
    a_cyclic = (a * 0.3).reshape(list(a.shape) + [1])
    img = np.concatenate(
        [
            10 + 20 * np.cos(a_cyclic),
            30 + 50 * np.sin(a_cyclic),
            155 - 80 * np.cos(a_cyclic),
        ],
        2,
    )
    img[a == a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()
    return PIL.Image.fromarray(a)


# Loop through the render cycles for. Mandlebrot plot.
def mandelbrot_helper(grid_c, current_values, counts, cycles):
    for i in range(cycles):
        # The Mandlebrot formula
        temp = current_values * current_values + grid_c
        not_diverged = torch.abs(temp) < 4
        current_values.copy_(temp)
        counts.copy_(torch.add(counts, not_diverged.double()))

# Render a Mandelbrot plot at the specified location, zoom, and render cycles.
def mandelbrot(render_size, center, zoom, cycles):
    f = zoom / render_size[0]

    real_start = center[1] - (render_size[1] / 2) * f
    real_end = real_start + render_size[1] * f
    imag_start = center[0] - (render_size[0] / 2) * f
    imag_end = imag_start + render_size[0] * f

    real_range = torch.arange(
        real_start, real_end, f, dtype=torch.float32, device=device
    )
    imag_range = torch.arange(
        imag_start, imag_end, f, dtype=torch.float32, device=device
    )
    real, imag = torch.meshgrid(real_range, imag_range, indexing="ij")
    grid_c = torch.complex(imag, real)
    current_values = torch.clone(grid_c)
    counts = torch.Tensor(torch.zeros_like(grid_c, dtype=torch.float32))

    mandelbrot_helper(grid_c, current_values, counts, cycles)
    return counts.cpu().numpy()

    import os

# Pytorch does not currently support complex numbers on MPS (as of 2024-01-07)
temp_device = device
if device == "mps":
    device = "cpu"

counts = mandelbrot(
    # render_size=(1920,1080), # HD
    render_size=(int(1280*1.5), int(960*1.5)),
    center=(-0.5, 0),
    zoom=1,
    cycles=200,
)

img = render(counts)
print(img.size)
img.show()

# restore device
device = temp_device

