import numpy as np
import matplotlib.pyplot as plt
import cv2

def stereographic_projection(image, fov):
    H, W = image.shape[:2]

    d = min(H, W)
    fov = np.deg2rad(fov)
    f = d / (2 * np.tan(fov / 2)) # assuming standard rectilinear lens

    y, x = np.indices((H, W))
    y, x = y - (H-1) / 2, x - (W-1) / 2
    rp = np.sqrt(y**2 + x**2)
    r0 = d / (2 * np.tan(0.5 * np.arctan(d / (2 * f))))
    ru = r0 * np.tan(0.5 * np.arctan(rp / f))

    x = x * rp / ru + (W-1)/2
    y = y * rp / ru + (H-1)/2

    out = cv2.remap(image, x.astype(np.float32), y.astype(np.float32),
                    cv2.INTER_LINEAR)

    plt.imshow(out)
    plt.show()

    return out