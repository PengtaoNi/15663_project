import numpy as np
import matplotlib.pyplot as plt

def get_meshes(image, fov, dim=(78, 103), padding=4):
    H, W = image.shape[:2]

    d = min(H, W)
    fov = np.deg2rad(fov)
    f = d / (2 * np.tan(fov / 2)) # assuming standard rectilinear lens

    h = dim[0] + 2 * padding
    w = dim[1] + 2 * padding
    
    y, x = np.indices((h, w), dtype=np.float32)
    y = y - (h-1) / 2
    x = x - (w-1) / 2
    y = y * (H-1) / (h-1)
    x = x * (W-1) / (w-1)

    source_mesh = np.stack((y, x), axis=-1)

    rp = np.sqrt(y**2 + x**2)
    r0 = d / (2 * np.tan(0.5 * np.arctan(d / (2 * f))))
    ru = r0 * np.tan(0.5 * np.arctan(rp / f))

    y = y * rp / ru
    x = x * rp / ru

    target_mesh = np.stack((y, x), axis=-1)

    return source_mesh, target_mesh

def plot_mesh(mesh):
    plt.figure(dpi=800)
    plt.plot(mesh[:, :, 1],   mesh[:, :, 0],   color='blue', linewidth=0.5)
    plt.plot(mesh[:, :, 1].T, mesh[:, :, 0].T, color='blue', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()