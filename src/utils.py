import numpy as np
import matplotlib.pyplot as plt
import cv2

# def stereographic_projection(image, fov):
#     H, W = image.shape[:2]

#     d = min(H, W)
#     # d = np.sqrt(H**2 + W**2)
#     fov = np.deg2rad(fov)
#     f = d / (2 * np.tan(fov / 2)) # assuming standard rectilinear lens

#     y, x = np.indices((H, W), dtype=np.float32)
#     y = y - (H-1) / 2
#     x = x - (W-1) / 2
    
#     rp = np.sqrt(y**2 + x**2)
#     r0 = d / (2 * np.tan(0.5 * np.arctan(d / (2 * f))))
#     ru = r0 * np.tan(0.5 * np.arctan(rp / f))

#     y = y * rp / ru + (H-1)/2
#     x = x * rp / ru + (W-1)/2

#     out = cv2.remap(image, x, y, cv2.INTER_LINEAR)

#     plt.imshow(out)
#     plt.show()

#     return out

def get_meshes(image, fov, dim=(103, 78), padding=4):
    H, W = image.shape[:2]
    if H < W:
        dim = (dim[1], dim[0])

    h = dim[0] + 2 * padding
    w = dim[1] + 2 * padding

    H = (H-1) * (h-1) / dim[0] + 1
    W = (W-1) * (w-1) / dim[1] + 1

    d = max(H, W)
    # d = np.sqrt(H**2 + W**2)
    fov = np.deg2rad(fov)
    f = d / (2 * np.tan(fov / 2)) # assuming standard rectilinear lens
    
    y, x = np.indices((h, w), dtype=np.float32)
    y = y - (h-1) / 2
    x = x - (w-1) / 2
    y = y * (H-1) / (h-1)
    x = x * (W-1) / (w-1)

    source_mesh = np.stack((y, x), axis=-1)

    rp = np.sqrt(y**2 + x**2)
    r0 = d / (2 * np.tan(0.5 * np.arctan(d / (2 * f))))
    ru = r0 * np.tan(0.5 * np.arctan(rp / f))

    y = y * ru / rp
    x = x * ru / rp

    y_ratio = (H-1) / (y.max() - y.min())
    x_ratio = (W-1) / (x.max() - x.min())
    ratio = min(y_ratio, x_ratio)

    y = y * ratio
    x = x * ratio

    stereographic_mesh = np.stack((y, x), axis=-1)

    return source_mesh, stereographic_mesh

def warp(image, mesh, downsample=1):
    H, W = image.shape[:2]
    H, W = H // downsample, W // downsample

    image = cv2.resize(image, (W, H))
    out = np.full_like(image, 255)
    
    mesh = cv2.resize(mesh, (W, H))
    mesh = mesh / downsample

    indices_y = mesh[:, :, 0] + (H - 1) / 2 # (H, W)
    indices_x = mesh[:, :, 1] + (W - 1) / 2 # (H, W)

    for y in range(H-1):
        for x in range(W-1):
            tl_y, tl_x = indices_y[y, x], indices_x[y, x]
            tr_y, tr_x = indices_y[y, x+1], indices_x[y, x+1]
            bl_y, bl_x = indices_y[y+1, x], indices_x[y+1, x]
            br_y, br_x = indices_y[y+1, x+1], indices_x[y+1, x+1]

            t, b = min(tl_y, tr_y), max(bl_y, br_y)
            l, r = min(tl_x, bl_x), max(tr_x, br_x)

            for j in range(np.ceil(t).astype(int), np.floor(b).astype(int)+1):
                for i in range(np.ceil(l).astype(int), np.floor(r).astype(int)+1):
                    if j < 0 or j >= H or i < 0 or i >= W:
                        continue

                    # Check if point (i, j) is inside the polygon
                    polygon = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]])
                    is_inside = cv2.pointPolygonTest(polygon, (i, j), False) >= 0

                    if is_inside:
                        ty = (tl_y + tr_y) / 2
                        by = (bl_y + br_y) / 2
                        lx = (tl_x + bl_x) / 2
                        rx = (tr_x + br_x) / 2
                        frac_y = (j - ty) / (by - ty)
                        frac_x = (i - lx) / (rx - lx)

                        out[j, i] = (1 - frac_y) * (1 - frac_x) * image[y, x] + \
                                    frac_y * (1 - frac_x) * image[y+1, x] + \
                                    (1 - frac_y) * frac_x * image[y, x+1] + \
                                    frac_y * frac_x * image[y+1, x+1]

    return out

def plot_mesh(mesh, save=None):
    plt.figure(dpi=800)
    plt.plot(mesh[:, :, 1],   mesh[:, :, 0],   color='blue', linewidth=0.5)
    plt.plot(mesh[:, :, 1].T, mesh[:, :, 0].T, color='blue', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

def radial_sigmoid(image, source_mesh, fov):
    H, W = image.shape[:2]
    
    d = np.sqrt(H**2 + W**2) / 2
    fov = np.deg2rad(fov)
    
    r_100  = d * np.tan(np.deg2rad(100)/2) / np.tan(fov/2)
    rb = r_100 / (np.log(1/0.01 - 1) - np.log(1/0.99 - 1))
    ra = rb * np.log(1/0.01 - 1)

    r = np.linalg.norm(source_mesh, axis=-1)
    m = 1 / (1 + np.exp(-(r - ra) / rb))

    return m