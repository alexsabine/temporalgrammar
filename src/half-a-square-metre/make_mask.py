# Builds mask.webp from photo.webp: R = running water, G = saxifrage leaves, B = all living cover.
# Thresholds were tuned by eye against the photograph. Run: python3 make_mask.py
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
im = Image.open('photo.webp').convert('RGB'); W, H = im.size
hsv = np.asarray(im.convert('HSV')).astype(float)
h = hsv[..., 0] / 255 * 360; s = hsv[..., 1] / 255; v = hsv[..., 2] / 255
def poly(pts, b):
    m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).polygon(pts, fill=255)
    return np.asarray(m.filter(ImageFilter.GaussianBlur(b))) / 255.
def blur(m, r):
    return np.asarray(Image.fromarray((np.clip(m, 0, 1) * 255).astype('uint8')).filter(ImageFilter.GaussianBlur(r))) / 255.
water = poly([(0, 1300), (230, 1275), (400, 1262), (520, 1300), (640, 1290), (760, 1235), (900, 1195), (1146, 1140), (1146, 1528), (0, 1528)], 22)
upper = poly([(0, 0), (1146, 0), (1146, 960), (720, 1000), (600, 1120), (330, 1230), (0, 1250)], 14)
yy, xx = np.mgrid[0:H, 0:W]
leaf = ((h > 64) & (h < 100) & (v > 0.5) & (s > 0.25) & (s < 0.68)).astype(float)
zone = ((xx > 600) & (yy < 880)) | ((xx > 200) & (xx < 400) & (yy > 320) & (yy < 720)) | ((yy < 200) & (xx > 480))
leaf = (blur(leaf, 3) > 0.45) * zone
leaf = blur(np.clip(blur(leaf, 6) * 2.0, 0, 1), 3)
veg = (((h > 40) & (h < 68) & (s > 0.5) & (v > 0.2)) | ((h > 38) & (h < 70) & (s > 0.36) & (v > 0.3) & (yy < 330))).astype(float)
veg = np.clip(blur(veg, 5) * 1.3 + leaf, 0, 1) * upper
m = np.dstack([water, leaf, veg])
Image.fromarray((m * 255).astype('uint8')).resize((W // 2, H // 2), Image.LANCZOS).save('mask.webp', quality=86)
print('living cover %.2f  water %.2f' % (veg.mean(), water.mean()))
