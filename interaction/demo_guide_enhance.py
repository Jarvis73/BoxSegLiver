import numpy as np
import matplotlib.pyplot as plt
import cv2

npy1 = r"E:\Temp\112_nf_sp_dp\outputs_npy_volume_36_11\down_conv2_mod_conv2.npy"
npy2 = r"E:\Temp\112_nf_sp_dp\outputs_npy_volume_36_11_g\down_conv2_mod_conv2.npy"
npy3 = r"E:\Temp\112_nf_sp_dp\outputs_npy_volume_36_11_g\spatial_conv2.npy"
npy4 = r"E:\Temp\112_nf_sp_dp\inputs_npy\volume-36-11.npy"

feat1 = np.load(npy1)
feat2 = np.load(npy2)
sp = np.load(npy3)
inp = np.load(npy4)
print(inp.shape)

s1 = feat1[0, 124:180, 40:140, 34]
s2 = feat2[0, 124:180, 40:140, 34]
s0 = sp[0, 124:180, 40:140, 34]
ip = inp[0, 248:360, 80:280, 1]

s0 = cv2.resize(s0, (400, 240), cv2.INTER_LINEAR)
s1 = cv2.resize(s1, (400, 240), cv2.INTER_LINEAR)
s2 = cv2.resize(s2, (400, 240), cv2.INTER_LINEAR)
ip = cv2.resize(ip, (400, 240), cv2.INTER_LINEAR)

min_, max_ = min(s1.min(), s2.min()), max(s1.max(), s2.max())
s1_scale = (s1 - min_) / (max_ - min_) * 255
s2_scale = (s2 - min_) / (max_ - min_) * 255
s0_scale = (s0 - s0.min()) / (s0.max() - s0.min()) * 255
ip_scale = (ip - ip.min()) / (ip.max() - ip.min()) * 255

cv2.imwrite(r"D:\0WorkSpace\MedicalImageSegmentation\snapshots\demo_guide_enhance\sp.png", s0_scale.astype(np.uint8))
cv2.imwrite(r"D:\0WorkSpace\MedicalImageSegmentation\snapshots\demo_guide_enhance\s1.png", s1_scale.astype(np.uint8))
cv2.imwrite(r"D:\0WorkSpace\MedicalImageSegmentation\snapshots\demo_guide_enhance\s2.png", s2_scale.astype(np.uint8))
cv2.imwrite(r"D:\0WorkSpace\MedicalImageSegmentation\snapshots\demo_guide_enhance\input.png", ip_scale.astype(np.uint8))

# plt.imshow(ip, cmap="gray")
# plt.show()
