import numpy as np
import cv2

channels = 4
rows = 10
cols = 10
masks = np.arange(channels * rows * cols) + 100
masks = masks.reshape((4, 10, 10)).astype(np.float32)
print(masks.astype(np.int32))

dst_rows = 15
dst_cols = 15
masks = masks.transpose((1, 2, 0))
dst_masks = cv2.resize(masks, (dst_cols, dst_rows), interpolation=cv2.INTER_LINEAR)
if len(dst_masks.shape) == 2:
    dst_masks = dst_masks[:, :, None]

dst_masks = dst_masks.transpose((2, 0, 1))

print(dst_masks.astype(np.int32))
