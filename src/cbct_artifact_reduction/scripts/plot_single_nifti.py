import os

import cbct_artifact_reduction.dataprocessing as dp
import cbct_artifact_reduction.utils as utils
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
from skimage.morphology import binary_closing, binary_opening, square

path_to_file = os.path.join(utils.ROOT_DIR, "output/frames/256x256/141_0.nii.gz")

numpy_array = dp.single_nifti_to_numpy(path_to_file)
numpy_array = numpy_array
numpy_array = numpy_array.astype(np.float32)
numpy_array = cv2.normalize(numpy_array, None, 0, 255, cv2.NORM_MINMAX)
# numpy_array = denoise_bilateral(numpy_array, sigma_color=1, sigma_spatial=15)
# canny = CannyTrackbar(numpy_array)
plt.imshow(numpy_array)
plt.show()
# avg = np.mean(numpy_array)
# sigma = 0.33
# lower_threshold = int(max(0, (1.0 - sigma) * avg))
# upper_threshold = int(min(255, (1.0 + sigma) * avg))
# footprint = morphology.disk(10)
# numpy_array = numpy_array - morphology.white_tophat(numpy_array, footprint=footprint)
plt.imshow(numpy_array, cmap="gray")

edges = (
    feature.canny(
        numpy_array,
        sigma=3,
    )
    * 255
)
closed_edges = binary_closing(edges, square(5)) * 255
closed_edges = closed_edges.astype(np.uint8)

contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filled_contours = np.zeros_like(closed_edges)
hulls = []
for countour in contours:
    hulls.append(cv2.convexHull(countour))


# contours = tuple(hulls)
cv2.drawContours(filled_contours, contours, -1, 255, -1)
plt.imshow(filled_contours)
plt.show()


filled_contours = binary_opening(filled_contours, square(10), mode="ignore") * 255


contours, _ = cv2.findContours(
    filled_contours.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
print([cv2.contourArea(contour) for contour in sorted_contours])
n = 3

largest_contours = sorted_contours[:n]

contours_image = np.zeros_like(filled_contours)
contours_image = contours_image.astype(np.uint8)
cv2.drawContours(contours_image, largest_contours, -1, 255, -1)

plt.imshow(contours_image)
plt.show()
plt.imshow(numpy_array)
plt.show()

# plt.imshow(filled_contours)
# plt.show()
# plt.imshow(edges, cmap="gray")
# plt.show()
