from math import floor

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import scipy.stats as stats


def generateRandomHeight(h_loc, h_scale, random_state=None):
    h = stats.norm.rvs(loc=h_loc, scale=h_scale, random_state=random_state)
    height = abs(floor(h))
    if height == 0:
        return 1
    else:
        return height


def generateRandomWidth(w_loc, w_scale, random_state=None):
    w = stats.norm.rvs(loc=w_loc, scale=w_scale, random_state=random_state)
    w = abs(floor(w))

    if w == 0:
        return 1
    else:
        return w


def generateRotationAngle(r_loc, r_scale, random_state=None):
    r = stats.norm.rvs(loc=r_loc, scale=r_scale, random_state=random_state)
    r = floor(r)
    if r == 0:
        return 0
    else:
        return r


def generateCoordinates(x_loc, x_scale, y_loc, y_scale, random_state=None):
    x = stats.uniform.rvs(loc=x_loc, scale=x_scale, random_state=random_state)
    y = stats.norm.rvs(loc=y_loc, scale=y_scale, random_state=random_state)
    return abs(floor(x)), abs(floor(y))


class ImplantMaskCreator:
    """Create random masks for implant regions."""

    def __init__(self, resolution: tuple[int, int]) -> None:
        self.resolution = resolution

    def generate_mask(self, random_state=None) -> np.ndarray:
        h = generateRandomHeight(
            self.resolution[1] / 2,
            self.resolution[0] / 32,
            random_state=random_state,
        )
        w = generateRandomWidth(
            self.resolution[1] * 7 / 110,
            self.resolution[1] / 64,
            random_state=random_state,
        )

        x, y = generateCoordinates(
            0,
            self.resolution[0],
            self.resolution[1] // 2,
            0.1,
            random_state=random_state,
        )
        r = generateRotationAngle(
            0,
            20,
            random_state=random_state,
        )

        rectangle = np.ones((h, w), dtype=int)

        rotated_rectangle = ndimage.rotate(rectangle, r, reshape=True, mode="constant")
        rotated_height, rotated_width = rotated_rectangle.shape

        # logic to stay inside the image
        start_0 = min(max(0, y - rotated_height // 2), self.resolution[0])
        start_1 = min(max(0, x - rotated_width // 2), self.resolution[1])
        end_0 = max(
            min(self.resolution[0], (y - rotated_height // 2) + rotated_height), 0
        )
        end_1 = max(
            min(self.resolution[1], (x - rotated_width // 2) + rotated_width), 0
        )

        rec_start_0 = min(max(0, -(y - rotated_height // 2)), rotated_height)
        rec_start_1 = min(max(0, -(x - rotated_width // 2)), rotated_width)
        rec_end_0 = min(rotated_height, self.resolution[0] - start_0)
        rec_end_1 = min(rotated_width, self.resolution[1] - start_1)

        mask = np.zeros(self.resolution, dtype=int)
        mask[start_0:end_0, start_1:end_1] = rotated_rectangle[
            rec_start_0:rec_end_0, rec_start_1:rec_end_1
        ]

        return mask

    def generate_mask_with_n_implants(self, n: int, random_state=None) -> np.ndarray:
        mask = np.zeros(self.resolution, dtype=int)
        for _ in range(n):
            random_state = hash(random_state)
            implant = self.generate_mask(random_state=random_state)
            mask = mask + implant

        mask = np.clip(mask, 0, 1)
        return mask

    def generate_mask_with_random_amount_of_implants(
        self, lower, upper, random_state=None
    ) -> np.ndarray:
        """Generate a mask with a random amount of implants.

        Args:
            lower (int): The lower bound for the number of implants.
            upper (int): The upper bound for the number of implants.

        Returns:
            np.ndarray: The generated mask.
        """
        n = np.random.randint(lower, upper)
        return self.generate_mask_with_n_implants(n, random_state=random_state)


if __name__ == "__main__":
    creator = ImplantMaskCreator((256, 256))
    mask = creator.generate_mask_with_n_implants(2)
    plt.imshow(mask)
    plt.show()
