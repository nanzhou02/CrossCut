from .base import BaseTransform
import jittor as jt

class Resize(BaseTransform):
    def __init__(self, target_size=400):
        super().__init__()
        self.target_size = target_size
        self._input_shape = None  
        self.scale_h = 1.0
        self.scale_w = 1.0

    def transform(self, image_nd, clicks_lists):
        B, C, H, W = image_nd.shape
        self._input_shape = (H, W)

        self.scale_h = self.target_size / H
        self.scale_w = self.target_size / W

        image_resized = jt.nn.interpolate(
            image_nd, size=(self.target_size, self.target_size),
            mode='bilinear', align_corners=True
        )

        new_clicks = []
        for click in clicks_lists[0]:
            y, x = click.coords
            new_y = y * self.scale_h
            new_x = x * self.scale_w
            new_clicks.append(click.copy(coords=(new_y, new_x)))

        return image_resized, [new_clicks]

    def inv_transform(self, prob_map):
        if self._input_shape is None:
            return prob_map

        return jt.nn.interpolate(
            prob_map, size=self._input_shape,
            mode='bilinear', align_corners=True
        )

    def reset(self):
        self._input_shape = None
        self.scale_h = 1.0
        self.scale_w = 1.0

