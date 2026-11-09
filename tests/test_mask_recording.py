"""mask_recording: nothing away from the hand survives, in either eye."""

import numpy as np

from mask_recording import hand_mask, mask_frame

LEFT = [[100.0, 100.0], [200.0, 100.0], [150.0, 200.0]]
RIGHT = [[300.0, 300.0], [400.0, 300.0], [350.0, 400.0]]


def test_the_mask_covers_the_joints_and_uses_raw_rows():
    mask = hand_mask(LEFT, 4)
    assert mask.shape == (240, 640)
    assert mask[50, 100] and mask[50, 200] and mask[100, 150]  # joints at y / 2
    assert not mask[0, 0] and not mask[120, 300]
    assert hand_mask(LEFT, 20).sum() > mask.sum()  # the margin grows it


def test_each_eye_keeps_only_its_own_outline():
    frame = np.full((240, 640, 2), 200, dtype=np.uint8)
    out = mask_frame(frame, LEFT, RIGHT, 4)
    assert out[50, 150, 0] == 200 and out[50, 150, 1] == 0
    assert out[160, 350, 1] == 200 and out[160, 350, 0] == 0
    assert (out[..., 0][~hand_mask(LEFT, 4)] == 0).all()
    assert (out[..., 1][~hand_mask(RIGHT, 4)] == 0).all()
    assert (frame == 200).all()  # the input is untouched
