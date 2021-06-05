import cv2
import matplotlib.pyplot as plt
import numpy as np
import Box2D
from MiDaS import depth_utils

kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))


def get_color_rect(frame, fit_rect=True):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (45, 25, 25), (95, 255, 255))

    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_medium)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contour = max(contours, key=lambda c: cv2.contourArea(c))
    area = cv2.contourArea(contour)
    if area < 1000:
        return None
    if fit_rect:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return np.int0(box)
    else:
        hull = cv2.convexHull(contour)
        scale_down = np.ceil(hull.shape[0] / 16)
        new_hull = []
        for i, h in enumerate(hull):
            if i % scale_down == 0:
                new_hull.append(h)
        return np.int0(new_hull).reshape((-1, 2))


def add_ball(world, bodies, pos, vel, radius=10):
    bodyDef = Box2D.b2BodyDef()
    bodyDef.type = Box2D.b2_dynamicBody
    bodyDef.position = Box2D.b2Vec2(pos[0], pos[1])
    bodyDef.linearVelocity = Box2D.b2Vec2(vel[0], vel[1])
    body = world.CreateBody(bodyDef)

    shape = Box2D.b2CircleShape(radius=radius)
    body.CreateFixture(shape=shape, density=1e-1,
                       restitution=1.)

    bodies.append(body)


# depth_utils.load_model()
# video = cv2.VideoCapture(1)
# while True:
#     ret, frame = video.read()
#
#     canvas = depth_utils.predict_depth(frame[:, :, ::-1])
#     _, mask = cv2.threshold(canvas, 200, 255, cv2.THRESH_BINARY)
#     # canvas = frame.copy()
#     # bbox = get_color_rect(canvas[:, :, ::-1])
#     # if bbox is not None:
#     #     cv2.drawContours(canvas, [bbox], 0, (0, 0, 255), 2)
#
#     cv2.imshow('Frame', mask)
#     key = cv2.waitKey(1)
#     if key == ord(' '):
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         plt.imshow(hsv)
#         plt.show()
#     if key == ord('q'):
#         break
