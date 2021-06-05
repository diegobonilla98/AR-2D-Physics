import utils
from utils import cv2, Box2D
import pygame
import matplotlib.pyplot as plt
import math
from utils import np
from MiDaS import depth_utils


def convertScreen2World(x, y):
    x = screenSize.x - x
    return Box2D.b2Vec2((x + viewOffset.x) / viewZoom, ((screenSize.y - y + viewOffset.y) / viewZoom))


def convertWorld2Screen(point):
    x = (point.x * viewZoom) - viewOffset.x
    y = (point.y * viewZoom) - viewOffset.y
    y = screenSize.y - y
    return int(round(x)), int(round(y))


def get_box_angle(box):
    vec = box[1] - box[0]
    angle = math.atan2(vec[1], vec[0])
    return math.degrees(angle)


depth_utils.load_model()

video = cv2.VideoCapture(1)
TARGET_FPS = 20
TIME_STEP = 1.0 / TARGET_FPS
width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
screenSize = Box2D.b2Vec2(width, height)
viewOffset = Box2D.b2Vec2(0, 0.0)
viewZoom = 1.0

pygame.init()
screen = pygame.display.set_mode((width, height), 0, 32)
pygame.display.set_caption('AR 2D Physics')
clock = pygame.time.Clock()

world = Box2D.b2World(gravity=(-0.0, -1000.0), doSleep=True)

polygon = Box2D.b2PolygonShape()
polygon.vertices = [(-740.0, -0.5), (740.0, -0.5), (740.0, 0.5), (-740.0, 0.5)]
mega_rect = world.CreateStaticBody(
    position=(0, 0),
    shapes=polygon
)

ground_body = world.CreateStaticBody(
    position=(0, 0),
    shapes=Box2D.b2PolygonShape(box=(width, 0.5))
)

balls_radius = 10
balls_colors = []
balls = []
num_balls = 20
for pos in np.linspace(0, width, num_balls):
    utils.add_ball(world, balls, (pos, height / 1.1), (0, 0), radius=balls_radius)
    balls_colors.append(tuple(np.random.randint(0, 255, size=(3,))))

Box2D.b2_maxPolygonVertices = 100

not_stop = True
while not_stop:
    ret, frame = video.read()
    execcccccc = frame[:, :, ::-1]
    world.Step(TIME_STEP, 8, 3)

    world.DestroyBody(mega_rect)
    depth = depth_utils.predict_depth(execcccccc)
    _, mask = cv2.threshold(depth, np.max(depth) - 55, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=lambda c: cv2.contourArea(c))
    hull = cv2.convexHull(contour).reshape((-1, 2))
    scale_down = np.ceil(hull.shape[0] / 16)
    new_hull = []
    for i, h in enumerate(hull):
        if i % scale_down == 0:
            new_hull.append(h)
    cv2.drawContours(frame, [np.array(new_hull)[:, np.newaxis, :]], 0, (0, 0, 255), 2)

    frame = frame[:, :, ::-1]
    world_coords = list(map(lambda x: convertScreen2World(x[0], x[1]), new_hull))
    polygon = Box2D.b2PolygonShape()
    polygon.vertices = world_coords
    mega_rect = world.CreateStaticBody(
        shapes=polygon
    )

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    screen.fill((0, 0, 0, 0))
    surf = pygame.surfarray.make_surface(frame)
    screen.blit(surf, (0, 0))

    for ball, color in zip(balls, balls_colors):
        pos = convertWorld2Screen(ball.position)
        pygame.draw.circle(screen, color, pos, balls_radius)

    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            not_stop = False
    clock.tick(TARGET_FPS)

pygame.quit()
