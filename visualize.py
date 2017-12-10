import numpy as np
import pandas as pd
import pygame
import glob
import time

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

pred_path = "./test-predictions-epoch44"
true_path = "data/test/CH2_final_evaluation.csv"
img_path = "data/test/center/*.jpg"

preds = pd.read_csv(pred_path)
true = pd.read_csv(true_path)
filenames = glob.glob(img_path)

preds['path'] = preds['frame_id']
preds['frame_id'] = preds['frame_id'].map(lambda x: x.split('/')[-1].split('.')[0])

true.columns = ['frame_id', 'true_steering_angle', 'public']
true['frame_id'] = true['frame_id'].map(lambda x: str(x))
assert(len(preds) == len(true))

preds_and_true = preds.merge(true, left_on='frame_id', right_on='frame_id')
preds_and_true.head()
preds_and_true.sort_values(by='frame_id')

pygame.init()
size = (640, 320)
pygame.display.set_caption("Data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
myfont = pygame.font.SysFont("monospace", 15)

# for i in range(1000):
total = float(len(filenames))
total_diff = 0.0
for i in range(len(filenames)):
    angle = preds_and_true["steering_angle"].iloc[i] # radians
    true_angle = preds_and_true["true_steering_angle"].iloc[i] # radians

    # add image to screen
    img = pygame.image.load(preds_and_true["path"].iloc[i])
    screen.blit(img, (0, 0))

    # add text
    pred_txt = myfont.render("Prediction:" + str(round(angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees
    true_txt = myfont.render("True angle:" + str(round(true_angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees

    diff = abs(angle - true_angle)
    total_diff += diff / total
    print(diff)

    screen.blit(pred_txt, (10, 280))
    screen.blit(true_txt, (10, 300))

    # draw steering wheel
    radius = 50
    pygame.draw.circle(screen, WHITE, [320, 300], radius, 2)

    # draw cricle for true angle
    x = radius * np.cos(np.pi/2 + true_angle)
    y = radius * np.sin(np.pi/2 + true_angle)
    pygame.draw.circle(screen, WHITE, [320 + int(x), 300 - int(y)], 7)

    # draw cricle for predicted angle
    x = radius * np.cos(np.pi/2 + angle)
    y = radius * np.sin(np.pi/2 + angle)
    pygame.draw.circle(screen, BLACK, [320 + int(x), 300 - int(y)], 5)

    time.sleep(.3)
    #pygame.display.update()
    pygame.display.flip()

print ("TOTAL DIFF: {}".format(total_diff))
