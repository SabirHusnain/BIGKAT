# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pygame
import numpy as np
import pandas as pd
from pygame.locals import *


#        print self.position


def run():

    class target:

        def __init__(self, position, size, color):

            self.init_position = position
            self.init_size = size
            self.init_color = color

            self.position = position
            self.size = size
            self.color = color

            self.surface = pygame.Surface(self.size)
            self.surface.fill(self.color)

        def blit(self, background):
            """Pass the object you want to blit too"""

            background.blit(self.surface, self.position)

        def update_pos(self, new_pos):
            """update the target position"""
            self.position = new_pos

        def reset(self):
            """reset the target to it's original state"""
            self.position = self.init_position
            self.size = self.init_size
            self.color = self.init_color

        def linear_horizontal_displacement(self, t, speed):
            """given time t, and speed move the target"""

            self.position = (self.position[0] + t * speed, self.position[1])

    pygame.init()
    display_info = pygame.display.Info()
    screen = pygame.display.set_mode(
        (display_info.current_w, display_info.current_h), pygame.FULLSCREEN | pygame.DOUBLEBUF)
    pygame.display.set_caption('Oscar Study')
    pygame.mouse.set_visible(1)

    clock = pygame.time.Clock()

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    target = target((20, 200), (100, 50), (0, 0, 0))

    # Display some text
    font = pygame.font.Font(None, 36)
    text1 = font.render("Which target was faster?", 1, (10, 10, 10))
    textpos1 = text1.get_rect()
    textpos1.centerx = background.get_rect().centerx
    textpos1.centery = background.get_rect().centery - 200

    text2 = font.render(
        "Target 1: Press a, Target 2: Press l", 1, (10, 10, 10))
    textpos2 = text2.get_rect()
    textpos2.centerx = background.get_rect().centerx
    textpos2.centery = background.get_rect().centery

    # Where to save response data to
    response_file = open("responses.csv", 'a')
    response_file.write("Response\n")

    N_trial = 20
    pause_time = np.array(
        [[np.random.uniform(0, 3), np.random.uniform(0, 3)] for i in range(N_trial)])
    correct_answer = [1 if i is True else 2 for i in (
        pause_time[:, 0] > pause_time[:, 1])]
    pd.Series(correct_answer).to_csv("correct_answers.csv")

    state_sequence = ['trial', 'trial', 'response'] * pause_time.shape[0]
    pause_time = pause_time.flatten().tolist()  # Make it pop able

    visual_time = 0.0

    game_state = 'load_state'  # Start in the load state

    response = None

    while True:

        visual_dt = clock.tick_busy_loop()  # Should not be faster than the joystick loop
        visual_time += visual_dt / 1000.0
        background.fill((250, 250, 250))

        pressed_keys = pygame.key.get_pressed()

        if game_state == 'trial':

            if visual_time > 2.0:
                target.linear_horizontal_displacement(visual_dt, 2.2)
                target.blit(background)  # Draw the target on the background
            else:
                target.blit(background)

            if visual_time > 5.0:
                game_state = 'load_state'  # After x amount of time enter load state

        elif game_state == 'response':

            background.blit(text1, textpos1)
            background.blit(text2, textpos2)

            if pressed_keys[K_a]:

                response = 1
                response_file.write("{}\n".format(response))

            elif pressed_keys[K_l]:

                response = 2
                response_file.write("{}\n".format(response))

            elif pressed_keys[K_SPACE]:

                response = 3
                response_file.write("{}\n".format(response))

            if response != None:

                game_state = 'load_state'  # Once a respone is issued enter load state

        elif game_state == 'load_state':

            response = None
            target.reset()
            try:
                game_state = state_sequence.pop(0)
                pause = pause_time.pop(0)
            except IndexError:
                pygame.quit()
                response_file.close()
                quit()
            visual_time = 0.0

        if pressed_keys[K_ESCAPE]:
            pygame.quit()
            response_file.close()
            quit()

        screen.blit(background, (0, 0))  # Blit the background to the screen
        pygame.display.flip()
        pygame.event.pump()
