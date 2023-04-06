# -*- coding: utf-8 -*-
"""
Using pygame play a simple graphical clock as fast as possible. Used to test the synchrony of the two raspberry Pi cameras
"""

import pygame, os

pygame.init()


width, height = 500,500

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Digital Counter")

background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill((255,255,255))

font = pygame.font.Font(None, 80)

t = 0
text = font.render("{}".format(t), 1,(50,50,50))
textpos = text.get_rect()
textpos.centerx = background.get_rect().centerx
textpos.centery = background.get_rect().centery

while True:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
    background.fill((255,255,255))
        
    background.blit(text, textpos)
    
    screen.blit(background, (0,0))
    pygame.display.flip()    
    
    t+=1
    text = font.render("{}".format(t), 1,(10,10,10))
    
