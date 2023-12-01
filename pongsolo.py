import pygame 
from pong import Game 


whidth, heigth = 700, 500
window = pygame.display.set_mode((whidth,heigth))

game = Game(window, whidth, heigth)

run = True
clock = pygame.time.Clock()
while run:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break
    keys= pygame.key.get_pressed()
    if keys[pygame.K_w]:_
        game.move_paddle(left=True, up=True)
    if keys[pygame.K_s]:
        game.move_paddle(left=True,up=False)
        
    game.loop()
    game.draw(False,True)
    pygame.display.update()

pygame.quit()