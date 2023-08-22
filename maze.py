import pygame
import numpy as np
import sys

pygame.init()

width, height = 400, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Maze Game with RL")

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)

maze = [
    "##########",
    "#        #",
    "# ###E## #",
    "# #      #",
    "# ### ####",
    "#        #",
    "##########"
]

num_states = len(maze) * len(maze[0])
state_mapping = {(x, y): y * len(maze[0]) + x for y, row in enumerate(maze) for x, cell in enumerate(row) if cell != '#'}

num_actions = 4
learning_rate = 0.1
discount_factor = 0.9
exploration_prob = 0.2

Q = np.zeros((num_states, num_actions))

player_x = 1
player_y = 1
player_size = 20


def choose_action(state):
    if np.random.uniform(0, 1) < exploration_prob:
        return np.random.randint(num_actions)
    else:
        return np.argmax(Q[state, :])


def update_Q(state, action, reward, next_state):
    best_next_action = np.argmax(Q[next_state, :])
    Q[state, action] += learning_rate * (reward + discount_factor * Q[next_state, best_next_action] - Q[state, action])


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    current_state = state_mapping[(player_x, player_y)]
    chosen_action = choose_action(current_state)

    new_player_x, new_player_y = player_x, player_y

    if chosen_action == 0 and maze[new_player_y - 1][new_player_x] != "#":
        new_player_y -= 1
    elif chosen_action == 1 and maze[new_player_y + 1][new_player_x] != "#":
        new_player_y += 1
    elif chosen_action == 2 and maze[new_player_y][new_player_x - 1] != "#":
        new_player_x -= 1
    elif chosen_action == 3 and maze[new_player_y][new_player_x + 1] != "#":
        new_player_x += 1

    if maze[new_player_y][new_player_x] == "E":
        reward = 1
    else:
        reward = 0

    new_state = state_mapping[(new_player_x, new_player_y)]
    update_Q(current_state, chosen_action, reward, new_state)

    player_x, player_y = new_player_x, new_player_y

    screen.fill(black)

    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == "#":
                pygame.draw.rect(screen, white, (x * player_size, y * player_size, player_size, player_size))
            elif cell == "E":
                pygame.draw.rect(screen, red, (x * player_size, y * player_size, player_size, player_size))
            elif (x, y) == (player_x, player_y):
                pygame.draw.rect(screen, white, (x * player_size, y * player_size, player_size, player_size))

    pygame.display.update()
