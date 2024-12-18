import sys
sys.path.append('doccer')

import pygame 
import numpy as np
import pickle
from omegaconf import OmegaConf
from engines.simulator import Simulator 
from utils.parse_config import load_config
from data_collect.file_helper import FileHelper 

visualize_config = dict(
    screen_size=(1500, 700),
    court_color=(28, 28, 30),
    agent=dict(
        radius=30,
        agent0_color=(0, 255, 0),
        agent1_color=(255, 0, 255)
    ),
    ball=dict(
        radius=20,
        color=(255, 255, 255),
    ),
    boundary=dict(
        offset=100,
        radius=5,
        color=(0, 255, 255),
    ),
    goal_length=200,
    rendering_fps=60,
)
visualize_config = OmegaConf.create(visualize_config)
similator_config = OmegaConf.create(load_config('config/toy_config.py'))
similator_config = similator_config.simulator

def get_initial_state():
    return np.array([
        visualize_config.screen_size[0] * 0.25, visualize_config.screen_size[1] / 2,
        0, 0,
        visualize_config.screen_size[0] * 0.75, visualize_config.screen_size[1] / 2,
        0, 0,
        visualize_config.screen_size[0] / 2, visualize_config.screen_size[1] / 2,
        0, 0,
    ])
    
def get_action(keys):
    return np.array([
        keys[pygame.K_w],
        keys[pygame.K_a],
        keys[pygame.K_s],
        keys[pygame.K_d],
        keys[pygame.K_b],
        keys[pygame.K_UP],
        keys[pygame.K_LEFT],
        keys[pygame.K_DOWN],
        keys[pygame.K_RIGHT],
        keys[pygame.K_SLASH],
    ], dtype=bool)
    
def get_boundary_segments():
    left_top = (visualize_config.boundary.offset, visualize_config.boundary.offset)
    left_bottom = (visualize_config.boundary.offset, visualize_config.screen_size[1] - visualize_config.boundary.offset)
    left_goal_top = (visualize_config.boundary.offset, (visualize_config.screen_size[1] - visualize_config.goal_length) / 2)
    left_goal_bottom = (visualize_config.boundary.offset, (visualize_config.screen_size[1] + visualize_config.goal_length) / 2)
    
    right_top = (visualize_config.screen_size[0] - visualize_config.boundary.offset, visualize_config.boundary.offset)
    right_bottom = (visualize_config.screen_size[0] - visualize_config.boundary.offset, visualize_config.screen_size[1] - visualize_config.boundary.offset)
    right_goal_top = (visualize_config.screen_size[0] - visualize_config.boundary.offset, (visualize_config.screen_size[1] - visualize_config.goal_length) / 2)
    right_goal_bottom = (visualize_config.screen_size[0] - visualize_config.boundary.offset, (visualize_config.screen_size[1] + visualize_config.goal_length) / 2)
    
    return [
        (left_top, right_top),
        (left_bottom, right_bottom),
        (left_top, left_goal_top),
        (left_bottom, left_goal_bottom),
        (right_top, right_goal_top),
        (right_bottom, right_goal_bottom),
    ]
    
def draw(state, static_boundary_segments, screen):
    pygame.draw.circle(surface=screen, color=visualize_config.agent.agent0_color, center=(int(state[0]), int(state[1])), radius=visualize_config.agent.radius)
    pygame.draw.circle(surface=screen, color=visualize_config.agent.agent1_color, center=(int(state[4]), int(state[5])), radius=visualize_config.agent.radius)
    pygame.draw.circle(surface=screen, color=visualize_config.ball.color, center=(int(state[8]), int(state[9])), radius=visualize_config.ball.radius)
    
    for segment in static_boundary_segments:
        pygame.draw.line(surface=screen, color=visualize_config.boundary.color, start_pos=segment[0], end_pos=segment[1], width=visualize_config.boundary.radius)

def main():
    
    data_log = dict(
        state=np.zeros((100000, 12)),
        action=np.zeros((100000, 10), dtype=bool),
    )    
    
    pygame.init()
    screen = pygame.display.set_mode(visualize_config.screen_size)
    pygame.display.set_caption("Soccer Game")
    clock = pygame.time.Clock()
    
    static_boundary_segments = get_boundary_segments()
    
    simulator = Simulator(similator_config)
    simulator.initialize()
    simulator.set_state(get_initial_state())
    
    fps_font = pygame.font.SysFont('Arial', 15)
    
    started = False
    counter = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
                
        keys = pygame.key.get_pressed()
        action = get_action(keys)
        if not started and np.any(action):
                started = True
        action = simulator.action_correction(action)
        
        result = simulator.conduct_action(action)
        if result is not None:
            break
        
        state = simulator.get_state()
        if started:
            data_log['state'][counter] = state 
            data_log['action'][counter] = action 
            counter += 1
        screen.fill(visualize_config.court_color)
        fps = clock.get_fps()
        fps_text = fps_font.render(f"FPS: {fps:.2f}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))
        draw(state, static_boundary_segments, screen)
        pygame.display.flip()
        clock.tick(visualize_config.rendering_fps)
        
    print(f"Game over! Winner: {result}")
    
    for key in data_log.keys():
        data_log[key] = data_log[key][:counter]
    
    file_helper = FileHelper(data_dir='data')
    with open(file_helper.get_cur_file_path(), 'wb') as f:
        pickle.dump(data_log, f)
        
if __name__ == "__main__":
    main()