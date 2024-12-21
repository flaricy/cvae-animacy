import numpy as np
import omegaconf 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

class StateVisualizer:
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig, scaling_factor:float=1.0):
        self.cfg = cfg 
        self.fps = cfg.action_fps
        self.scaling_factor=scaling_factor
        self.fig, self.axs = plt.subplots(1, 2)
        for i in range(0, 2):
            self.axs[i].set_xlim(0, cfg.screen_size[0])
            self.axs[i].set_ylim(0, cfg.screen_size[1])
            self.axs[i].set_aspect('equal')
            self.axs[i].axis('off')
            
        self.axs[0].set_title("generated")
        self.axs[1].set_title("ground truth")
        
        self.plt_data = [0, 0]
        
        for i in range(2):
            player1 = Circle((0, 0), radius=self.cfg.agent.radius, color='red')
            player2 = Circle((0, 0), radius=self.cfg.agent.radius, color='blue')
            ball = Circle((0, 0), radius=self.cfg.ball.radius, color='green')
            self.plt_data[i] = dict(
                player1=player1,
                player2=player2,
                ball=ball
            )
            for value in self.plt_data[i].values():
                self.axs[i].add_patch(value)
                
        self.visualize_boundary()
        
        
    def visualize_boundary(self):
        left_top = (self.cfg.boundary.offset, self.cfg.boundary.offset)
        left_bottom = (self.cfg.boundary.offset, self.cfg.screen_size[1] - self.cfg.boundary.offset)
        left_goal_top = (self.cfg.boundary.offset, (self.cfg.screen_size[1] - self.cfg.goal_length) / 2)
        left_goal_bottom = (self.cfg.boundary.offset, (self.cfg.screen_size[1] + self.cfg.goal_length) / 2)
        
        right_top = (self.cfg.screen_size[0] - self.cfg.boundary.offset, self.cfg.boundary.offset)
        right_bottom = (self.cfg.screen_size[0] - self.cfg.boundary.offset, self.cfg.screen_size[1] - self.cfg.boundary.offset)
        right_goal_top = (self.cfg.screen_size[0] - self.cfg.boundary.offset, (self.cfg.screen_size[1] - self.cfg.goal_length) / 2)
        right_goal_bottom = (self.cfg.screen_size[0] - self.cfg.boundary.offset, (self.cfg.screen_size[1] + self.cfg.goal_length) / 2)
        
        upper_part = np.array([left_goal_top, left_top, right_top, right_goal_top]) # (num_points, 2)
        lower_part = np.array([left_goal_bottom, left_bottom, right_bottom, right_goal_bottom]) # (num_points, 2)
        for i in range(0, 2):
            self.axs[i].plot(upper_part[:, 0], upper_part[:, 1])
            self.axs[i].plot(lower_part[:, 0], lower_part[:, 1])
        
    def visualize(self, generated_state:np.ndarray, gt_state:np.ndarray):
        # state: (num_frames, 12)
        self.visualize_boundary()
        generated_state = generated_state / self.scaling_factor
        gt_state = gt_state / self.scaling_factor 
            
        def update(frame):
            def update_each_state(plt_data, state):
                plt_data['player1'].set_center(state[frame, [0, 1]])
                plt_data['player2'].set_center(state[frame, [4, 5]])
                plt_data['ball'].set_center(state[frame, [8, 9]])
                
            update_each_state(self.plt_data[0], generated_state)
            update_each_state(self.plt_data[1], gt_state)
            
        ani = FuncAnimation(self.fig, func=update, frames=generated_state.shape[0])
        return ani
        