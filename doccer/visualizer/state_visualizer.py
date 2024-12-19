import rerun as rr 
import numpy as np
import omegaconf 

class StateVisualizer:
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig):
        self.cfg = cfg 
        self.fps = cfg.action_fps
        
    def visualize_boundary(self):
        left_top = (self.cfg.boundary.offset, self.cfg.boundary.offset)
        left_bottom = (self.cfg.boundary.offset, self.cfg.screen_size[1] - self.cfg.boundary.offset)
        left_goal_top = (self.cfg.boundary.offset, (self.cfg.screen_size[1] - self.cfg.goal_length) / 2)
        left_goal_bottom = (self.cfg.boundary.offset, (self.cfg.screen_size[1] + self.cfg.goal_length) / 2)
        
        right_top = (self.cfg.screen_size[0] - self.cfg.boundary.offset, self.cfg.boundary.offset)
        right_bottom = (self.cfg.screen_size[0] - self.cfg.boundary.offset, self.cfg.screen_size[1] - self.cfg.boundary.offset)
        right_goal_top = (self.cfg.screen_size[0] - self.cfg.boundary.offset, (self.cfg.screen_size[1] - self.cfg.goal_length) / 2)
        right_goal_bottom = (self.cfg.screen_size[0] - self.cfg.boundary.offset, (self.cfg.screen_size[1] + self.cfg.goal_length) / 2)
        
        rr.log(
            "boundary",
            rr.LineStrips2D(
                strips=[
                    [left_top, right_top],
                    [left_bottom, right_bottom],
                    [left_top, left_goal_top],
                    [left_bottom, left_goal_bottom],
                    [right_top, right_goal_top],
                    [right_bottom, right_goal_bottom],
                ],
                colors=(0, 255, 255),
                radii=self.cfg.boundary.radius,
            ),
            static=True,
        )
        
    def visualize(self, state:np.ndarray):
        self.visualize_boundary()
        for t in range(state.shape[0]):
            rr.set_time_seconds("stable_time", t / self.fps)
            rr.log(
                "agent0", 
                rr.Points2D(
                    positions=state[t, 0:2],
                    colors=(0, 255, 0),
                    radii=self.cfg.agent.radius,
                )
            )
            rr.log(
                "agent1",
                rr.Points2D(
                    positions=state[t, 4:6],
                    colors=(255, 0, 255),
                    radii=self.cfg.agent.radius,
                )
            )
            rr.log(
                "ball",
                rr.Points2D(
                    positions=state[t, 8:10],
                    colors=(255, 255, 255),
                    radii=self.cfg.ball.radius
                )
            )