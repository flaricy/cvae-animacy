import pymunk 
from omegaconf import OmegaConf
import omegaconf
from typing import List
import numpy as np

class Simulator:
    COLLTYPE_AGENT = 1
    COLLTYPE_BALL = 2
    COLLTYPE_BOUNDARY = 3
    
    K_W, K_A, K_S, K_D, K_Z, K_UP, K_LEFT, K_DOWN, K_RIGHT, K_SLASH = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    def __init__(self, cfg:omegaconf.dictconfig.DictConfig):
        self.cfg = cfg
        
    def initialize(self):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = self.cfg.space_damping
        
        self.agents = [self._add_agent(0), self._add_agent(1)]
        self.ball = self._add_ball()
        self.boundary_segments = self._add_boundary()
        
        self.collision_handlers = self._add_collision_handler()
        
    def set_state(self, state:np.ndarray):
        state = Simulator.parse_state(state)
        for agent_id, agent in enumerate(self.agents):
            agent['body'].position = pymunk.Vec2d(*state['agents'][agent_id]['position'])
            agent['body'].velocity = pymunk.Vec2d(*state['agents'][agent_id]['velocity'])
            
        self.ball['body'].position = pymunk.Vec2d(*state['ball']['position'])
        self.ball['body'].velocity = pymunk.Vec2d(*state['ball']['velocity'])
        
    def conduct_action(self, action:List[bool]):
        '''
        [w, a, s, d, kick_0, up, down, left, right, kick_1]
        '''
        joint_force_0, joint_force_1 = self._get_force_on_agents(action)
        self.agents[0]['body'].apply_force_at_local_point(joint_force_0, (0, 0))
        self.agents[1]['body'].apply_force_at_local_point(joint_force_1, (0, 0))
        
        kicked = self._get_kick_on_balls(action)
        for agent_id in range(2):
            if kicked[agent_id]:
                direction = (self.ball['body'].position - self.agents[agent_id]['body'].position).normalized()
                self.ball['body'].apply_impulse_at_local_point(direction * self.cfg.agent.kick_strength, (0, 0))

        self._limit_agent_speed()
        self._limit_agent_position()
        
        for _ in range(self.cfg.simulation_fps // self.cfg.action_fps):
            self.space.step(1/self.cfg.simulation_fps)
            
    def get_state(self):
        return np.array([
            self.agents[0]['body'].position.x, self.agents[0]['body'].position.y,
            self.agents[0]['body'].velocity.x, self.agents[0]['body'].velocity.y,
            self.agents[1]['body'].position.x, self.agents[1]['body'].position.y,
            self.agents[1]['body'].velocity.x, self.agents[1]['body'].velocity.y,
            self.ball['body'].position.x, self.ball['body'].position.y,
            self.ball['body'].velocity.x, self.ball['body'].velocity.y,
        ])
        
    def _add_agent(self, agent_id:int):
        body = pymunk.Body(mass=self.cfg.agent.mass, moment=float('inf'))
        shape = pymunk.Circle(body, radius=self.cfg.agent.radius, offset=(0, 0))
        shape.collision_type = Simulator.COLLTYPE_AGENT
        shape.elasticity = 0
        self.space.add(body, shape)
        return dict(body=body, shape=shape)
    
    def _add_ball(self):
        body = pymunk.Body(mass=self.cfg.ball.mass, moment=float('inf'))
        shape = pymunk.Circle(body, radius=self.cfg.ball.radius, offset=(0, 0))
        shape.collision_type = Simulator.COLLTYPE_BALL
        shape.elasticity = 0.75
        self.space.add(body, shape)
        return dict(body=body, shape=shape)
    
    def _add_boundary(self):
        staticbody = self.space.static_body
        
        left_top = (self.cfg.boundary.offset, self.cfg.boundary.offset)
        left_bottom = (self.cfg.boundary.offset, self.cfg.screen_size[1] - self.cfg.boundary.offset)
        left_goal_top = (self.cfg.boundary.offset, (self.cfg.screen_size[1] - self.cfg.goal_length) / 2)
        left_goal_bottom = (self.cfg.boundary.offset, (self.cfg.screen_size[1] + self.cfg.goal_length) / 2)
        
        right_top = (self.cfg.screen_size[0] - self.cfg.boundary.offset, self.cfg.boundary.offset)
        right_bottom = (self.cfg.screen_size[0] - self.cfg.boundary.offset, self.cfg.screen_size[1] - self.cfg.boundary.offset)
        right_goal_top = (self.cfg.screen_size[0] - self.cfg.boundary.offset, (self.cfg.screen_size[1] - self.cfg.goal_length) / 2)
        right_goal_bottom = (self.cfg.screen_size[0] - self.cfg.boundary.offset, (self.cfg.screen_size[1] + self.cfg.goal_length) / 2)
        
        boundary_segments = [
            pymunk.Segment(staticbody, left_top, right_top, self.cfg.boundary.radius),
            pymunk.Segment(staticbody, left_bottom, right_bottom, self.cfg.boundary.radius),
            pymunk.Segment(staticbody, left_top, left_goal_top, self.cfg.boundary.radius),
            pymunk.Segment(staticbody, left_bottom, left_goal_bottom, self.cfg.boundary.radius),
            pymunk.Segment(staticbody, right_top, right_goal_top, self.cfg.boundary.radius),
            pymunk.Segment(staticbody, right_bottom, right_goal_bottom, self.cfg.boundary.radius),
        ]
        for segment in boundary_segments:
            segment.collision_type = Simulator.COLLTYPE_BOUNDARY
            segment.elasticity = 0.75
            self.space.add(segment)
        return dict(
            body=staticbody,
            shapes=boundary_segments
        )
        
    def _add_collision_handler(self):
        def begin_agent_ball(arbiter, space, data):
            arbiter.elasticity = 0.0
            arbiter.friction = 0.0
            return True
        
        def begin_agent_agent(arbiter, space, data):
            arbiter.elasticity = 0.0
            arbiter.friction = 0.0
            return True
        
        def begin_ball_boundary(arbiter, space, data):
            arbiter.friction = 0.0
            return True
        
        handler_agent_ball = self.space.add_collision_handler(Simulator.COLLTYPE_AGENT, Simulator.COLLTYPE_BALL)
        handler_agent_ball.begin = begin_agent_ball
        
        
        handler_agent_agent = self.space.add_collision_handler(Simulator.COLLTYPE_AGENT, Simulator.COLLTYPE_AGENT)
        handler_agent_agent.begin = begin_agent_agent
        
        handler_ball_boundary = self.space.add_collision_handler(Simulator.COLLTYPE_BALL, Simulator.COLLTYPE_BOUNDARY)
        handler_ball_boundary.begin = begin_ball_boundary
        
        handler_agent_boundary = self.space.add_collision_handler(Simulator.COLLTYPE_AGENT, Simulator.COLLTYPE_BOUNDARY)
        handler_agent_boundary.begin = lambda arbiter, space, data: False
        return [handler_agent_ball, handler_agent_agent, handler_ball_boundary, handler_agent_boundary]

    def parse_state(state:np.ndarray):
        return dict(
            agents=[
                dict(
                    position=state[0:2],
                    velocity=state[2:4]
                ),
                dict(
                    position=state[4:6],
                    velocity=state[6:8]
                )
            ],
            ball=dict(position=state[8:10], velocity=state[10:12])
        )
        
        
    def _get_force_on_agents(self, action:List[bool]):
        joint_force_0 = pymunk.Vec2d(0.0, 0.0)
        if action[Simulator.K_W]:
            joint_force_0 += pymunk.Vec2d(0.0, -1.0)    
        if action[Simulator.K_S]:
            joint_force_0 += pymunk.Vec2d(0.0, 1.0)
        if action[Simulator.K_A]:
            joint_force_0 += pymunk.Vec2d(-1.0, 0.0)
        if action[Simulator.K_D]:
            joint_force_0 += pymunk.Vec2d(1.0, 0.0)
        joint_force_0 = joint_force_0.normalized() * self.cfg.agent.force_length
            
        joint_force_1 = pymunk.Vec2d(0.0, 0.0)
        if action[Simulator.K_UP]:
            joint_force_1 += pymunk.Vec2d(0.0, -1.0)
        if action[Simulator.K_DOWN]:
            joint_force_1 += pymunk.Vec2d(0.0, 1.0)
        if action[Simulator.K_LEFT]:
            joint_force_1 += pymunk.Vec2d(-1.0, 0.0)
        if action[Simulator.K_RIGHT]:
            joint_force_1 += pymunk.Vec2d(1.0, 0.0)
        joint_force_1 = joint_force_1.normalized() * self.cfg.agent.force_length
        
        return joint_force_0, joint_force_1
    
    def _get_kick_on_balls(self, action:List[bool]):
        def is_kick_key_pressed(agent_id):
            if agent_id == 0:
                return action[Simulator.K_Z]
            elif agent_id == 1:
                return action[Simulator.K_SLASH]

        def check_valid_kick(agent_id):
            p1 = self.agents[agent_id]['body'].position
            p2 = self.ball['body'].position 
            dis = (p1 - p2).length 
            return dis <= self.cfg.agent.radius + self.cfg.ball.radius + 2
            
        kicked = [False, False]
        for agent_id, agent in enumerate(self.agents):
            if is_kick_key_pressed(agent_id) and check_valid_kick(agent_id):
                kicked[agent_id] = True 
        return kicked
    
    def _limit_agent_speed(self):
        for agent in self.agents:
            speed = agent['body'].velocity.length 
            if speed > self.cfg.agent.maximum_speed:
                agent['body'].velocity = agent['body'].velocity.normalized() * self.cfg.agent.maximum_speed
                
    def _limit_agent_position(self):
        for agent in self.agents:
            x = max(agent['body'].position.x, 0)
            x = min(x, self.cfg.screen_size[0])
            
            y = max(agent['body'].position.y, 0)
            y = min(y, self.cfg.screen_size[1])
            agent['body'].position = x, y