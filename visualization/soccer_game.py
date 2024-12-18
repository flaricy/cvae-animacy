import sys
import pygame 
import pymunk 
import pymunk.pygame_util
from omegaconf import OmegaConf
import numpy as np

config = dict(
    screen_size = (1500, 700),
    space_damping = 0.3,
    agent=dict(
        mass=3,
        radius=30,
        force_length=2000,
        maximum_speed=200,
        agent0_color=(255, 0, 0),
        agent1_color=(0, 0, 255),
        kick_strength=700,
    ),
    ball=dict(
        mass=1,
        radius=20,
        color=(0, 255, 0)
    ),
    boundary=dict(
        offset=100,
        radius=5,
    ),
    fps=60,
)
config = OmegaConf.create(config)

COLLTYPE_AGENT = 1
COLLTYPE_BALL = 2
COLLTYPE_BOUNDARY = 3

def parse_force_key(keys):
    joint_force_0 = pymunk.Vec2d(0.0, 0.0)
    if keys[pygame.K_w]:
        joint_force_0 += pymunk.Vec2d(0.0, -1.0)    
    if keys[pygame.K_s]:
        joint_force_0 += pymunk.Vec2d(0.0, 1.0)
    if keys[pygame.K_a]:
        joint_force_0 += pymunk.Vec2d(-1.0, 0.0)
    if keys[pygame.K_d]:
        joint_force_0 += pymunk.Vec2d(1.0, 0.0)
    joint_force_0 = joint_force_0.normalized() * config.agent.force_length
        
    joint_force_1 = pymunk.Vec2d(0.0, 0.0)
    if keys[pygame.K_UP]:
        joint_force_1 += pymunk.Vec2d(0.0, -1.0)
    if keys[pygame.K_DOWN]:
        joint_force_1 += pymunk.Vec2d(0.0, 1.0)
    if keys[pygame.K_LEFT]:
        joint_force_1 += pymunk.Vec2d(-1.0, 0.0)
    if keys[pygame.K_RIGHT]:
        joint_force_1 += pymunk.Vec2d(1.0, 0.0)
    joint_force_1 = joint_force_1.normalized() * config.agent.force_length
    return joint_force_0, joint_force_1

def parse_kick_key(keys, agents, ball):
    def is_kick_key_pressed(agent_id):
        if agent_id == 0:
            return keys[pygame.K_z]
        elif agent_id == 1:
            return keys[pygame.K_SLASH]
        
    def check_valid_kick(agent_id):
        p1 = agents[agent_id]['body'].position
        p2 = ball['body'].position 
        dis = (p1 - p2).length 
        return dis <= config.agent.radius + config.ball.radius + 2
        
    kicked = [False, False]
    for agent_id, agent in enumerate(agents):
        if is_kick_key_pressed(agent_id) == False:
            agent['kicking'] = False 
        else:
            agent['kicking'] = True 
            if check_valid_kick(agent_id):
                kicked[agent_id] = True 
                agent['kicking'] = False
    return kicked
            
    

def add_agent(space, agent_id):
    body = pymunk.Body(mass=config.agent.mass, moment=pymunk.moment_for_circle(config.agent.mass, 0, config.agent.radius))
    if agent_id == 0:
        body.position = config.screen_size[0] * 0.25, config.screen_size[1] / 2
    else:
        body.position = config.screen_size[0] * 0.75, config.screen_size[1] / 2
    
    shape = pymunk.Circle(body, radius=config.agent.radius, offset=(0, 0))
    shape.friction = 0
    shape.collision_type = COLLTYPE_AGENT
    space.add(body, shape)
    return dict(body=body, shape=shape, kicking=False)

def add_ball(space):
    body = pymunk.Body(mass=config.ball.mass, moment=pymunk.moment_for_circle(config.ball.mass, 0, config.ball.radius))
    body.position = config.screen_size[0] / 2, config.screen_size[1] / 2

    shape = pymunk.Circle(body, radius=config.ball.radius, offset=(0, 0))
    shape.friction = 0
    shape.collision_type = COLLTYPE_BALL
    shape.elasticity = 0.9
    space.add(body, shape)
    return dict(body=body, shape=shape)

def add_boundary(space):
    staticbody = space.static_body 
    left_top = (config.boundary.offset, config.boundary.offset)
    right_top = (config.screen_size[0] - config.boundary.offset, config.boundary.offset)
    left_bottom = (config.boundary.offset, config.screen_size[1] - config.boundary.offset)
    right_bottom = (config.screen_size[0] - config.boundary.offset, config.screen_size[1] - config.boundary.offset)
    
    boundary_segments = [
        pymunk.Segment(staticbody, left_top, right_top, config.boundary.radius),
        pymunk.Segment(staticbody, right_top, right_bottom, config.boundary.radius),
        pymunk.Segment(staticbody, right_bottom, left_bottom, config.boundary.radius),
        pymunk.Segment(staticbody, left_bottom, left_top, config.boundary.radius),
    ]
    for segment in boundary_segments:
        segment.friction = 0
        segment.elasticity = 0.9
        segment.collision_type = COLLTYPE_BOUNDARY
        space.add(segment)
    return dict(
        body=staticbody,
        shapes=boundary_segments
    )
    

def add_collision_handler(space):
    handler_agent_ball = space.add_collision_handler(COLLTYPE_AGENT, COLLTYPE_BALL)
    handler_agent_ball.elasticity = 0.0
    handler_agent_ball.friction = 0.0
    
    handler_agent_agent = space.add_collision_handler(COLLTYPE_AGENT, COLLTYPE_AGENT)
    handler_agent_agent.elasticity = 0.0
    handler_agent_agent.friction = 0.0
    
    handler_ball_boundary = space.add_collision_handler(COLLTYPE_BALL, COLLTYPE_BOUNDARY)
    handler_ball_boundary.elasticity = 0.75
    handler_ball_boundary.friction = 0
    
    handler_agent_boundary = space.add_collision_handler(COLLTYPE_AGENT, COLLTYPE_BOUNDARY)
    handler_agent_boundary.begin = lambda arbiter, space, data: False
    return [handler_agent_ball, handler_agent_agent, handler_ball_boundary, handler_agent_boundary]

def draw(agents, ball, segments, screen):
    def draw_body(body, color, radius):
        pygame.draw.circle(surface=screen, color=color, center=(int(body.position.x), int(body.position.y)), radius=radius)
    draw_body(agents[0]['body'], config.agent.agent0_color, config.agent.radius)
    draw_body(agents[1]['body'], config.agent.agent1_color, config.agent.radius)
    draw_body(ball['body'], config.ball.color, config.ball.radius)
    for segment in segments['shapes']:
        pygame.draw.line(screen, (0, 0, 0), segment.a, segment.b, config.boundary.radius)

def main():
    pygame.init()
    screen = pygame.display.set_mode(config.screen_size)
    pygame.display.set_caption("Soccer Game")
    clock = pygame.time.Clock()
    # draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    space.damping = config.space_damping
    
    agents = [add_agent(space, 0), add_agent(space, 1)]
    soccer_ball = add_ball(space)
    segments = add_boundary(space)
    
    handlers = add_collision_handler(space)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
        
        keys = pygame.key.get_pressed()    
        joint_force_0, joint_force_1 = parse_force_key(keys)
        agents[0]['body'].apply_force_at_local_point(joint_force_0, (0, 0))
        agents[1]['body'].apply_force_at_local_point(joint_force_1, (0, 0))
        
        kicked = parse_kick_key(keys, agents, soccer_ball)
        for agent_id in range(2):
            if kicked[agent_id]:
                direction = (soccer_ball['body'].position - agents[agent_id]['body'].position).normalized()
                soccer_ball['body'].apply_impulse_at_local_point(direction * config.agent.kick_strength, (0, 0))
        
        # region: limit speed
        for agent in agents:
            speed = agent['body'].velocity.length 
            if speed > config.agent.maximum_speed:
                agent['body'].velocity = agent['body'].velocity.normalized() * config.agent.maximum_speed
        # endregion
        
        # region: limit position 
        for agent in agents:
            x = max(agent['body'].position.x, 0)
            x = min(x, config.screen_size[0])
            
            y = max(agent['body'].position.y, 0)
            y = min(y, config.screen_size[1])
            agent['body'].position = x, y
        # endregion
        
        screen.fill((255, 255, 255))
        draw(agents, soccer_ball, segments,screen)
        # space.debug_draw(draw_options)
        space.step(1/config.fps)
        pygame.display.flip()
        clock.tick(config.fps)
        
if __name__ == "__main__":
    main()