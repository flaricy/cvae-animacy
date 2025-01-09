import sys 
import pygame 
import pymunk 
import pymunk.pygame_util

def add_ball(space):
    mass = 3
    radius = 25 
    body = pymunk.Body(mass=mass, moment=float('inf'))
    body.position=1000, 500
    shape = pymunk.Circle(body, radius, (0, 0))
    shape.friction = 0
    space.add(body, shape)
    return body, shape
    

def main():
    pygame.init()
    screen = pygame.display.set_mode((1500, 700))
    pygame.display.set_caption("Soccer Game")
    clock = pygame.time.Clock()
    
    space = pymunk.Space()
    space.gravity = (0.0, 0.0)
    space.damping = 0.3
    
    ball_body, ball_shape = add_ball(space)
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    force_length = 1000
    speed_threshold = 200
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            sys.exit(0)
        forces = []
        if keys[pygame.K_w]:
            forces.append((0, -1))
        if keys[pygame.K_s]:
            forces.append((0, 1))
        if keys[pygame.K_a]:
            forces.append((-1, 0))
        if keys[pygame.K_d]:
            forces.append((1, 0))
        sum_force = (sum(force[0] for force in forces), sum(force[1] for force in forces))
        sum_force = pymunk.Vec2d(*sum_force)
        sum_force = sum_force.normalized() * force_length
        ball_body.apply_force_at_local_point(sum_force, (0, 0))
            
        speed = ball_body.velocity.length
        if speed > speed_threshold:
            ball_body.velocity = ball_body.velocity.normalized() * speed_threshold
                        
                
        screen.fill((255, 255, 255))
                
        space.debug_draw(draw_options)
        space.step(1/30.0)
        pygame.display.flip()
        clock.tick(50)
        
if __name__ == "__main__":
    main()