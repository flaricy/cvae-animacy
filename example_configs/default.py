from sched import scheduler

STATE_DIM=12
ACTION_DIM=10
LATENT_DIM=64

config=dict(
    gen_model=dict(
        type='ControlVAE',
        conditional_prior=dict(
            type='ConditionalPrior',
            state_dim=STATE_DIM,
            output_dim=[512, 512, LATENT_DIM],
            act='elu',
            std=0.3,
        ),
        posterior=dict(
            type='ApproximatePosterior',
            state_dim=STATE_DIM,
            output_dim=[512, 512, LATENT_DIM],
            act='elu',
            std=0.3,
        ),
        policy=dict(
            type='PolicyModel',
            num_experts=6,
            expert_network=dict(
                output_dim=[512, 512, 512, ACTION_DIM],
                state_dim=STATE_DIM,
                latent_dim=LATENT_DIM,
                act='elu',
                std=0.05,
            ),
            gate_network=dict(
                dim=[STATE_DIM + LATENT_DIM, 64, 64],
                act='elu',
            ),
        ),
    ),

    world_model=dict(
        type='WorldModel',
        dim=[STATE_DIM + ACTION_DIM, 512, 512, 512, 512, STATE_DIM],
        act='elu',
    ),

    world_model_trainer=dict(
        epochs=100,
        dataset=dict(
            path=dict(
                raw_data_path='data/version3_20fps',
            ),
            to_tensor=True,
            sample=dict(
                max_length=2,
                downsample_rate=1,
                scaling=0.001,
            ),
        ),
        dataloader=dict(
            batch_size=256,
            shuffle=True,
        ),
        optimizer=dict(
            type='AdamW',
            lr=1e-5,
            weight_decay=1e-4,
        ),
        scheduler=dict(
            type='StepLR',
            step_size=100000,
            gamma=0.1,
        ),
        loss=dict(
            position_weight=0.5,
            velocity_weight=0.5,
        ),
    ),
    
    device='mps',
    
    simulator=dict(
        screen_size = (1500, 700),
        space_damping = 0.2,
        agent=dict(
            mass=2,
            radius=30,
            force_length=2000,
            maximum_speed=500,
            kick_strength=900,
        ),
        ball=dict(
            mass=1,
            radius=20,
            reach_distance=5,
        ),
        boundary=dict(
            offset=100,
            radius=5,
        ),
        goal_length=200,
        simulation_fps=120,
        action_fps=60,
    )
)