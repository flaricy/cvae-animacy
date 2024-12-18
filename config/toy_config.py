STATE_DIM=6
ACTION_DIM=10
LATENT_DIM=64

config=dict(
    dataset=dict(
        path=dict(
            raw_data_path='data/1V1_data_30/processed',
        ),
        to_tensor=True,
    ),

    model=dict(
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
                dim=[STATE_DIM + LATENT_DIM, 64, 64]
            ),
        ),
        world_model=dict(
            type='WorldModel',
            dim=[STATE_DIM + ACTION_DIM, 512, 512, 512, 512, STATE_DIM],
            act='elu',
        ),
    ),
    
    train=dict(
        gt_dataloader=dict(
            batch_size=1,
            num_workers=8,
        ),
        epochs=10000,
        optimizer=dict(
            type='Adam',
            lr=1e-4,
            weight_decay=1e-5,
        ),
        scheduler=dict(
            type='StepLR',
            step_size=5000,
            gamma=0.1,
        ),
    ),
    
    device='mps',
    
    simulator=dict(
        screen_size = (1500, 700),
        space_damping = 0.5,
        agent=dict(
            mass=3,
            radius=30,
            force_length=2000,
            maximum_speed=200,
            kick_strength=700,
        ),
        ball=dict(
            mass=1,
            radius=20,
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