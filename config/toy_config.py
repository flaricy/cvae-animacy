STATE_DIM=12
ACTION_DIM=10
LATENT_DIM=64

config=dict(
    dataset=dict(
        path=dict(
            raw_data_path='data/version1_60fps',
        ),
        to_tensor=False,
        sample=dict(
            max_length=128,
            downsample_rate=1,
        )
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
                dim=[STATE_DIM + LATENT_DIM, 64, 64],
                act='elu',
            ),
            
        ),
        world_model=dict(
            type='WorldModel',
            dim=[STATE_DIM + ACTION_DIM, 512, 512, 512, 512, STATE_DIM],
            act='elu',
        ),
    ),
    
    train=dict(
        epochs=10000,
        dynamic_dataset=dict(
            max_num_trajectories=5000,
        ),
        collector=dict(
            num_trajectories=12,
        ),
        update_world_model=dict(
            clip_length=8,
            dataloader=dict(
                batch_size=512,
                shuffle=True,
            ),
            num_updates=8,
        ),
        update_policy_model=dict(
            clip_length=24,
            dataloader=dict(
                batch_size=512,
                shuffle=True
            ),
            num_updates=8,
        ),
        state_loss=dict(
            position_weight=0.5,
            velocity_weight=0.5,
            decay_factor_gamma=0.95,
        ),
        kl_divergence_loss=dict(
            factor_beta=0.01,
            beta_update_step=500,
            beta_update_multiplier=1.122,
            decay_factor_gamma=0.95,
        ),
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