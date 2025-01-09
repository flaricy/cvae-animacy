from sched import scheduler

STATE_DIM = 12
ACTION_DIM = 10
LATENT_DIM = 64

config = dict(
    gen_model=dict(
        type='LSTMModel',
        input_size=STATE_DIM,
        hidden_size=512,
        proj_size=ACTION_DIM,
        dropout=0.1,
    ),

    gen_model_trainer=dict(
        epochs=100,
        dataset=dict(
            path=dict(
                raw_data_path='data/version3_20fps',
            ),
            to_tensor=True,
            sample=dict(
                max_length=40,
                downsample_rate=1,
                scaling=0.001,
            ),
        ),
        dataloader=dict(
            batch_size=128,
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
    ),

    world_model=dict(
        type='TransformerWorldModel',
        hidden_dim=128,
        dim_feedforward=512,
        nhead=4,
        dropout=0.1,
        num_layers=4,
        act='elu',
    ),

    world_model_trainer=dict(
        epochs=1000,
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
        screen_size=(1500, 700),
        space_damping=0.2,
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