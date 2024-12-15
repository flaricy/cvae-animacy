STATE_DIM=10
ACTION_DIM=10
LATENT_DIM=64

config=dict(
    dataset=dict(
        path=dict(
            raw_data_path='data/1V1_data_30/processed'
        ),
        sample=dict(
            stride=3,
            length=30,
            interval=1,
            downsample_rate=10
        ),
    )

    model=dict(
        type='ControlVAE',
        conditional_prior=dict(
            type='ConditionalPrior'
            state_dim=STATE_DIM,
            output_dim=[512, 512, LATENT_DIM],
            act='elu',
            std=0.3
        ),
        posterior=dict(
            type='ApproximatePosterior',
            state_dim=STATE_DIM,
            output_dim=[512, 512, LATENT_DIM],
            act='elu',
            std=0.3
        ),
        policy=dict(
            type='PolicyModel',
            num_experts=6,
            expert_network=dict(
                output_dim=[512, 512, 512, ACTION_DIM],
                state_dim=STATE_DIM,
                latent_dim=LATENT_DIM,
                act='elu',
                std=0.05
            ),
            gate_network=dict(
                dim=[STATE_DIM + LATENT_DIM, 64, 64]
            )
        )
        world_model=dict(
            type='WorldModel',
            dim=[STATE_DIM + ACTION_DIM, 512, 512, 512, 512, STATE_DIM],
            act='elu',
        )
    )
)