from gymnasium.envs.registration import register
register(
    id='AddictiveEnv_Avanzato',
    entry_point='AddictiveReward.envs:AddictiveEnv_Avanzato',
)
register(
    id='AddictiveEnv_Semplificato',
    entry_point='AddictiveReward.envs:AddictiveEnv_Semplificato',
)
register(
    id='AddictiveEnv_Raffinato',
    entry_point='AddictiveReward.envs:AddictiveEnv_Raffinato',
)