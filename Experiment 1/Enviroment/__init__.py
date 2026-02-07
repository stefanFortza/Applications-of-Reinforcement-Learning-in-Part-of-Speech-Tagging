from gymnasium.envs.registration import register
# default
register(
    id='UniversalPosTagging-v0', 
    entry_point='Enviroment.UniversalPosTaggingEnv:UniversalPosTaggingEnv',
    max_episode_steps=200 
)
#  grammer penalties
register(
    id='UniversalPosTagging-v1', 
    entry_point='Enviroment.UniversalPosTaggingEnv2:UniversalPosTaggingEnv',
    max_episode_steps=200 
)

# Prev Curr Next embedings + grammer penalties
register(
    id='UniversalPosTagging-v2', 
    entry_point='Enviroment.UniversalPosTaggingEnv3:UniversalPosTaggingEnv',
    max_episode_steps=200 
)

# Prev Curr Next embedings + lexicon + grammer penalties
register(
    id='UniversalPosTagging-v3', 
    entry_point='Enviroment.UniversalPosTaggingEnv4:UniversalPosTaggingEnv',
    max_episode_steps=200 
)
# Curr next embeddings + lexicon + grammer penalietes + aggresive
register(
    id='UniversalPosTagging-v4', 
    entry_point='Enviroment.UniversalPosTaggingEnv4:UniversalPosTaggingEnv',
    max_episode_steps=200 
)

# Curr next embed + lexicon + grammer penalies + smoth
register(
    id='UniversalPosTagging-v5', 
    entry_point='Enviroment.UniversalPosTaggingEnv6:UniversalPosTaggingEnv',
    max_episode_steps=200 
)

# Curr next embed + lexicon + grammer penalies + smoth 
register(
    id='UniversalPosTagging-v6', 
    entry_point='Enviroment.UniversalPosTaggingEnv7:UniversalPosTaggingEnv',
    max_episode_steps=200 
)