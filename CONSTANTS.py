# File in which all constant are defined (it's way proper like that 10/11/2018)

# REDLOCK PARAMS
RETRY_COUNT = 1
RETRY_DELAY = 0.001
LOCK_LIFE_LONG = 10000  # 10 secs I think
NAME_REDLOCK = "currently_updating_or_finding_priorities"
# REDLOCK PARAMS

# REDIS STRING CONSTANT
PRIORITIES_STR = "priorities:"
INDEX_ACTOR_STR = "index_actor:"
STEP_ACTOR_STR = "step_actor:"
STEP_LEARNER_STR = "step_learner:"
IS_FULL_ACTOR_STR = "is_full_actor:"
MAX_PRIORITY_STR = "max_priority"
TRANSITIONS_STR = "transitions"
MODEL_WEIGHT_STR = "model_weight"
# REDIS STRING CONSTANT

hack_set_full_capacity_to_true = 200000  # This is just a hack, really
# not important to understand (the computational gain is probably insignificant)
# Number of steps between HACK of set memory_full to true,
# the redis memory is passed by copy in another thread and thus his memory_full
# parameter is never updated
TIME_TO_SLEEP = 2  # Time to sleep when learner is behind
