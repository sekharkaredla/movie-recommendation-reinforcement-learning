class Train:
    def __init__(self):
        self.actor = None
        self.critic = None
        self.actor_target = None
        self.critic_target = None
        self.user_states = {}
