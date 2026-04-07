class CustomerSupportEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return {"state": "start"}

    def step(self, action):
        self.step_count += 1
        return {
            "reward": 0.5,
            "done": self.step_count >= 3
        }

