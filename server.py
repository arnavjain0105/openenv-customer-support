from fastapi import FastAPI

app = FastAPI()

# Dummy environment (fix for openenv issue)
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

env = CustomerSupportEnv()

# API endpoints
@app.get("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: dict):
    return env.step(action)
