from fastapi import FastAPI
from pydantic import BaseModel

print("✅ NEW VERSION DEPLOYED")

app = FastAPI()

# Dummy env (safe fallback)
class DummyEnv:
    def reset(self):
        return {"state": "reset"}

    def step(self, action):
        return {"state": "next", "reward": 1, "done": False}

env = DummyEnv()

# ✅ Root endpoint (fix 404)
@app.get("/")
def home():
    return {"status": "running"}

# ✅ MUST be POST (fix your error)
@app.post("/reset")
def reset():
    return env.reset()

# ✅ Step endpoint
class Action(BaseModel):
    action: dict = {}

@app.post("/step")
def step(action: Action):
    return env.step(action.action)

# ✅ Extra safety endpoint
@app.post("/predict")
def predict(action: Action):
    return env.step(action.action)
