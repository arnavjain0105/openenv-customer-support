from fastapi import FastAPI
from pydantic import BaseModel

print("🔥 APP.PY IS RUNNING 🔥")

app = FastAPI()

class DummyEnv:
    def reset(self):
        return {"state": "reset"}

    def step(self, action):
        return {"state": "next", "reward": 1, "done": False}

env = DummyEnv()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/reset")
def reset():
    return env.reset()

class Action(BaseModel):
    action: dict = {}

@app.post("/step")
def step(action: Action):
    return env.step(action.action)

@app.post("/predict")
def predict(action: Action):
    return env.step(action.action)
