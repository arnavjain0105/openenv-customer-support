from fastapi import FastAPI
from openenv.env import CustomerSupportEnv
from openenv.models import Action

app = FastAPI()
env = CustomerSupportEnv()

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/reset")
def reset():
    return env.reset().dict()

@app.post("/step")
def step(action: dict):
    action_obj = Action(**action)
    obs, reward, done, info = env.step(action_obj)

    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()
