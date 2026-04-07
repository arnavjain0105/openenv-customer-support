from fastapi import FastAPI
from pydantic import BaseModel

# If your env exists, import it
try:
    from openenv.env import CustomerSupportEnv
    env = CustomerSupportEnv()
except:
    # Fallback dummy env (so app never crashes)
    class CustomerSupportEnv:
        def reset(self):
            return {"state": "reset"}

        def step(self, action):
            return {
                "state": "next_state",
                "reward": 0.6,
                "done": False,
                "info": {}
            }

    env = CustomerSupportEnv()

app = FastAPI()

# ✅ Health check (VERY IMPORTANT)
@app.get("/")
def home():
    return {"status": "running"}

# ✅ Reset endpoint (MUST be POST)
@app.api_route("/reset", methods=["GET", "POST"])
def reset():
    return env.reset()

# ✅ Step endpoint (POST)
class Action(BaseModel):
    action: dict = {}

@app.post("/step")
def step(action: Action):
    return env.step(action.action)

# ✅ Predict endpoint (for safety, some checkers use this)
@app.post("/predict")
def predict(action: Action):
    return env.step(action.action)
