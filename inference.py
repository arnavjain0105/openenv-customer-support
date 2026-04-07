import asyncio
import os
import textwrap
from typing import List, Optional

import requests
from openai import OpenAI

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN")

if not API_KEY:
    raise ValueError("HF_TOKEN is not set in environment variables")


ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://127.0.0.1:8000"
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or "https://router.huggingface.co/v1"

MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "customer_support")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "openenv_customer_support")

MAX_STEPS = 3
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5

_MAX_REWARD_PER_STEP = 1.0
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a customer support AI agent.
    Your task:
    - Classify the email (billing, tech, refund)
    - Write a helpful response
    - Decide if escalation is needed

    Output format:
    category: <category>
    response: <text>
    escalate: <true/false>
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, email_subject: str, email_body: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Email Subject: {email_subject}
        Email Body: {email_body}

        Previous steps:
        {history_block}

        Provide classification, response, and escalation decision.
        """
    ).strip()


def parse_model_output(text: str):
    text_lower = text.lower()

    category = "tech"
    if "billing" in text_lower:
        category = "billing"
    elif "refund" in text_lower:
        category = "refund"

    escalate = "true" in text_lower or "escalate" in text_lower

    return category, text.strip(), escalate


def get_model_message(client: OpenAI, step: int, email_subject: str, email_body: str, history: List[str]):
    user_prompt = build_user_prompt(step, email_subject, email_body, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_model_output(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "tech", "Default response", False


async def main() -> None:
    client = OpenAI(base_url=LLM_BASE_URL,api_key=API_KEY)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = requests.post(f"{ENV_BASE_URL}/reset").json()

        email = result["current_email"]
        email_subject = email["subject"]
        email_body = email["body"]

        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            category, response_text, escalate = get_model_message(
                client, step, email_subject, email_body, history
            )

            action = {
                "category": category,
                "response": response_text,
                "escalate": escalate
            }

            result = requests.post(f"{ENV_BASE_URL}/step", json=action).json()

            reward = result["reward"]["score"]
            done = result["done"]
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=category, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {category} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
