from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MyEnvV4Action:
    message: str


@dataclass
class MyEnvV4Observation:
    echoed_message: str


@dataclass
class MyEnvV4StepResult:
    observation: MyEnvV4Observation
    reward: float
    done: bool


class MyEnvV4Env:
    """Local compatibility env used by inference.py.

    This implements the small async API surface expected by the runner:
    - from_docker_image(image_name)
    - reset()
    - step(MyEnvV4Action)
    - close()
    """

    def __init__(self, image_name: str | None = None, max_steps: int = 8) -> None:
        self._image_name = image_name
        self._max_steps = max_steps
        self._step_index = 0
        self._done = False

    @classmethod
    async def from_docker_image(cls, image_name: str | None) -> "MyEnvV4Env":
        return cls(image_name=image_name)

    async def reset(self) -> MyEnvV4StepResult:
        self._step_index = 0
        self._done = False
        return MyEnvV4StepResult(
            observation=MyEnvV4Observation(echoed_message=""),
            reward=0.0,
            done=False,
        )

    async def step(self, action: MyEnvV4Action) -> MyEnvV4StepResult:
        if self._done:
            return MyEnvV4StepResult(
                observation=MyEnvV4Observation(echoed_message=action.message),
                reward=0.0,
                done=True,
            )

        self._step_index += 1
        message = action.message or ""
        reward = float(len(message) * 0.1)
        self._done = self._step_index >= self._max_steps

        return MyEnvV4StepResult(
            observation=MyEnvV4Observation(echoed_message=message),
            reward=reward,
            done=self._done,
        )

    async def close(self) -> None:
        return None
