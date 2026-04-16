"""gRPC client — runs the sim, sends observations to server, receives actions.

Usage:
    uv run python -m serving.grpc_client
    uv run python -m serving.grpc_client --server localhost:50051 --episodes 5
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
import uuid
from pathlib import Path

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import grpc
import imageio
import numpy as np
from PIL import Image

from proto import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)


class InferenceClient:

    def __init__(self, server_addr: str = "localhost:50051") -> None:
        self._channel = grpc.insecure_channel(
            server_addr,
            options=[
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ],
        )
        self._stub = inference_pb2_grpc.InferenceServiceStub(self._channel)
        logger.info("Connected to server at %s", server_addr)

    def load_model(self, model_id: str, pretrained_path: str, gpu_id: int = -1) -> dict:
        resp = self._stub.LoadModel(inference_pb2.LoadModelRequest(
            model_id=model_id, pretrained_path=pretrained_path, gpu_id=gpu_id,
        ))
        return {"success": resp.success, "message": resp.message, "gpu_id": resp.gpu_id, "memory_mb": resp.memory_used_mb}

    def unload_model(self, model_id: str) -> dict:
        resp = self._stub.UnloadModel(inference_pb2.UnloadModelRequest(model_id=model_id))
        return {"success": resp.success, "message": resp.message}

    def list_models(self) -> list[dict]:
        resp = self._stub.ListModels(inference_pb2.ListModelsRequest())
        return [{"model_id": m.model_id, "path": m.pretrained_path, "gpu": m.gpu_id, "requests": m.total_requests} for m in resp.models]

    def status(self) -> dict:
        resp = self._stub.GetStatus(inference_pb2.StatusRequest())
        return {
            "gpus": [{"id": g.gpu_id, "name": g.name, "used_mb": g.used_memory_mb, "total_mb": g.total_memory_mb, "models": list(g.loaded_models)} for g in resp.gpus],
            "total_models": resp.total_models,
            "total_requests": resp.total_requests_served,
            "uptime_s": resp.uptime_seconds,
        }

    def new_episode_id(self) -> str:
        return str(uuid.uuid4())

    def predict(
        self,
        model_id: str,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        priority: int = 1,
        episode_id: str | None = None,
    ) -> tuple[np.ndarray, float]:
        """Send observation to server, return (actions, inference_time_ms)."""
        ep_id = episode_id or str(uuid.uuid4())
        step_id = uuid.uuid4().hex[:8]

        image_frames = []
        for cam_name, img_arr in images.items():
            jpeg_bytes = self._encode_jpeg(img_arr)
            image_frames.append(inference_pb2.ImageFrame(
                camera_name=cam_name, data=jpeg_bytes,
                width=img_arr.shape[1], height=img_arr.shape[0], encoding="jpeg",
            ))

        request = inference_pb2.PredictRequest(
            request_id=f"{ep_id}-{step_id}",
            model_id=model_id, priority=priority,
            images=image_frames, state=state.tolist(),
            timestamp_ns=time.time_ns(),
        )

        resp = self._stub.Predict(request)
        return np.array(resp.actions, dtype=np.float32), resp.inference_time_ms

    def close(self) -> None:
        self._channel.close()

    @staticmethod
    def _encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def run_client_loop(
    server_addr: str,
    model_id: str,
    pretrained_path: str,
    task: str,
    episodes: int,
    output_dir: str,
    no_video: bool,
) -> None:
    from envs.aloha_env import AlohaEnv, AlohaEnvConfig

    client = InferenceClient(server_addr)

    logger.info("Requesting model load: %s", model_id)
    result = client.load_model(model_id, pretrained_path)
    if not result["success"]:
        logger.warning("Load response: %s", result["message"])

    logger.info("Server status: %s", client.status())

    from serving.action_buffer import ActionBuffer

    env_cfg = AlohaEnvConfig(task=task)
    env = AlohaEnv(env_cfg, device="cpu")

    for ep in range(episodes):
        seed = 42 + ep
        episode_id = client.new_episode_id()
        logger.info("Episode %d/%d  seed=%d", ep + 1, episodes, seed)

        obs_dict = env.reset(seed=seed)
        buffer = ActionBuffer(action_dim=env.action_dim)
        frames: list[np.ndarray] = []
        total_reward = 0.0
        steps = 0
        success = False
        done = False
        inf_ms = 0.0

        while not done:
            if buffer.size() == 0:
                images = {}
                state = np.zeros(env.action_dim, dtype=np.float32)
                for key, val in obs_dict.items():
                    if "images" in key:
                        cam_name = key.split(".")[-1]
                        img_np = (val.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        images[cam_name] = img_np
                    elif "state" in key:
                        state = val.cpu().numpy()

                actions, inf_ms = client.predict(model_id, images, state, episode_id=episode_id)
                buffer.push(actions.reshape(-1, env.action_dim) if actions.size > env.action_dim else actions.reshape(1, -1))

            action = buffer.pop()
            obs_dict, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated
            if "is_success" in info:
                success = success or bool(info["is_success"])

            if not no_video:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

        logger.info(
            "  reward=%.2f  steps=%d  success=%s  buf_size=%d  inf_time=%.1fms",
            total_reward, steps, success, buffer.size(), inf_ms,
        )

        if not no_video and frames:
            path = Path(output_dir) / f"episode_{ep:03d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(path), frames, fps=env.fps)
            logger.info("  Saved video: %s", path)

    env.close()
    client.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
    )
    parser = argparse.ArgumentParser(description="VLA Inference Client")
    parser.add_argument("--server", type=str, default="localhost:50051")
    parser.add_argument("--model-id", type=str, default="act-transfer-cube")
    parser.add_argument("--model-path", type=str, default="lerobot/act_aloha_sim_transfer_cube_human")
    parser.add_argument("--task", type=str, default="AlohaTransferCube-v0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs/client_eval")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    run_client_loop(
        server_addr=args.server, model_id=args.model_id,
        pretrained_path=args.model_path, task=args.task,
        episodes=args.episodes, output_dir=args.output_dir,
        no_video=args.no_video,
    )


if __name__ == "__main__":
    main()
