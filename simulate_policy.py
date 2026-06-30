import sys
import os
import argparse
import torch
import numpy as np
import cv2
import gymnasium as gym

# Setup imports correctly for ManiSkill and squint
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import logging
logging.disable(level=logging.WARN)

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.visualization.misc import tile_images
import utils
import envs
import mani_skill.envs

from train_squint import DeployAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='SO101LiftCube-v1')
    parser.add_argument('--checkpoint', type=str, default='runs/baseline/ckpt.pt')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--save_video', action='store_true', help='Save simulation to a video file instead of opening a window')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        'num_envs': 1,
        'seed': 42,
        'obs_mode': 'rgb+segmentation', 
        'render_mode': 'rgb_array',
        'image_size': 128,
        'domain_randomization': True,
        'window_size': 512,
    }

    print(f"Instantiating: {args.env_id}")
    sensor_size = {'width': config['image_size'], 'height': config['image_size']}
    
    env = gym.make(
        args.env_id,
        obs_mode=config['obs_mode'],
        render_mode=config['render_mode'],
        sensor_configs=sensor_size,
        human_render_camera_configs=sensor_size,
        num_envs=config['num_envs'],
        domain_randomization=config['domain_randomization'],
        reconfiguration_freq=None,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    obs, info = env.reset(seed=config['seed'])

    print(f"Loading checkpoint from: {args.checkpoint}")
    wandb_config = None
    if args.checkpoint == 'wandb':
        wandb_config = {
            'wandb_entity': 'aboardman04-rif-robotics',
            'wandb_project_name': 'maniskill-so101',
            'agent_name': 'squint',
            'env_id': args.env_id,
            'version': 'latest',
            'seed': 42  # Try a few seeds if this fails
        }
    
    agent = DeployAgent(env, obs, target_image_size=16, device=device)
    try:
        agent.load_checkpoint(args.checkpoint, wandb_config)
    except Exception as e:
        if args.checkpoint == 'wandb':
            # Fallback for seed search logic if 42 is wrong
            for seed in [1, 2, 3, 4, 5]:
                try:
                    wandb_config['seed'] = seed
                    agent.load_checkpoint(args.checkpoint, wandb_config)
                    break
                except Exception:
                    pass
    agent.eval()

    window_size = config['window_size']
    
    print(f"Running task: {args.env_id}. Press 'q' or Esc to exit.")

    for step in range(args.steps):
        with torch.no_grad():
            obs_tensor = {
                'rgb': obs['rgb'].to(device),
                'state': obs['state'].to(device)
            }
            action = agent(obs_tensor).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        done = (terminated | truncated).any()

        render_rgb = env.render()
        
        obs_rgb_vis = obs['rgb']
        if obs_rgb_vis.shape[-1] != 3 and obs_rgb_vis.shape[-1] % 3 == 0:
            obs_rgb_vis = obs_rgb_vis[..., :3]
        
        render_h, render_w = render_rgb.shape[1], render_rgb.shape[2]
        if obs_rgb_vis.shape[1] != render_h or obs_rgb_vis.shape[2] != render_w:
            obs_rgb_vis = torch.nn.functional.interpolate(
                obs_rgb_vis.permute(0, 3, 1, 2).float(),
                size=(render_h, render_w),
                mode='nearest',
            ).permute(0, 2, 3, 1).to(torch.uint8)
            
        paired = torch.cat([obs_rgb_vis, render_rgb], dim=2)
        rgb = tile_images(paired, nrows=1).cpu().numpy().astype(np.uint8)
        rgb = cv2.resize(rgb, dsize=(window_size * 2, window_size))
        
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if args.save_video:
            if not hasattr(env, 'video_writer'):
                os.makedirs('videos', exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = rgb.shape[:2]
                env.video_writer = cv2.VideoWriter(
                    f'videos/{args.env_id}_policy.mp4',
                    fourcc,
                    20.0,
                    (w, h)
                )
                print(f"Saving video to videos/{args.env_id}_policy.mp4 ...")
            env.video_writer.write(rgb)
        else:
            cv2.imshow(f"{args.env_id} - Interleaved: Obs | Render per env", rgb)
            key = cv2.waitKey(30)
            if key == 27 or key == ord('q'):
                break

        if done:
            obs, info = env.reset()

    if args.save_video and hasattr(env, 'video_writer'):
        env.video_writer.release()
    env.close()
    cv2.destroyAllWindows()
    print(f"Finished: {args.env_id}")

if __name__ == "__main__":
    main()
