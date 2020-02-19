import dmc2gym
import time
import tqdm

env = dmc2gym.make(
        domain_name='reacher',
        task_name='easy',
        seed=0,
        visualize_reward=False,
        from_pixels=True,
        height=84,
        width=84,
        frame_skip=3
    )

t = time.time()
obs = env.reset()
for i in tqdm.tqdm(range(1000)):
    a = env.action_space.sample()
    o,r,d,info = env.step(a)
    if d:
        break

print('time',time.time()-t)