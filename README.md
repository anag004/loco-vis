Code for `Legged Locomotion in Challenging Terrain using Egocentric Vision`

Training details:
```
python3 train.py --task a1 --max_iterations 100000 --headless
python3 train_dagger_recurrent.py --task a1 --max_iterations 100000 --load_run <phase 1 policy> --num_envs 256
python3 train_ff_rma.py --task a1 --max_iterations 100000 --load_run <phase 1 policy> --num_envs 256
```
