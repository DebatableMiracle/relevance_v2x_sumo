# relevance_v2x_sumo
1. TTC event frequency (no communication, isolates raw conflict rate):

bash
python test_ttc_frequency.py --map manhattan --cav 0.5 --vehicles 40 --steps 2000

2. No-comm vs. random-broadcast comparison (real, honest numbers even without a trained model):

bash
python test_comparison.py --map manhattan --cav 0.5 --vehicles 40 --steps 2000

3. Same comparison, including your trained policy (only if you get a saved model before Monday):

bash
python test_comparison.py --map manhattan --cav 0.5 --vehicles 40 --steps 2000 --model ../models/ippo_final.zip
Useful variations for exploring density/CAV% sensitivity
bash
# higher density
python test_comparison.py --map manhattan --cav 0.5 --vehicles 60 --steps 2000

# lower CAV penetration
python test_comparison.py --map manhattan --cav 0.2 --vehicles 40 --steps 2000

# higher CAV penetration
python test_comparison.py --map manhattan --cav 0.9 --vehicles 40 --steps 2000

# on your original small grid instead of Manhattan
python test_ttc_frequency.py --map grid --cav 0.5 --vehicles 25 --steps 2000
Training (once you're ready to actually run it)
bash
python train_ippo.py --map manhattan --cav 0.5 --k 3 --vehicles 40 --n_steps 512 --n_updates 200

Short smoke-test version first (confirm it runs before committing to a long run):

bash
python train_ippo.py --map manhattan --cav 0.5 --k 3 --vehicles 40 --n_steps 128 --n_updates 10

GUI-visible version (slower, real traci not libsumo, for watching behavior):

bash
python train_ippo.py --map manhattan --cav 0.5 --k 3 --vehicles 40 --n_steps 128 --n_updates 10 --gui
Benchmark / speed check (if things feel slow again)
bash
python benchmark.py --backend libsumo --map manhattan --cav 0.5 --k 3 --vehicles 40
python benchmark.py --backend traci --map manhattan --cav 0.5 --k 3 --vehicles 40
Sanity checks (already validated, but good to have if something breaks again)
bash
python test_determinism.py
python test_occlusion_flip.py
python test_manhattan_sanity.py
wandb (if running training)
bash
wandb login   # one-time

Runs auto-log to your v2x-broadcast-selection project when train_ippo.py runs with --wandb (default on).