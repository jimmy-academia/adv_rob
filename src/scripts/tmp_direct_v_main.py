import sys
sys.path.append('.')
from utils import run_command

run_command('python main.py --task tmp_direct --train_env ZLQH_dir --record_path_suffix _dir --direct')
run_command('python main.py --task tmp_direct')
