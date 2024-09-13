import sys
sys.path.append('adv')
from flat_square import flat_square_attack
from patch_square import PatchSquareAttack
from basic_attacks import pgd_attack, square_attack, auto_attack, auto_attack_dict, patch_square_attack
from adv_training import adversarial_training