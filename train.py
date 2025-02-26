import sim
from cfg_loader import parse
from trainers import make_trainer, play_a_game
import gymnasium as gym


if __name__ == "__main__":
    args = parse()
    if args["test_mode"]:
        args["cfg"]["env"]["render_mode"] = "human"
        print(play_a_game(args["cfg"]))
    else:
        make_trainer(args["cfg"]).train()    
