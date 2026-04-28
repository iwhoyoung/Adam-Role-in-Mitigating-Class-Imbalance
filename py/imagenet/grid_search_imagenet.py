import subprocess
import argparse


def run_grid_search(model_name,opt_name,lrs,scheduler,epoch):
    for lr in lrs:
        cmd = [
            "python", "main_10class.py",
            "-a", model_name,
            "--lr", str(lr),
            "--optimizer",opt_name,
            "--scheduler",scheduler,
            "-p", "1",
            "--epochs",str(int(epoch)),
            "-b", "512","/home/wangjzh/adam_optimizer/data/imagenet2012_ceil"
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(["rm","-rf","/home/wangjzh/adam_optimizer/data/imagenet2012/val/.ipynb_checkpoints"])
        subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search for model training")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., resnet18)")
    parser.add_argument("--opt_name", type=str, required=True, help="Optimizer name (e.g., sgd or adam)")
    parser.add_argument("--scheduler", type=str, required=True, help="Optimizer name (e.g., sgd or adam)")
    #parser.add_argument("--dataset_name", type=str, required=True, help="Optimizer name (e.g., sgd or adam)")
    #parser.add_argument("--datapath", type=str, required=True, help="List of learning rates to try")
    parser.add_argument("--lrs", nargs='+', type=float, required=True, help="List of learning rates to try")    
    parser.add_argument("--epochs",  type=float, required=True, help="List of learning rates to try")    
    #parser.add_argument("--batch_size", type=str, required=True, help="List of learning rates to try")    
    #parser.add_argument("--cuda_visible_devices", type=str, required=True, help="List of learning rates to try")    
    args = parser.parse_args()

    run_grid_search(args.model_name,args.opt_name,args.lrs,args.scheduler,args.epochs)
