import subprocess
import argparse


def run_grid_search(model_name, opt_name, lrs, batch_size,dataset_name="cifar_r10", datapath="adam_optimizer/data/cifar100_lt_outputs/cifar100-lt-r-10",visible_device="0,1,2,3",seed='0'):
    for lr in lrs:
        cmd = [
            "python", "main_sup_base.py",
            "--model_name", model_name,
            "--opt_name", opt_name,
            "--lr", str(lr),
            "--dataset_name", dataset_name,
            "--datapath", datapath,
            "--cuda_visible_devices", visible_device,
            "--nThreads","8",
            "--batch_size",str(batch_size),
            "--seed",str(seed)
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search for model training")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., resnet18)")
    parser.add_argument("--opt_name", type=str, required=True, help="Optimizer name (e.g., sgd or adam)")
    parser.add_argument("--dataset_name", type=str, required=True, help="Optimizer name (e.g., sgd or adam)")
    parser.add_argument("--datapath", type=str, required=True, help="List of learning rates to try")
    parser.add_argument("--lrs", nargs='+', type=float, required=True, help="List of learning rates to try")    
    parser.add_argument("--batch_size", type=str, required=True, help="List of learning rates to try")    
    parser.add_argument("--cuda_visible_devices", type=str, required=True, help="List of learning rates to try")    
    parser.add_argument('--seed', type=int, default='0', help='可见的CUDA设备')
    args = parser.parse_args()

    run_grid_search(args.model_name, args.opt_name, args.lrs,args.batch_size,args.dataset_name,args.datapath,args.cuda_visible_devices,args.seed)
