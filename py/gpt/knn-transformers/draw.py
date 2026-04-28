import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_group_losses(json_path: str | Path) -> None:
    """
    读取指定 JSON → root.log_history，并绘制 group_0_loss~group_9_loss 共 10 条 loss 曲线

    Parameters
    ----------
    json_path : str | pathlib.Path
        JSON 文件路径。结构示例:
        {
            "root": {
                "log_history": [
                    {"step": 0, "group_0_loss": 1.2, "group_1_loss": 1.1, ...},
                    {"step": 1, "group_0_loss": 1.1, ...},
                    ...
                ]
            }
        }
    """
    json_path = Path(json_path)

    # ---------- 1. 读取数据 ----------
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    log_history = data["log_history"]

    # ---------- 2. 收集每组 loss ----------
    losses = {i: [] for i in range(10)}
    total_loss=[]
    steps = {i: [] for i in range(10)}  # x 轴：用 step 字段，如缺省则用索引
    total_step=[]
    i=0
    
    for idx, record in enumerate(log_history):
        #print(record)
        if "loss" in record.keys():
            total_step.append(record.get("step", idx))
            total_loss.append(record.get("loss",np.nan))
        # key = f"group_{i}_loss"
        # if key in record.keys():
        #     steps[i].append(record.get("step", idx))
        #     losses[i].append(record.get(key, np.nan))  # 缺失值填 nan，便于连续绘图
        # else:
        #     print("skip:"+key)
        #     total_step.append(record.get("step", idx))
        #     total_loss.append(record.get("loss",np.nan))
        #     print(record)
        #     continue
        i=(i+1)%10
    
    # ---------- 3. 绘图 ----------
    
    def block_average(x, y, k: int = 100):
        """
        把 (x, y) 每 k 个点取一次平均。
        返回降采样后的 (x_avg, y_avg)。
        """
        x = np.asarray(x)
        y = np.asarray(y)
        n = (len(y) // k) * k         # 保证能整除
        if n == 0:
            return x, y               # 不足 k 个点直接返回原曲线
        x_avg = x[:n].reshape(-1, k).mean(axis=1)
        y_avg = y[:n].reshape(-1, k).mean(axis=1)
        return x_avg, y_avg

    plt.figure(figsize=(10, 6))
    # for i in range(10):
    #     ds_step, ds_loss = block_average(steps[i], losses[i], k=100)
    #     plt.plot(
    #         ds_step,
    #         ds_loss,
    #         label=f"group_{i}_loss",
    #         linewidth=1.6,
    #     )
    print(total_step)
    print(total_loss)
    ds_step, ds_loss = block_average(total_step, total_loss, k=1)
    plt.plot(
            ds_step,
            ds_loss,
            label=f"total_loss",
            linewidth=1.6,
    )

    plt.title("Group Loss Curves")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend(ncol=2)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("totalloss.png", dpi=150)


# 用法示例
if __name__ == "__main__":
    plot_group_losses("finetune_gpt2_wikitext103_4gpu_sgd/trainer_state.json")
