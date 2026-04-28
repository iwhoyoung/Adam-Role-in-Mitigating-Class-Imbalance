import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from pathlib import Path

def _find_group_cols(df: pd.DataFrame) -> list:
    """找到数据框中的分组列（假设列名包含'group'）"""
    return [col for col in df.columns if 'group' in col.lower()]

def plot_total_loss(adam_df: pd.DataFrame, sgd_df: pd.DataFrame,adambn_df: pd.DataFrame, adamini_df: pd.DataFrame,adamsbn: pd.DataFrame, out_dir: Path):
    """Plot Adam vs SGD total loss."""
    fig = plt.figure(figsize=(10, 7))  # 创建图形并赋值给 fig
    ax = plt.gca()  # 获取当前坐标轴对象

    
    
    ax.plot(
        np.arange(len(adamini_df)) * 500, 
        adamini_df["total_loss"],
        label="Adam-S",
        linestyle="-",
        linewidth=6,
        color="tab:brown",
    )
    ax.plot(
        np.arange(len(sgd_df)) * 500, 
        adamsbn["total_loss"],
        label="Adam-S-LDN",
        linestyle="-",
        linewidth=6,
        color="tab:purple",
    )
    ax.plot(
        np.arange(len(adam_df)) * 500, 
        adam_df["total_loss"],
        label="Adam",
        linestyle="-",
        linewidth=6,
        color="tab:blue",
    )
    ax.plot(
        np.arange(len(adambn_df)) * 500, 
        adambn_df["total_loss"],
        label="Adam-LDN",
        linestyle="-",
        linewidth=6,
        color="tab:red",
    )
    # ax.plot(
    #     np.arange(len(sgd_df)) * 500, 
    #     sgd_df["total_loss"],
    #     label="SGD",
    #     linestyle="-",
    #     linewidth=6,
    #     color="tab:orange",
    # )
    
    

    ax.set_xlabel("Step", fontsize=40)
    ax.set_ylabel("Train Loss", fontsize=40)
    ax.legend(fontsize=40)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
    ax.set_ylim(bottom=-0.1,top=14)  # 设置y轴起点为0
    
    plt.tight_layout()

    out_path = out_dir / "total_loss_adam_vs_sgd.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    #print(f"[✔] 已保存 {out_path.relative_to(Path.cwd())}")


def plot_group_losses(df: pd.DataFrame, optimiser: str, out_dir: Path):
    """Plot the group loss curves for a single optimiser."""
    group_cols = _find_group_cols(df)
    if not group_cols:
        print(f"[!] 未找到分组列，跳过绘制{optimiser}的分组损失图")
        return
        
    cmap = cm.viridis(np.linspace(0, 1, len(group_cols)))

    fig = plt.figure(figsize=(10, 7))  # 创建图形并赋值给 fig
    ax = plt.gca()  # 获取当前坐标轴对象

    for idx, col in enumerate(group_cols):
        ax.plot(
            np.arange(len(df)) * 500, 
            df[col],
            label=f"Group {idx}",
            linestyle="-",
            linewidth=6,
            color=cmap[idx],
            alpha=0.8
        )

    ax.set_xlabel("Step", fontsize=40)
    ax.set_ylabel("Train Loss", fontsize=40)
    #ax.set_title(f"{optimiser.upper()} – Group Losses", fontsize=40)
    #ax.legend(fontsize=40, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
    ax.set_ylim(bottom=-0.1,top=12)  # 设置y轴起点为0
    
    plt.tight_layout()
    #plt.tight_layout(rect=[0, 0, 0.85, 1])
    filename = f"{optimiser.lower()}_group_losses.png"
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    #print(f"[✔] 已保存 {out_path.relative_to(Path.cwd())}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='绘制GPT模型的损失对比图')
    parser.add_argument('--adam_csv', required=True, help='包含Adam优化器损失数据的CSV文件路径')
    # parser.add_argument('--sgd_csv', required=True, help='包含SGD优化器损失数据的CSV文件路径')
    # parser.add_argument('--adambn_csv', required=True, help='包含SGD优化器损失数据的CSV文件路径')
    # parser.add_argument('--adamini_csv', required=True, help='包含SGD优化器损失数据的CSV文件路径')
    # parser.add_argument('--adamsbn_csv', required=True, help='包含SGD优化器损失数据的CSV文件路径')
    # parser.add_argument('--adamsgd_m_csv', required=True, help='包含SGD优化器损失数据的CSV文件路径')
    parser.add_argument('--output_dir', required=True, help='图表输出目录')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取CSV数据
    try:
        adam_df = pd.read_csv(args.adam_csv)
        # sgd_df = pd.read_csv(args.sgd_csv)
        # adambn_df=pd.read_csv(args.adambn_csv)
        # adamini_df=pd.read_csv(args.adamini_csv)
        # adam_sbn = pd.read_csv(args.adamsbn_csv)
        # adam_sgd_m = pd.read_csv(args.adamsgd_m_csv)
        # print(f"[✔] 成功读取数据: Adam={len(adam_df)}行, SGD={len(sgd_df)}行, ADAMBN={len(adambn_df)}行")
    except Exception as e:
        print(f"[!] 读取CSV文件失败: {str(e)}")
        return
    
    # 绘制总损失对比图
    # plot_total_loss(adam_df, sgd_df,adambn_df,adamini_df,adam_sbn, output_dir)
    
    # 绘制各组损失图
    plot_group_losses(adam_df, "adam", output_dir)
    # plot_group_losses(adambn_df, "adambn", output_dir)
    # plot_group_losses(sgd_df, "sgd", output_dir)
    # plot_group_losses(adamini_df, "adamini", output_dir)
    # plot_group_losses(adam_sbn, "adamsbn", output_dir)
    # plot_group_losses(adam_sgd_m, "adam_sgd_m", output_dir)

if __name__ == "__main__":
    main()
