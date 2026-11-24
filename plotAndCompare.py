import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_cpu = np.array(Image.open("julia_cpu.ppm"))
img_gpu = np.array(Image.open("julia_gpu.ppm"))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

bg_color = "#0F172A"
card_color = "#1E293B"
text_color = "#F8FAFC"
muted_text = "#CBD5E1"

fig = plt.figure(figsize=(16, 8.2), dpi=300, facecolor=bg_color)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.22,
                      left=0.04, right=0.97, top=0.75, bottom=0.17)

kernel_speedup = np.mean(cpu_times) / np.mean(gpu_kernel_times)
total_speedup = np.mean(cpu_times) / np.mean(gpu_total_times)

fig.suptitle(
    f"CUDA vs. Single-Core CPU: {kernel_speedup:.1f}x Kernel Speedup (1000×1000 Julia Set)",
    fontsize=20, fontweight="bold", color=text_color, y=0.94
)
fig.text(
    0.5, 0.87,
    f"1,000,000 pixels × 200 max iterations  |  NVIDIA T4 vs. Host CPU (g++ -O3)  |  {total_speedup:.1f}x end-to-end incl. PCIe copy",
    ha="center", fontsize=13, fontweight="normal", color=muted_text
)

# Panel 1: CPU Render
ax0 = fig.add_subplot(gs[0])
ax0.imshow(img_cpu)
ax0.set_title(f"CPU Render\n{np.mean(cpu_times):.2f} ms avg", fontsize=15, color=text_color, pad=12)
ax0.set_xticks([])
ax0.set_yticks([])
for spine in ax0.spines.values():
    spine.set_edgecolor("#334155")
    spine.set_linewidth(2)

# Panel 2: GPU Render
ax1 = fig.add_subplot(gs[1])
ax1.imshow(img_gpu)
ax1.set_title(
    f"NVIDIA T4 GPU Render\n{np.mean(gpu_kernel_times):.2f} ms kernel  |  {np.mean(gpu_total_times):.2f} ms total",
    fontsize=15, color=text_color, pad=12
)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_edgecolor("#334155")
    spine.set_linewidth(2)

# Panel 3: Bar Chart
ax2 = fig.add_subplot(gs[2])
ax2.set_facecolor(card_color)

labels = ["CPU\n(1 Thread)", "GPU\n(Kernel Only)", "GPU\n(Kernel + Copy)"]
means = [np.mean(cpu_times), np.mean(gpu_kernel_times), np.mean(gpu_total_times)]
stds = [np.std(cpu_times), np.std(gpu_kernel_times), np.std(gpu_total_times)]
colors = ["#3B82F6", "#10B981", "#F59E0B"]

bars = ax2.bar(
    labels, means, yerr=stds, capsize=7, width=0.55,
    color=colors, edgecolor="white", linewidth=1.2,
    error_kw={"ecolor": text_color, "elinewidth": 2, "capthick": 2}
)

ax2.set_yscale("log")
ax2.set_ylim(1, max(means) * 3.5)
ax2.set_ylabel("Execution Time (ms, log scale)", fontsize=13.5, color=text_color, labelpad=10)
ax2.set_title("Latency Comparison (Lower is Better)", fontsize=15, color=text_color, pad=12)

ax2.tick_params(axis="x", labelsize=12.5, colors=text_color, pad=8)
ax2.tick_params(axis="y", labelsize=12.5, colors=text_color)
ax2.grid(axis="y", which="major", linestyle="--", linewidth=1, color="#475569", alpha=0.6)
ax2.set_axisbelow(True)

for spine in ["top", "right"]:
    ax2.spines[spine].set_visible(False)
for spine in ["bottom", "left"]:
    ax2.spines[spine].set_color("#64748B")
    ax2.spines[spine].set_linewidth(1.5)

for bar, m in zip(bars, means):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        m * 1.28,
        f"{m:.2f} ms",
        ha="center", va="bottom",
        fontsize=13.5, fontweight="bold", color=text_color
    )

ax2.text(
    0.96, 0.92,
    f"Kernel Speedup: {kernel_speedup:.1f}x\nTotal Speedup: {total_speedup:.1f}x",
    transform=ax2.transAxes,
    ha="right", va="top",
    fontsize=12.5, fontweight="bold", color="#10B981",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#0F172A", edgecolor="#10B981", linewidth=1.5)
)

fig.text(
    0.04, 0.04,
    "Benchmark: 10 runs after warmup (std dev error bars)  •  grid<<<1000×1000, 1>>>  •  float32 precision",
    fontsize=11, fontweight="normal", color=muted_text
)
fig.text(
    0.97, 0.04,
    "© Dr. Amir Mirmojarabian  |  linkedin.com/in/dr-amir-m-03a4b2113",
    ha="right", fontsize=11, fontweight="bold", color=muted_text
)

output_path = "julia_cuda_linkedin.png"
plt.savefig(output_path, dpi=300, facecolor=bg_color)
plt.show()
