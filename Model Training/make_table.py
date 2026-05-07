import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(14, 4))
fig.patch.set_facecolor("#1a1f2e")
ax.set_facecolor("#1a1f2e")
ax.axis("off")

cols = ["Threshold","Trades","Trades/Yr","Win%","TP Exits","SL Exits","EOD Exits","Net P&L","Avg/Trade","Max Drawdown"]
rows = [
    ["0.45","886","886.6","37.9%","322","542","22","-$33,005","-$37","$53,699"],
    ["0.50","345","345.2","36.5%","122","217","6","-$14,429","-$42","$34,364"],
    ["0.55","127","127.1","37.0%","46","79","2","-$6,432","-$51","$10,753"],
    ["0.60","18","18.0","27.8%","5","13","0","-$6,702","-$372","$8,174"],
]

col_widths = [0.09,0.07,0.09,0.07,0.08,0.08,0.08,0.10,0.10,0.13]
x_starts = []
x = 0.01
for w in col_widths:
    x_starts.append(x)
    x += w

fig.text(0.5, 0.97, "NQ Ensemble Backtest \u2014 Threshold Sensitivity Analysis",
         ha="center", va="top", fontsize=14, fontweight="bold", color="white", fontfamily="monospace")
fig.text(0.5, 0.90, "Long-Only  |  6.0x TP / 4.0x SL ATR Barriers  |  $4 Commission + 1-Tick Slippage",
         ha="center", va="top", fontsize=9, color="#aaaaaa", fontfamily="monospace")

header_y = 0.76
for col, xs, w in zip(cols, x_starts, col_widths):
    ax.add_patch(mpatches.FancyBboxPatch((xs, header_y-0.06), w-0.005, 0.10,
        boxstyle="square,pad=0", facecolor="#252d3d",
        transform=fig.transFigure, figure=fig, clip_on=False))
    fig.text(xs + (w-0.005)/2, header_y, col, ha="center", va="center",
             fontsize=8.5, color="#7eb8f7", fontweight="bold",
             fontfamily="monospace", transform=fig.transFigure)

row_colors = ["#1a1f2e", "#20253a"]
for r, row in enumerate(rows):
    y  = header_y - 0.14 - r * 0.14
    bg = "#2a2a1a" if r == 2 else row_colors[r % 2]
    for i, (cell, xs, w) in enumerate(zip(row, x_starts, col_widths)):
        ax.add_patch(mpatches.FancyBboxPatch((xs, y-0.06), w-0.005, 0.10,
            boxstyle="square,pad=0", facecolor=bg,
            transform=fig.transFigure, figure=fig, clip_on=False))
        if i in (7, 8):    color = "#ff6b6b"
        elif i == 9:       color = "#ffaa44"
        elif i == 3:       color = "#ffe066"
        else:              color = "white"
        fig.text(xs + (w-0.005)/2, y, cell, ha="center", va="center",
                 fontsize=9, color=color, fontfamily="monospace",
                 transform=fig.transFigure)
    if r == 2:
        fig.text(x, y, "  << Target Range", ha="left", va="center",
                 fontsize=7.5, color="#ffe066", fontfamily="monospace",
                 transform=fig.transFigure)

out = r"C:\Users\chadm\OneDrive\Desktop\Quant Lab\Model Training\Outputs\threshold_sensitivity.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="#1a1f2e")
print(f"Saved: {out}")
