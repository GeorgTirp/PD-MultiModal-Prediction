def raincloud_plot(data: pd.DataFrame, modality_name: str, features_list: list, safe_path: str = "") -> None:
    """Create a refined raincloud plot using seaborn for better aesthetics."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Prepare data
    data_list = [data[col].dropna().values for col in data.columns]
    x_positions = np.arange(1, len(data_list) + 1)
    
    # Define colors based on modality
    if modality_name == "UPDRS":
        boxplots_colors = ['#009E73', '#CC79A7', '#525252']
        violin_colors = ['#009E73', '#CC79A7', '#525252']
        scatter_colors = ['#FF595E', '#FF595E', '#FF595E']
    else:
        boxplots_colors = ['#009E73', '#CC79A7']
        violin_colors = ['#009E73', '#CC79A7']
        scatter_colors = ['#FF595E', '#FF595E']
    # color palette from DL project:
    #custom_palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#525252"]
    # Set width parameters
    box_width = 0.8
    violin_width = 0.6
    violin_shift = box_width / 2  # Align violin to inner edge of boxplot

    # Violin plot (half-violin) aligned to boxplot
    for idx, (x, values) in enumerate(zip(x_positions, data_list)):
        sns.violinplot(
            y=values,
            inner=None,
            cut=0,
            linewidth=0,
            color=violin_colors[idx],
            ax=ax,
            width=violin_width,
            split=True
        )
        for collection in ax.collections[-1:]:
            if not isinstance(collection, PolyCollection):
                continue  # skip non-violin objects
            for path in collection.get_paths():
                verts = path.vertices
                mean_x = np.mean(verts[:, 0])
                
                if modality_name == "UPDRS":
                    #
                    # Keep your existing "all-same-side" logic here.
                    # For simplicity, let's do "left half" for all.
                    #
                    verts[:, 0] = np.clip(verts[:, 0], -np.inf, mean_x)
                    # Shift everything to x - violin_shift
                    shift_amount = x - violin_shift

                else:
                    # We have 2 columns. For idx=0 => left half, idx=1 => right half
                    if idx == 0:
                        # Clip to left half
                        verts[:, 0] = np.clip(verts[:, 0], -np.inf, mean_x)
                        # Shift to left
                        shift_amount = x - violin_shift
                    else:
                        # Clip to right half
                        verts[:, 0] = np.clip(verts[:, 0], mean_x, np.inf)
                        # Shift to right
                        shift_amount = x + violin_shift

                # Apply the shift
                verts[:, 0] += shift_amount
                path.vertices = verts
        
    
    # Boxplot (make it slimmer)
    bp = sns.boxplot(
        data=data,
        width=0.2,
        showcaps=True,
        whiskerprops={'color': 'black'},
        medianprops={'color': 'black'},
        flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red', 'alpha': 0.5},
        ax=ax
    )
    for patch, color in zip(bp.artists, boxplots_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
   
    
    # Scatter plot (overlay on boxplot)
    for idx, (x, values) in enumerate(zip(x_positions, data_list)):
        x_jitter = np.random.uniform(low=-0.05, high=0.05, size=len(values)) + x
        ax.scatter(x_jitter, values, s=10, color=scatter_colors[idx], alpha=0.7)
        
        # Draw lines for BDI modality
        if modality_name == "BDI" and idx < len(x_positions) - 1:
            next_values = data_list[idx + 1]
            for i in range(len(values)):
                if i < len(next_values):
                    x_start, y_start = x_jitter[i], values[i]
                    x_end, y_end = x_positions[idx + 1] + np.random.uniform(low=-0.05, high=0.05), next_values[i]
                    slope = y_end - y_start
                    if slope > 0:
                        line_color = "red"
                    elif slope < 0:
                        line_color = "green"
                    else:
                        line_color = "grey"
                    ax.add_line(Line2D([x_start, x_end], [y_start, y_end], color=line_color, alpha=0.4))

    # Labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(features_list)
    ax.set_ylabel(modality_name)
    ax.set_title(f"{modality_name} Raincloud Plot")
    
    # Add legend for BDI
    if modality_name == "BDI":
        line_colors = {"increase": "blue", "decrease": "red", "no change": "grey"}
        legend_elements = [
            Line2D([0], [0], color=line_colors["increase"], lw=2, label="Deterioration"),
            Line2D([0], [0], color=line_colors["decrease"], lw=2, label="Improvement"),
            Line2D([0], [0], color=line_colors["no change"], lw=2, label="No change"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
    
    plt.savefig(safe_path + modality_name + "_raincloud_plot.png")
    plt.show()
    plt.close()