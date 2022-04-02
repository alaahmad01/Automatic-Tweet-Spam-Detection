import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay


def calibration_plot(X_test, Y_test, pipes):
    """Plot calibration figure"""

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    colors = plt.cm.get_cmap("Dark2")
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = [
        CalibrationDisplay.from_estimator(
            pipes[0],
            X_test,
            Y_test,
            n_bins=10,
            name='NB',
            ax=ax_calibration_curve,
            color=colors(0),
        ),
        CalibrationDisplay.from_estimator(
            pipes[1],
            X_test,
            Y_test,
            n_bins=10,
            name='RF',
            ax=ax_calibration_curve,
            color=colors(1),
        ),
        CalibrationDisplay.from_estimator(
            pipes[2],
            X_test,
            Y_test,
            n_bins=10,
            name='NN',
            ax=ax_calibration_curve,
            color=colors(1),
        )
    ]

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")

    # Add histogram
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]

    for i in range(len(pipes)):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])
        ax.hist(
            calibration_displays[i].y_prob,
            range=(0, 1),
            bins=10,
            label=str(i),
            color=colors(0),
        )
        ax.set(title=str(i), xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.show()
