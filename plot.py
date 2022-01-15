import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay


def calibration_plot(X_test, Y_test, pipeline_NB, pipeline_RF, pipeline_NN):
    """Plot calibration figure"""

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    colors = plt.cm.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}

    display = CalibrationDisplay.from_estimator(
        pipeline_NB,
        X_test,
        Y_test,
        n_bins=10,
        name='NB',
        ax=ax_calibration_curve,
        color=colors(0),
    )
    calibration_displays['NB'] = display

    display = CalibrationDisplay.from_estimator(
        pipeline_RF,
        X_test,
        Y_test,
        n_bins=10,
        name='RF',
        ax=ax_calibration_curve,
        color=colors(1),
    )
    calibration_displays['RF'] = display

    display = CalibrationDisplay.from_estimator(
        pipeline_NN,
        X_test,
        Y_test,
        n_bins=10,
        name='NN',
        ax=ax_calibration_curve,
        color=colors(2),
    )
    calibration_displays['NN'] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots")

    # Add histogram
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]

    row, col = grid_positions[0]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays['NB'].y_prob,
        range=(0, 1),
        bins=10,
        label='NB',
        color=colors(0),
    )
    ax.set(title='NB', xlabel="Mean predicted probability", ylabel="Count")

    row, col = grid_positions[1]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays['RF'].y_prob,
        range=(0, 1),
        bins=10,
        label='RF',
        color=colors(1),
    )
    ax.set(title='RF', xlabel="Mean predicted probability", ylabel="Count")

    row, col = grid_positions[2]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays['NN'].y_prob,
        range=(0, 1),
        bins=10,
        label='NN',
        color=colors(2),
    )
    ax.set(title='NN', xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.show()
