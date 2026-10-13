import numpy as np

def unwrap_with_nans(theta):
    """Unwrap angles while preserving NaNs and aligning segments across gaps."""
    theta_unwrapped = theta.copy()

    valid_indices = np.where(np.isfinite(theta))[0]
    segments = np.split(valid_indices, np.where(np.diff(valid_indices) > 1)[0] + 1)

    previous_end = None

    for segment in segments:
        if len(segment) == 0:
            continue

        values = np.unwrap(theta[segment])

        if previous_end is not None:
            offset = 2 * np.pi * np.round((previous_end - values[0]) / (2 * np.pi))
            values += offset

        theta_unwrapped[segment] = values
        previous_end = values[-1]

    return theta_unwrapped