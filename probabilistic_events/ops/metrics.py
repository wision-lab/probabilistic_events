import torch
from loguru import logger
from torch import Tensor


class PSNR:
    """
    A class to compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    This utility can be used in two ways:
    1.  **Stateless:** By calling the instance directly `psnr(pred, true)`, you can
        compute the PSNR for a single pair of tensors.
    2.  **Stateful:** By using the `update()` method over several batches and then
        calling `compute()`, you can get the aggregate PSNR over a dataset or
        an epoch. This is useful for tracking metrics during model training
        or visualization loops.

    Attributes:
        max_value (float): The maximum possible pixel value of the images
                           (e.g., 1.0 for normalized float images, 255.0 for
                           uint8 images).
    """

    def __init__(self, max_value: float = 1.0):
        """
        Initializes the PSNR metric calculator.

        Args:
            max_value (float): The maximum value of the input tensors.
                               Defaults to 1.0.
        """
        if not isinstance(max_value, (int, float)) or max_value <= 0:
            raise ValueError("`max_value` must be a positive number.")
        self.max_value = max_value
        self._sum_squared_error: float = 0.0
        self._num_elements: int = 0

    def __call__(self, pred: Tensor, true: Tensor) -> Tensor:
        """
        Computes the PSNR for a single pair of prediction and ground truth tensors.

        This is a stateless operation.

        Args:
            pred (Tensor): The predicted tensor.
            true (Tensor): The ground truth tensor.

        Returns:
            Tensor: A scalar tensor containing the PSNR value.
        """
        mse = torch.mean((pred - true) ** 2)
        return self._compute_psnr(mse)

    def update(self, pred: Tensor, true: Tensor) -> None:
        """
        Updates the internal state with a new batch of predictions and ground truths.

        This is a stateful operation. Use `compute()` to get the final metric.

        Args:
            pred (Tensor): The predicted tensor batch.
            true (Tensor): The ground truth tensor batch.
        """
        if pred.shape != true.shape:
            raise ValueError(
                "Prediction and ground truth tensors must have the same shape."
            )
        self._sum_squared_error += torch.sum((pred - true) ** 2)
        self._num_elements += pred.numel()

    def compute(self) -> Tensor:
        """
        Computes the PSNR over all accumulated batches since the last reset.

        This is a stateful operation.

        Returns:
            Tensor: A scalar tensor containing the aggregate PSNR value.
                    Returns inf if the mean squared error is zero.
        """
        if self._num_elements == 0:
            raise RuntimeError("PSNR must be updated with at least one batch.")
        mean_squared_error = self._sum_squared_error / self._num_elements
        return self._compute_psnr(mean_squared_error)

    def compute_and_reset(self) -> Tensor:
        """
        A convenience method to compute the aggregate PSNR and then reset the state.

        Returns:
            Tensor: A scalar tensor containing the aggregate PSNR value.
        """
        result = self.compute()
        self.reset()
        return result

    def reset(self) -> None:
        """Resets the internal state of the metric."""
        self._sum_squared_error = 0.0
        self._num_elements = 0

    def _compute_psnr(self, mse: Tensor) -> Tensor:
        """Helper function to calculate PSNR from a given MSE."""
        if mse == 0.0:
            return torch.tensor(float("inf"))
        return 20.0 * torch.log10(torch.tensor(self.max_value)) - 10.0 * torch.log10(
            mse
        )


# --- Usage Example ---
if __name__ == "__main__":
    # Create two example tensors (e.g., images)
    # `true_image` is a simple gradient
    # `pred_image` is the same gradient with some added noise
    true_image = torch.linspace(0, 1, 256 * 256).reshape(1, 1, 256, 256)
    noise = (torch.rand_like(true_image) - 0.5) * 0.1  # Small random noise
    pred_image = (true_image + noise).clamp(0, 1)

    # === 1. Stateless Usage (for a single batch) ===
    logger.info("--- Stateless Usage ---")
    psnr_calculator = PSNR(max_value=1.0)
    psnr_value = psnr_calculator(pred_image, true_image)
    logger.info(f"PSNR for a single pair of images: {psnr_value.item():.2f} dB")

    # === 2. Stateful Usage (for an entire dataset/epoch) ===
    logger.info("--- Stateful Usage ---")
    # Simulate an visualization loop over a few batches
    stateful_psnr = PSNR(max_value=1.0)
    num_batches = 5

    for i in range(num_batches):
        # Create slightly different noisy images for each "batch"
        batch_noise = (torch.rand_like(true_image) - 0.5) * (0.05 * (i + 1))
        batch_pred = (true_image + batch_noise).clamp(0, 1)

        # Update the metric with the new batch
        stateful_psnr.update(batch_pred, true_image)
        logger.info(f"Updated with batch {i + 1}")

    # After the loop, compute the aggregate PSNR
    aggregate_psnr = stateful_psnr.compute()
    logger.info(
        f"Aggregate PSNR over {num_batches} batches: {aggregate_psnr.item():.2f} dB"
    )

    # Demonstrate reset
    stateful_psnr.reset()
    logger.info(
        f"Internal state after reset: error={stateful_psnr._sum_squared_error}, "
        f"elements={stateful_psnr._num_elements}"
    )
