"""Training metrics tracker with visualization."""


class MetricsTracker:
    """Track and display training metrics with visual improvements."""

    def __init__(self):
        self.history = []
        self.best_loss = float("inf")
        self.best_epoch = 0

    def update(self, epoch: int, metrics: dict):
        """Update metrics for current epoch.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric values
        """
        self.history.append(metrics)

        if metrics["loss"] < self.best_loss:
            self.best_loss = metrics["loss"]
            self.best_epoch = epoch

    def get_previous(self, metric_name: str):
        """Get previous epoch's metric value.

        Args:
            metric_name: Name of the metric to retrieve

        Returns:
            Previous metric value or None if not available
        """
        if len(self.history) < 2:
            return None
        return self.history[-2].get(metric_name)

    def print_header(self):
        """Print column headers for metrics table."""
        print("\n" + "=" * 100)
        print(
            f"{'Epoch':>6} | {'Loss':>10} | {'ΔLoss':>10} | {'Improve':>8} | "
            f"{'Reg':>8} | {'MaxNorm':>8} | {'Outside':>7} | {'Status':>10}"
        )
        print("=" * 100)

    def print_epoch_summary(self, epoch: int, metrics: dict, total_epochs: int):
        """Print compact summary of epoch with improvement indicators.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of current metrics
            total_epochs: Total number of epochs planned
        """
        prev_loss = self.get_previous("loss")

        # Calculate improvement
        if prev_loss is not None:
            delta = metrics["loss"] - prev_loss
            pct_change = (delta / prev_loss) * 100 if prev_loss != 0 else 0

            if delta < 0:
                status = "✓ BETTER"
                delta_str = f"{delta:+.4f}"
                pct_str = f"{pct_change:+.2f}%"
                improve_color = "\033[92m"  # Green
            else:
                status = "✗ WORSE"
                delta_str = f"{delta:+.4f}"
                pct_str = f"{pct_change:+.2f}%"
                improve_color = "\033[91m"  # Red

            reset_color = "\033[0m"
        else:
            delta_str = "---"
            pct_str = "---"
            status = "FIRST"
            improve_color = ""
            reset_color = ""

        # Format output
        outside_pct = (
            (metrics["outside_count"] / metrics["total_nodes"]) * 100
            if metrics["total_nodes"] > 0
            else 0
        )

        print(
            f"{epoch:6d} | "
            f"{metrics['loss']:10.6f} | "
            f"{improve_color}{delta_str:>10}{reset_color} | "
            f"{improve_color}{pct_str:>8}{reset_color} | "
            f"{metrics['reg_loss']:8.6f} | "
            f"{metrics['max_norm']:8.4f} | "
            f"{outside_pct:6.2f}% | "
            f"{improve_color}{status:>10}{reset_color}"
        )

        # Additional info every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"       └─ Best: {self.best_loss:.6f} @ epoch {self.best_epoch} | "
                f"Norms: [{metrics['min_norm']:.4f}, {metrics['mean_norm']:.4f}, {metrics['max_norm']:.4f}]"
            )

    def print_final_summary(self):
        """Print final training summary with overall statistics."""
        print("\n" + "=" * 100)
        print("TRAINING SUMMARY")
        print("=" * 100)
        print(f"Total epochs: {len(self.history)}")
        print(f"Best loss: {self.best_loss:.6f} (epoch {self.best_epoch})")

        if len(self.history) >= 2:
            first_loss = self.history[0]["loss"]
            last_loss = self.history[-1]["loss"]
            total_improvement = first_loss - last_loss
            pct_improvement = (total_improvement / first_loss) * 100
            print(f"Total improvement: {total_improvement:+.6f} ({pct_improvement:+.2f}%)")

        print("=" * 100 + "\n")
