"""Design-first mode: an AI writes a small network's weights from nothing, then training polishes them."""

from .network import Network
from .tasks import TASKS, get_task, random_network
from .train import compare, train_network

__all__ = ["Network", "TASKS", "get_task", "random_network", "compare", "train_network"]
