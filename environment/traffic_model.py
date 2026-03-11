"""
Traffic generation model for the satellite IoT environment.

This module encapsulates the simple random-traffic model used in the CA-DRL
paper: sensors generate short-burst transmissions with bounded duration, and
arrivals follow a Bernoulli process with fixed probability per time step.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class TrafficConfig:
    max_traffic_duration: int
    sensor_arrival_rate: float


class TrafficModel:
    """
    Simple Bernoulli arrival traffic model.

    At each time step a new sensor request arrives with probability
    `sensor_arrival_rate`. The requested transmission duration is uniformly
    sampled from {1, ..., max_traffic_duration}. The data size is sampled from
    a short-burst range (100–1000 bits) as in the reference implementation.
    """

    def __init__(self, config: TrafficConfig) -> None:
        self.config = config

    def maybe_generate_request(self, current_time: int) -> tuple | None:
        """
        Either generate a new (arrival_time, data_size_bits, duration_steps)
        tuple for the given time step, or return None if no request arrives.
        """
        if random.random() >= self.config.sensor_arrival_rate:
            return None

        arrival_time = current_time
        data_size = random.randint(100, 1000)
        duration = random.randint(1, self.config.max_traffic_duration)
        return arrival_time, data_size, duration


