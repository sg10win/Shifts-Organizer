import numpy as np
from collections import defaultdict
import pandas as pd


class ShiftScheduler:
    def __init__(self, start_hour, end_hour, min_rest, min_shift, max_shift, priorities):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.min_rest = min_rest
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.priorities = priorities
        self.total_hours = end_hour - start_hour
        self.schedule = []
        self.work_distribution = defaultdict(int)  # Track hours worked by each worker
        self.remaining_hours = {}

    def equally_distribute_hours(self, workers):
        total_hours_per_worker = self.total_hours // len(workers)
        extra_hours = self.total_hours % len(workers)

        # Distribute the extra hours
        return {worker: total_hours_per_worker + (1 if i < extra_hours else 0) for i, worker in enumerate(workers)}

    def assign_shifts(self, workers):
        self.remaining_hours = self.equally_distribute_hours(workers)
        current_worker = None
        shift_start = self.start_hour
        last_shift_end = defaultdict(lambda: None)  # Track the last shift end time for each worker

        for hour in range(self.start_hour, self.end_hour):
            if not current_worker or (hour - shift_start >= self.max_shift):
                if current_worker:
                    # Close the current shift and calculate rest time for the same worker
                    shift_length = hour - shift_start  # Calculate shift length
                    if last_shift_end[current_worker] is not None:
                        rest_time = shift_start - last_shift_end[current_worker]  # Correct rest calculation
                    else:
                        rest_time = None

                    self.schedule.append({
                        "worker": current_worker,
                        "start": shift_start,
                        "end": hour,
                        "rest_to_next_shift": rest_time,
                        "shift_length": shift_length
                    })
                    self.work_distribution[current_worker] += shift_length
                    self.remaining_hours[current_worker] -= shift_length
                    last_shift_end[current_worker] = hour  # Update last shift end time

                # Select a new worker for the next shift
                current_worker = self.get_best_worker(hour, workers, last_shift_end)
                shift_start = hour

            # Enforce max shift time or stop if the day ends
            if (hour + 1 - shift_start >= self.max_shift) or (hour == self.end_hour - 1):
                shift_length = hour + 1 - shift_start  # Calculate shift length
                if last_shift_end[current_worker] is not None:
                    rest_time = shift_start - last_shift_end[current_worker]
                else:
                    rest_time = None

                self.schedule.append({
                    "worker": current_worker,
                    "start": shift_start,
                    "end": hour + 1,
                    "rest_to_next_shift": rest_time,
                    "shift_length": shift_length
                })
                self.work_distribution[current_worker] += shift_length
                self.remaining_hours[current_worker] -= shift_length
                last_shift_end[current_worker] = hour + 1
                current_worker = None
                shift_start = hour + 1

        return self.schedule

    def get_best_worker(self, hour, workers, last_shift_end):
        available_workers = [
            w for w in workers
            if self.priorities[hour].get(w, 1) > 0 and
               (last_shift_end[w] is None or hour - last_shift_end[w] >= self.min_rest)
        ]

        # Sort by fewest remaining hours, then by priority (prefer higher priority)
        best_worker = min(available_workers, key=lambda w: (self.remaining_hours[w], -self.priorities[hour][w]))
        return best_worker


# Initialize priorities
free_priorities = {i: {'A': 1, 'B': 1, 'C': 1} for i in range(48)}
users_priorities = {
    0: {'A': 2, 'B': 1, 'C': 0},
    1: {'A': 1, 'B': 2, 'C': 0},
    2: {'A': 0, 'B': 1, 'C': 2}
    # Add more user-defined priorities for each hour as needed
}

# Merge user-defined priorities with default priorities
for u_p in users_priorities:
    free_priorities[u_p] = users_priorities[u_p].copy()

# Copy finalized priorities
priorities = free_priorities.copy()

# Example Usage
start_hour = 0
end_hour = 48
min_rest = 12
min_shift = 4
max_shift = 8

workers = ['A', 'B', 'C']

scheduler = ShiftScheduler(start_hour, end_hour, min_rest, min_shift, max_shift, priorities)
schedule = scheduler.assign_shifts(workers)

for shift in schedule:
    print(
        f"Worker {shift['worker']} works from hour {shift['start']} to hour {shift['end']} with rest to next shift: {shift['rest_to_next_shift']} and shift length: {shift['shift_length']} hours")

pd.DataFrame(schedule).to_excel('outputs\\schedule.xlsx')