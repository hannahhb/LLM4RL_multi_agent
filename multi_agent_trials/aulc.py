import os
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalar_from_event_file(file_path, scalar_name='Train/Return Mean'):
    # Initialize the EventAccumulator to read the event file
    event_acc = EventAccumulator(file_path)
    event_acc.Reload()

    # Check available tags
    tags = event_acc.Tags()
    # print(tags)
    if 'scalars' not in tags or scalar_name not in tags['scalars']:
        raise ValueError(f"Scalar '{scalar_name}' not found in the event file.")

    # Extract scalar values
    scalar_events = event_acc.Scalars(scalar_name)
    print(scalar_events)
    steps = [event.step for event in scalar_events]
    values = [event.value for event in scalar_events]
    # return 0
    return steps, values

def calculate_aulc(steps, values):
    # Calculate the Area Under the Learning Curve using the trapezoidal rule
    aulc = np.trapz(values, steps)
    return aulc

# Path to your .tfevents file
event_file_path = '/Users/hannah_mac/Documents/projects/Multi-agent RL /LLM4RL_multi_agent/log/ppo/lavadoorkey/train_rl-0/events.out.tfevents.1716098906.Hannahs-MacBook-Pro.local.36388.0'  # Update this path

# Extract steps and reward values
steps, values = extract_scalar_from_event_file(event_file_path)
# extract_scalar_from_event_file(event_file_path)
# Calculate AULC
aulc = calculate_aulc(steps, values)
print(f"Area Under the Learning Curve (AULC): {aulc}")

# Optional: Save the steps and values to a CSV for further analysis or visualization
# df = pd.DataFrame({'steps': steps, 'values': values})
# df.to_csv('reward_curve.csv', index=False)
