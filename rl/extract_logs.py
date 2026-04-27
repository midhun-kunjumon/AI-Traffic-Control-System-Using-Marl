import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_metrics(base_log_dir):
    report_file = "metrics_report.txt"
    with open(report_file, "w") as f:
        f.write(f"Scanning base log directory: {base_log_dir}\n")
        
        if not os.path.exists(base_log_dir):
            f.write(f"Directory not found: {base_log_dir}\n")
            return

        # Find all event files recursively
        event_files = []
        for root, dirs, files in os.walk(base_log_dir):
            for file in files:
                if "events.out.tfevents" in file:
                    event_files.append(os.path.join(root, file))

        if not event_files:
            f.write("No event files found anywhere.\n")
            return

        f.write(f"Found {len(event_files)} event file(s).\n")

        for path in event_files:
            rel_path = os.path.relpath(path, base_log_dir)
            f.write(f"\n{'='*50}\nProcessing: {rel_path}\n{'='*50}\n")
            
            try:
                ea = EventAccumulator(path)
                ea.Reload()
                
                tags = ea.Tags()['scalars']
                f.write(f"Available Tags: {tags}\n")
                
                # Extract relevant metrics
                target_tags = {
                    'Reward (Mean)': 'rollout/ep_rew_mean',
                    'Length (Mean)': 'rollout/ep_len_mean',
                    'Loss (Total)': 'train/loss',
                    'Value Loss': 'train/value_loss',
                    'Policy Loss': 'train/policy_gradient_loss'
                }
                
                for name, tag in target_tags.items():
                    if tag in tags:
                        events = ea.Scalars(tag)
                        values = [e.value for e in events]
                        steps = [e.step for e in events]
                        
                        f.write(f"\n--- {name} ({tag}) ---\n")
                        f.write(f"Total datapoints: {len(values)}\n")
                        if values:
                            f.write(f"First 5: {[round(v, 4) for v in values[:5]]}\n")
                            f.write(f"Last 5:  {[round(v, 4) for v in values[-5:]]}\n")
                            f.write(f"Min: {min(values):.4f}, Max: {max(values):.4f}\n")
                            if len(values) > 1:
                                f.write(f"Trend (First -> Last): {values[0]:.4f} -> {values[-1]:.4f}\n")
                            
            except Exception as e:
                f.write(f"Error reading {rel_path}: {e}\n")
    
    print(f"Metrics written to {report_file}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_path, "logs") # Scan parent logs folder
    extract_metrics(log_dir)
