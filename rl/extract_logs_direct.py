import os
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_metrics():
    # Hardcoded known path based on list_dir output
    base_file = r"c:\ai-traffic-control - marl\rl\logs\PPO_Quick_0\events.out.tfevents.1769274063.JITHUMITHU.9128.0"
    report_file = "metrics_direct_report.txt"
    
    with open(report_file, "w") as f:
        f.write(f"Target file: {base_file}\n")
        
        if not os.path.exists(base_file):
            f.write("File does not exist!\n")
            return

        try:
            ea = EventAccumulator(base_file)
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
                        f.write(f"First 5: {[v for v in values[:5]]}\n")
                        f.write(f"Last 5:  {[v for v in values[-5:]]}\n")
                        f.write(f"Min: {min(values)}, Max: {max(values)}\n")
                else:
                    f.write(f"\n--- {name} ({tag}) ---\n")
                    f.write("NOT FOUND in logs\n")
                        
        except Exception as e:
            f.write(f"Error reading file: {e}\n")

    print(f"Report written to {report_file}")

if __name__ == "__main__":
    extract_metrics()
