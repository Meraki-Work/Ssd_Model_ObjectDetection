import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Ganti dengan path ke folder training kamu
EVENTS_DIR = 'training'

# Cari file event
event_files = [os.path.join(EVENTS_DIR, f) for f in os.listdir(EVENTS_DIR) if f.startswith('events.out.tfevents')]
if not event_files:
    print("Tidak ditemukan file event di folder training.")
    exit()

# Gunakan file pertama (biasanya yang terbaru)
ea = event_accumulator.EventAccumulator(event_files[0])
ea.Reload()

# Ambil data loss
loss_tags = ['Loss/total_loss', 'Loss/classification_loss', 'Loss/localization_loss']
for tag in loss_tags:
    if tag not in ea.Tags()['scalars']:
        print(f"Tag '{tag}' tidak ditemukan di file event.")
        continue
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    plt.plot(steps, values, label=tag)

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
