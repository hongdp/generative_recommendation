import sys
import numpy as np
import grain.python as grain
from absl import flags

flags.FLAGS(sys.argv)

# Dummy data
enc_in = np.arange(100).reshape((10, 10))
tar = np.arange(10)

class InMemoryDataSource:
    def __init__(self, enc, tar):
        self.enc = enc
        self.tar = tar
    def __len__(self):
        return len(self.enc)
    def __getitem__(self, idx):
        return self.enc[idx], self.tar[idx]

source = InMemoryDataSource(enc_in, tar)
sampler = grain.IndexSampler(
    num_records=len(source),
    num_epochs=1,
    shard_options=grain.NoSharding(),
    shuffle=True,
    seed=42,
)
dataloader = grain.DataLoader(
    data_source=source,
    sampler=sampler,
    worker_count=2,
    worker_buffer_size=2,
    operations=[
        grain.Batch(batch_size=4, drop_remainder=True)
    ]
)

for batch in dataloader:
    print(batch[0].shape, batch[1].shape)

