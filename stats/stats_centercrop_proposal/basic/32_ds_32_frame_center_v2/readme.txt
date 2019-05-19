Dataset: /home/Dataset/aggregorio_videos_pytorch_centercrop/basic/train
Classes: ['A01', 'A02', 'A03', 'A04']
Classes to index:
Label: A01 index: 0
Label: A02 index: 1
Label: A03 index: 2
Label: A04 index: 3
Numero di frame: 32
Downsample: 2
Balance: True
Padding: False
removed: 59
Temporal annotation: /home/Dataset/crop_temp.json
Numero Azioni 276
A01 	 69
A02 	 69
A03 	 69
A04 	 69
Dataset: /home/Dataset/aggregorio_videos_pytorch_centercrop/basic/test
Classes: ['A01', 'A02', 'A03', 'A04']
Classes to index:
Label: A01 index: 0
Label: A02 index: 1
Label: A03 index: 2
Label: A04 index: 3
Numero di frame: 32
Downsample: 2
Balance: False
Padding: False
removed: 28
Temporal annotation: /home/Dataset/crop_temp.json
Numero Azioni 92
A01 	 30
A02 	 20
A03 	 32
A04 	 10
['conv3d_0c_1x1.conv3d.weight', 'conv3d_0c_1x1.conv3d.bias']
batch size: 32
numero epoche: 75
best model at epoch: 19
drop out prob: 0.0
SGD (
Parameter Group 0
    dampening: 0
    initial_lr: 0.1
    lr: 0.0010000000000000002
    momentum: 0.9
    nesterov: False
    weight_decay: 0.01
)
CrossEntropyLoss()
{'milestones': [20, 60], 'gamma': 0.1, 'base_lrs': [0.1], 'last_epoch': 74}
