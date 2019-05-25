Dataset: /home/Dataset/aggregorio_videos_pytorch_boxcrop/alerting/train
Classes: ['A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
Classes to index:
Label: A05 index: 0
Label: A06 index: 1
Label: A07 index: 2
Label: A08 index: 3
Label: A09 index: 4
Label: A10 index: 5
Label: A11 index: 6
Label: A12 index: 7
Numero di frame: 32
Downsample: 2
Balance: True
Padding: False
removed: 18
Temporal annotation: None
Numero Azioni 400
A05 	 50
A06 	 50
A07 	 50
A08 	 50
A09 	 50
A10 	 50
A11 	 50
A12 	 50
Dataset: /home/Dataset/aggregorio_videos_pytorch_boxcrop/alerting/test
Classes: ['A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
Classes to index:
Label: A05 index: 0
Label: A06 index: 1
Label: A07 index: 2
Label: A08 index: 3
Label: A09 index: 4
Label: A10 index: 5
Label: A11 index: 6
Label: A12 index: 7
Numero di frame: 32
Downsample: 2
Balance: False
Padding: False
removed: 10
Temporal annotation: None
Numero Azioni 120
A05 	 20
A06 	 20
A07 	 10
A08 	 13
A09 	 18
A10 	 10
A11 	 19
A12 	 10
['mixed_5b.branch_0.conv3d.weight', 'mixed_5b.branch_0.batch3d.weight', 'mixed_5b.branch_0.batch3d.bias', 'mixed_5b.branch_1.0.conv3d.weight', 'mixed_5b.branch_1.0.batch3d.weight', 'mixed_5b.branch_1.0.batch3d.bias', 'mixed_5b.branch_1.1.conv3d.weight', 'mixed_5b.branch_1.1.batch3d.weight', 'mixed_5b.branch_1.1.batch3d.bias', 'mixed_5b.branch_2.0.conv3d.weight', 'mixed_5b.branch_2.0.batch3d.weight', 'mixed_5b.branch_2.0.batch3d.bias', 'mixed_5b.branch_2.1.conv3d.weight', 'mixed_5b.branch_2.1.batch3d.weight', 'mixed_5b.branch_2.1.batch3d.bias', 'mixed_5b.branch_3.1.conv3d.weight', 'mixed_5b.branch_3.1.batch3d.weight', 'mixed_5b.branch_3.1.batch3d.bias', 'mixed_5c.branch_0.conv3d.weight', 'mixed_5c.branch_0.batch3d.weight', 'mixed_5c.branch_0.batch3d.bias', 'mixed_5c.branch_1.0.conv3d.weight', 'mixed_5c.branch_1.0.batch3d.weight', 'mixed_5c.branch_1.0.batch3d.bias', 'mixed_5c.branch_1.1.conv3d.weight', 'mixed_5c.branch_1.1.batch3d.weight', 'mixed_5c.branch_1.1.batch3d.bias', 'mixed_5c.branch_2.0.conv3d.weight', 'mixed_5c.branch_2.0.batch3d.weight', 'mixed_5c.branch_2.0.batch3d.bias', 'mixed_5c.branch_2.1.conv3d.weight', 'mixed_5c.branch_2.1.batch3d.weight', 'mixed_5c.branch_2.1.batch3d.bias', 'mixed_5c.branch_3.1.conv3d.weight', 'mixed_5c.branch_3.1.batch3d.weight', 'mixed_5c.branch_3.1.batch3d.bias', 'conv3d_0c_1x1.conv3d.weight', 'conv3d_0c_1x1.conv3d.bias']
batch size: 32
numero epoche: 15
best model at epoch: 11
drop out prob: 0.0
SGD (
Parameter Group 0
    dampening: 0
    initial_lr: 0.1
    lr: 0.010000000000000002
    momentum: 0.9
    nesterov: False
    weight_decay: 0.01
)
CrossEntropyLoss()
{'milestones': [10, 15], 'gamma': 0.1, 'base_lrs': [0.1], 'last_epoch': 14}
