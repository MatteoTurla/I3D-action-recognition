Dataset: Dataset/n_frames_b/16_frames/alerting/train
Numero Azioni 400
Classi: ['A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
Classi to index:
Label: A05 index: 0
Label: A06 index: 1
Label: A07 index: 2
Label: A08 index: 3
Label: A09 index: 4
Label: A10 index: 5
Label: A11 index: 6
Label: A12 index: 7
Transformazione temporale: center
Numero di frame se type != ALL: 16
Campionamento: 2
	bin: 50 	label: A05
	bin: 50 	label: A06
	bin: 50 	label: A07
	bin: 50 	label: A08
	bin: 50 	label: A09
	bin: 50 	label: A10
	bin: 50 	label: A11
	bin: 50 	label: A12
Dataset: Dataset/n_frames_b/16_frames/alerting/test
Numero Azioni 120
Classi: ['A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
Classi to index:
Label: A05 index: 0
Label: A06 index: 1
Label: A07 index: 2
Label: A08 index: 3
Label: A09 index: 4
Label: A10 index: 5
Label: A11 index: 6
Label: A12 index: 7
Transformazione temporale: center
Numero di frame se type != ALL: 16
Campionamento: 2
	bin: 20 	label: A05
	bin: 20 	label: A06
	bin: 10 	label: A07
	bin: 13 	label: A08
	bin: 18 	label: A09
	bin: 10 	label: A10
	bin: 19 	label: A11
	bin: 10 	label: A12
['conv3d_0c_1x1.conv3d.weight', 'conv3d_0c_1x1.conv3d.bias']
batch size: 64
SGD (
Parameter Group 0
    dampening: 0
    initial_lr: 0.1
    lr: 0.1
    momentum: 0.9
    nesterov: False
    weight_decay: 0
)
CrossEntropyLoss()
{'milestones': [30, 60], 'gamma': 0.1, 'base_lrs': [0.1], 'last_epoch': -1}
