Dataset: ./Dataset/n_frames_b/32_frames/basic/train
Numero Azioni 276
Classi: ['A01', 'A02', 'A03', 'A04']
Classi to index:
Label: A01 index: 0
Label: A02 index: 1
Label: A03 index: 2
Label: A04 index: 3
Transformazione temporale: center
Numero di frame se type != ALL: 32
Campionamento: 2
	bin: 69 	label: A01
	bin: 69 	label: A02
	bin: 69 	label: A03
	bin: 69 	label: A04
Dataset: ./Dataset/n_frames_b/32_frames/basic/test
Numero Azioni 92
Classi: ['A01', 'A02', 'A03', 'A04']
Classi to index:
Label: A01 index: 0
Label: A02 index: 1
Label: A03 index: 2
Label: A04 index: 3
Transformazione temporale: center
Numero di frame se type != ALL: 32
Campionamento: 2
	bin: 30 	label: A01
	bin: 20 	label: A02
	bin: 32 	label: A03
	bin: 10 	label: A04
['conv3d_0c_1x1.conv3d.weight', 'conv3d_0c_1x1.conv3d.bias']
SGD (
Parameter Group 0
    dampening: 0
    initial_lr: 0.1
    lr: 0.1
    momentum: 0.9
    nesterov: False
    weight_decay: 0.01
)
CrossEntropyLoss()
{'milestones': [20, 60], 'gamma': 0.1, 'base_lrs': [0.1], 'last_epoch': -1}
