_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/vockvboard.py',
    '../_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
# runtime settings
total_epochs = 12
