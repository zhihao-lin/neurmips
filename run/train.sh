python teacher_train.py --config-name replica-kitchen \
checkpoint.teacher='replica-test-teacher' \
train.epoch.teacher=10 train.epoch.validation=10 train.epoch.checkpoint=5

python experts_train.py --config-name replica-kitchen \
checkpoint.teacher='replica-test-teacher' checkpoint.experts='replica-test-experts' \
train.epoch.distill=10 train.epoch.finetune=30 train.epoch.validation=10 train.epoch.checkpoint=5

