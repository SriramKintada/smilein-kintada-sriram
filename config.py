import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

resize_x = 224
resize_y = 224
input_channels = 3

face_batch_size = 16
smile_batch_size = 32

face_epochs = 20
smile_epochs = 10

face_lr = 1e-4
smile_lr = 1e-4

smile_num_classes = 2

unknown_threshold = 0.5
smile_frames_required = 3

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

face_data_dir = "data/face_identity"
smile_data_dir = "data/smile"

checkpoints_dir = "checkpoints"
face_weights_path = "checkpoints/face_net_weights.pth"
smile_weights_path = "checkpoints/smile_net_weights.pth"

attendance_csv = "attendance.csv"