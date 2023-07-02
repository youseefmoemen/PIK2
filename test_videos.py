import torchvision
import torch
import matplotlib.pyplot as plt


def read_video(video_path):
    stream = 'video'
    video_reader = torchvision.io.VideoReader(video_path, stream)
    fms = [frame['data'] for frame in video_reader]
    video_frames = torch.stack(fms)
    selected_frames = [video_frames[i] for i in range(0, min(video_frames.shape[0], 900), 30)]
    return torch.stack(selected_frames)


if __name__ == '__main__':
    path = 'video_samples/test7.mp4'
    frames = read_video(path)
    print(frames.shape)
