import torchvision
import torch
from model.PIK import PIK


def read_test():
    video_path = 'video_samples/test1.mp4'
    stream = 'video'
    video_reader = torchvision.io.VideoReader(video_path, stream)
    fms = [frame['data'] for frame in video_reader]
    video_frames = torch.stack(fms)
    return video_frames[:3]


if __name__ == '__main__':
    frames = read_test()
    print(frames.shape)
    pik = PIK()
    result = pik.video_qa(frames, 'What?')
    print('Final result')
    print(result)
    for i in result:
        print(i)
