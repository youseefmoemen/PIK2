import torchvision
import torch
from model.PIK import PIK
import openai


def read_video(video_path):
    stream = 'video'
    video_reader = torchvision.io.VideoReader(video_path, stream)
    fms = [frame['data'] for frame in video_reader]
    video_frames = torch.stack(fms)
    return video_frames


if __name__ == '__main__':
    path = 'video_samples/test1.mp4'
    frames = read_video(path)
    print(frames.shape)
    pik = PIK()
    result = pik.video_qa(frames, 'What?')
    print('Final result')
    print('Len.results: ', len(result))
    print('----------------')
    for i in result:
        print(len(i))
        print(i)
