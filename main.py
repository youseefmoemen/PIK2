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


def answer(caption, question):
    openai.api_key = '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'
    prompt = f'Answer the question using the caption ' \
             f'Question: {question}, caption: {caption}'
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_token=100,
    )
    return response


if __name__ == '__main__':
    path = 'video_samples/test1.mp4'
    q = 'what is the animal ?'
    frames = read_video(path)
    print(frames.shape)
    pik = PIK()
    result = pik.video_qa(frames, q)
    cap = ''
    for i in result:
        cap += ' ' + i

    print(f'Final caption: {cap}')
    final_answer = answer(cap, q)
    print(f'question answer: {final_answer}')
