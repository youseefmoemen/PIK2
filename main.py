import torchvision
import torch
from model.PIK import PIK
import openai


def read_video(video_path):
    stream = 'video'
    video_reader = torchvision.io.VideoReader(video_path, stream)
    fms = [frame['data'] for frame in video_reader]
    video_frames = torch.stack(fms)
    selected_frames = [video_frames[i] for i in range(0, min(video_frames.shape[0], 900), 30)]
    return torch.stack(selected_frames)


def answer(caption, question):
    openai.api_key = ''  # Add your API key here
    prompt = f"the following text is video caption answer the following question:  {question} using the provided " \
             f"caption: {caption}"
    messages = [{
        'role': 'user',
        'content': prompt
    }]

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    reply = chat.choices[0].message.content
    print(f'Question: {question}')
    print(f'Answer: {reply}')


if __name__ == '__main__':
    path = 'video_samples/test6.mp4'
    q = 'what is in the video ?'
    frames = read_video(path)
    print(frames.shape)
    pik = PIK()
    result = pik.video_qa(frames, q)
    cap = ''
    for c in result:
        cap += ' ' + c[-1]

    print(f'Final caption: {cap}')
    answer(cap, q)
