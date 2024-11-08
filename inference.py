import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

if __name__ == '__main__':
    device = 'cuda:0'
    device = torch.device(device)
    clip_type = {
        'video': 'LanguageBind_Video',  # also LanguageBind_Video
        # 'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
        # 'thermal': 'LanguageBind_Thermal',
        # 'image': 'LanguageBind_Image',
        # 'depth': 'LanguageBind_Depth',
    }

    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    checkpoint_path = '/mnt/fast/nobackup/users/ef0036/LanguageBind/logs/bs128_a100_text_freeze_semantic_pop_bobsl_capt_8/checkpoints/epoch_18.pt'  # Update this with the actual path to your checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load only the state_dict
    model.load_state_dict(checkpoint['state_dict'], strict=True)  # Adjust to `strict=False` if needed

    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    # image = ['assets/image/0.jpg', 'assets/image/1.jpg']
    # audio = ['assets/audio/0.wav', 'assets/audio/1.wav']
    video = ['assets/video/0.mp4', 'assets/video/1.mp4']
    # depth = ['assets/depth/0.png', 'assets/depth/1.png']
    # thermal = ['assets/thermal/0.jpg', 'assets/thermal/1.jpg']
    language = ["Training a parakeet to climb up a ladder.", 'A lion climbing a tree to catch a monkey.']

    inputs = {
        # 'image': to_device(modality_transform['image'](image), device),
        'video': to_device(modality_transform['video'](video), device),
        # 'audio': to_device(modality_transform['audio'](audio), device),
        # 'depth': to_device(modality_transform['depth'](depth), device),
        # 'thermal': to_device(modality_transform['thermal'](thermal), device),
    }
    inputs['language'] = to_device(tokenizer(language, max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), device)

    with torch.no_grad():
        embeddings = model(inputs)

    print("Video x Text: \n",
          torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
    # print("Image x Text: \n",
    #       torch.softmax(embeddings['image'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
    # print("Depth x Text: \n",
    #       torch.softmax(embeddings['depth'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
    # print("Audio x Text: \n",
    #       torch.softmax(embeddings['audio'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
    # print("Thermal x Text: \n",
    #       torch.softmax(embeddings['thermal'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())

    # print("Video x Audio: \n",
    #       torch.softmax(embeddings['video'] @ embeddings['audio'].T, dim=-1).detach().cpu().numpy())
    # print("Image x Depth: \n",
    #       torch.softmax(embeddings['image'] @ embeddings['depth'].T, dim=-1).detach().cpu().numpy())
    # print("Image x Thermal: \n",
    #       torch.softmax(embeddings['image'] @ embeddings['thermal'].T, dim=-1).detach().cpu().numpy())