import os
import time

import pygame

from fish_speech.models.text2semantic.inference import main as generate
from pathlib import Path
from fish_speech.models.vqgan.inference import main as infer
from fish_speech.models.vqgan.inference import load_model
from fish_speech.models.text2semantic.inference import load_model as load_text2semantic_model 
from SmartAITool.core import *
import torch





class VoiceService:
    def __init__(self):
        self._output_dir = "outputs/"
        os.makedirs(self._output_dir, exist_ok=True)


    def fishspeech(self, text):
        start_time = time.time()
        execution_start_time = None  # Track execution start time excluding model loading
        w_checkpoint_path = Path("checkpoints/fish-speech-1.5")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = torch.half
         #---------------------------------load model---------------------------------#
        # Load model
        load_model_start = time.time()
        model = load_model(
            config_name="firefly_gan_vq",
            checkpoint_path="checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            device="cuda"
        )
        cprint(f"Model loading time: {time.time() - load_model_start:.2f} seconds")

        #----------------------------------load text2semantic model---------------------------------#
        load_text2semantic_start = time.time()
        
        text2semantic_model, decode_one_token = load_text2semantic_model(
            checkpoint_path=w_checkpoint_path, device="cuda", precision=precision, compile=True
        )
        cprint(f"Text2Semantic model loading time: {time.time() - load_text2semantic_start:.2f} seconds")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.device(device):
            text2semantic_model.setup_caches(
                max_batch_size=1,
                max_seq_len=text2semantic_model.config.max_seq_len,
                dtype=next(text2semantic_model.parameters()).dtype,
            )

        # Start execution timer after model loading
        execution_start_time = time.time()

        #----------------------------------load audio file---------------------------------#
        clone_start = time.time()
        infer(input_path=Path("/home/ubuntu/m15kh/fish-speech/jarvis.wav"), output_path=Path(self._output_dir + "clone.wav"),
              checkpoint_path="checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
              config_name="firefly_gan_vq", device="cuda", model=model)
        cprint(f"Voice cloning time: {time.time() - clone_start:.2f} seconds")

        #--------------------------------generate audio---------------------------------#
        generate_start = time.time()
        generate(text=text,
                       prompt_text=[
                           "accessing alarm and interface settings in this window you can set up your customized greeting and alarm preferences the world needs your expertise or at least your presence launching a series of displays to help guide you",
                       ],
                       prompt_tokens=[Path(self._output_dir + "clone.npy")],
                       checkpoint_path=Path("checkpoints/fish-speech-1.5"),
                       half=True,
                       device="cuda",
                       num_samples=2,
                       max_new_tokens=1024,
                       top_p=0.7,
                       repetition_penalty=1.2,
                       temperature=0.3,
                       compile=True,
                       seed=42,
                       iterative_prompt=True,
                       chunk_length=100,
                       output_dir=self._output_dir,
                       model=text2semantic_model,
                       decode_one_token=decode_one_token,
                       )
        cprint(f"Audio generation time: {time.time() - generate_start:.2f} seconds")

        cprint("Generated audio files:")

        generate_start = time.time()
        generate(text=text,
                       prompt_text=[
                           "accessing alarm and interface settings in this window you can set up your customized greeting and alarm preferences the world needs your expertise or at least your presence launching a series of displays to help guide you",
                       ],
                       prompt_tokens=[Path(self._output_dir + "clone.npy")],
                       checkpoint_path=Path("checkpoints/fish-speech-1.5"),
                       half=True,
                       device="cuda",
                       num_samples=2,
                       max_new_tokens=1024,
                       top_p=0.7,
                       repetition_penalty=1.2,
                       temperature=0.3,
                       compile=True,
                       seed=42,
                       iterative_prompt=True,
                       chunk_length=100,
                       output_dir=self._output_dir,
                       model=text2semantic_model,
                       decode_one_token=decode_one_token,
                       )
        cprint(f"Audio generation time: {time.time() - generate_start:.2f} seconds")

        cprint("Generated audio files:")

        #----------------------------------infer audio---------------------------------#
        infer_audio_start = time.time()
        infer(input_path=Path("outputs/codes_1.npy"), output_path=Path(self._output_dir + "1.wav"),
              checkpoint_path="checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
              config_name="firefly_gan_vq", device="cuda", model=model)
        cprint(f"Final audio inference time: {time.time() - infer_audio_start:.2f} seconds")

        # Calculate execution time excluding model loading
        cprint(f"Execution time (excluding model loading): {time.time() - execution_start_time:.2f} seconds")
        cprint(f"Total execution time: {time.time() - start_time:.2f} seconds")

    # def play(self, temp_audio_file):
    #     pygame.mixer.quit()
    #     pygame.mixer.init()
    #     pygame.mixer.music.load(temp_audio_file)
    #     pygame.mixer.music.stop()
    #     pygame.mixer.music.play()

    #     while pygame.mixer.music.get_busy():
    #         pygame.time.Clock().tick(10)

    #     pygame.mixer.music.stop()
    #     pygame.mixer.quit()

        # os.remove(temp_audio_file)