### System Info + NVIDIA Drivers

I'm starting on a fresh install of Ubuntu 22.04 Server on a rented RTX 4090 instance from [TensorDock](https://www.tensordock.com/).

My instance already comes with NVIDIA drivers installed. However, if your's doesn't (i.e. running `nvidia-smi` returns nothing), then I'd recommend installing [Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software). For Ubuntu 22.04 Server, the command is:
```
wget -nv -O- https://lambdalabs.com/install-lambda-stack.sh | I_AGREE_TO_THE_CUDNN_LICENSE=1 sh -
```

After verifying NVIDIA drivers, we can begin with the `pyannote` install.

### Installing PyTorch & pyannote.audio

First, we need [`uv`](https://github.com/astral-sh/uv), which is a better version of `pip`.
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```
Then we create a Python virtual environment and activate it.
```
uv venv env
source env/bin/activate
```
We then install PyTorch and `pyannote` (this will take a few mins).
```
uv pip install torch torchaudio pyannote.audio
```

### Running pyannote.audio Offline

By default, `pyannote-audio` talks to HuggingFace servers. On the very first run, it downloads the model, and then on _every single_ subsequent run, it talks to the servers as well (I assume for analytics collection).

This is problematic for a production system; once, I was running a lot of diarization jobs and the system just froze up. Turned out HuggingFace servers were down, which halted the diarization!

Fortunately we can run `pyannote` offline, the official instructions for which are [here](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/applying_a_pipeline.ipynb). They're a bit cryptic, so, to simplify things, I've condensed them below, and collected the needed models and `yaml` config file in my own Github repo.

```
cd ~
git clone https://github.com/7egment/pyannote-3.1-offline
cd pyannote-3.1-offline
sed -i "s/e6quisitory/$USER/g" config.yaml
cd ..
```

### Diarization Script

Let's make a folder called `diarize` and put our diarization script in there.

```
mkdir diarize
cd diarize
```

The script (`diarize.py`):
```
import json
import os
import time
from pyannote.audio import Pipeline
from tqdm import tqdm
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
import torchaudio

device = torch.device("cuda:0")
wav_file = "audio.wav"
start_time = time.time()

user_name = os.getenv('USER') or os.getenv('USERNAME')
pipeline = Pipeline.from_pretrained(f"/home/{user_name}/pyannote-3.1-offline/config.yaml")
pipeline.to(device)
waveform, sample_rate = torchaudio.load(wav_file)
waveform = waveform.to(device)

with ProgressHook() as hook:
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)

segments = []
for segment, _, speaker in tqdm(diarization.itertracks(yield_label=True), desc="Processing segments", total=len(diarization)):
    segments.append({
        "speaker": speaker,
        "start": segment.start,
        "end": segment.end
    })

with open('segments.json', 'w') as outfile:
    json.dump(segments, outfile)

end_time = time.time()
print(f"Script executed in {end_time - start_time:.2f} seconds")
```

Now we need a `.wav` file to test. Let's grab [this one](https://www.youtube.com/watch?v=dwE0qMxCpbE) using `yt-dlp`.
```
sudo apt update
sudo apt install -y ffmpeg
uv pip install -U yt-dlp[default]
yt-dlp -f 140 dwE0qMxCpbE -o audio.m4a
ffmpeg -i audio.m4a audio.wav; rm audio.m4a
```

Now we can run our diarization script.
```
python diarize.py
```

This will save the diarization result in a file called `segments.json`. We can dump the contents:
```
cat segments.json
```

Done!

### Notes on Memory Usage
The script loads the audio into GPU memory for faster processing. This means your GPU must have enough VRAM to fit in the audio. The amount of memory needed (in MB) is about: 
```
0.828822055 * <audio_duration_in_seconds>
```
So for an hour long `.wav` file, ~3 GB VRAM is needed.

**Additionally**, main memory (RAM) is also required, roughly:
```
0.55283525509 * <audio_duration_in_seconds>
```
So for an hour long `.wav` file, ~2 GB RAM is needed.
