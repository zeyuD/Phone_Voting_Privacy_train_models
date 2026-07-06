# Phone_Voting_Privacy

## About

This repo is about the phone voting privacy project.



## Current Progress
### 
- Android app for video capturing while experiment, Zoom poll-like layout.
- Face landmark and eye rotation extraction from video.
- Transformer-based model to infere vote.


## Arranged To Do List
- Optical flow-based algorithm.


## To Be Implemented
- Additional sensing modality.
- Additional voting devices and layouts


## Tips 


### Environment pre-requests

Download the submodule contents
```
git submodule update --init --recursive
```

Install mvts Transfomer package
```bash
pip install -r mvts_transformer_M/failsafe_requirements.txt
```

Create an experiments folder
```bash
mkdir experiments
```


### Pull videos from experiment Android device

Data saved directory
```bash
cd /sdcard/Android/data/com.example.phone_vote/files/data/username/
```

Pull data from phone
```bash
adb pull /sdcard/Android/data/com.example.phone_vote/files/data/username/ "D:\OneDrive - Southern Methodist University\Phone_Privacy\data\phone_s22"
```



### Preprocess videos

Downsample (to 480p)
```bash
python data_preprocess/video_downsample_480p_5q.py
```

Extract landmarks from the downsampled videos
```bash
python data_preprocess/video_2_landmarks_480p_5q.py
```

Extract eye rotation from the downsampled videos
```bash
python extract_eye_rotation/run_subprocess_phone_480p.py
```

Extract eye feature from landmarks
```
# To be added MATLAB scripts
```



### Remove files cached that are in .gitignore

```bash
git rm -r --cached .
git add .

```