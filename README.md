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


## Tips 


### Environment pre-requests

Install mvts Transfomer package
```bash
pip install -r failsafe_requirements.txt
```

Create an experiments folder
```bash
mkdir experiments
```


### Pull videos from experiment Android phone

Data saved directory
```bash
cd /sdcard/Android/data/com.example.phone_vote/files/data/test/
```

Pull data from phone
```bash
adb pull /sdcard/Android/data/com.example.phone_vote/files/data/test/ "D:\OneDrive - Southern Methodist University\Phone_Privacy\data\phone_s22"
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
python run_subprocess_phone.py
```



### Remove files cached that are in .gitignore

```bash
git rm -r --cached .
git add .

```