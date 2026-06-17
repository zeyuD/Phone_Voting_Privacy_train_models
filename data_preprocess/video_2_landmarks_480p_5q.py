# Jan 29, 2024. Created
# Feb 26, 2024. Count number of videos instead of hardcoding
# July 29, 2025. Updated setups and added directory configurations

import pandas as pd
import mediapipe as mp
import os
import time
import threading
import cv2
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions')))
from load_machine_config import load_machine_config

config = load_machine_config()

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
votes = ["A", "B", "C", "D", "E"]
# votes = ["A"]

# Create a dictionary of setups, each will have different setup and users
setups = {
    # "phone_s22": {
    #     # "users": ["Chuan", "Gujing", "Haofan", "Jimmy", "Jingwei", "Junwei", "Minjie", "minglei", "Mingxuan", "Rosie", "Sihang", "Wen", "Yirui", "Zeyu", "Zidan", "Ziyue", "Ziyue1"],
    #     # "users": ["JingweiZoom", "mingleiZoom", "MinjieZoom", "WeiZoom", "WenZoom"], # "ZeyuZoom"
    #     "users": ["JingweiObj", "ZeyuObj"],
    #     "save_dir": "phone_s22/"
    #       },
    "pad_op2": {
        "users": ["JingweiPad", "ZeyuPad"],
        "save_dir": "pad_op2/"
          }
}

for setup, details in setups.items():
    # Create save directory if not exist
    proj_path = os.path.join(config["data_dir"], "Phone_Privacy")
    vid_directory = proj_path + "/data/" + setup + "/"
    save_directory = proj_path + "/data/" + setup + "/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    users = details["users"]
    # print("Processing setup:", setup)
    # print("Users:", users)

    for v in range(len(votes)):
        vote = votes[v]
        # Create save vote directory if not exist
        save_vote_directory = save_directory + vote + "/"
        if not os.path.exists(save_vote_directory):
            os.makedirs(save_vote_directory)

        for u in range(len(users)):
            user = users[u]
            # Create save user directory if not exist
            save_user_directory = save_vote_directory + "/downsample_480p/"
            if not os.path.exists(save_user_directory):
                os.makedirs(save_user_directory)

            # Read folder and count the number of instances that contain "record"
            files_in_folder = os.listdir(vid_directory + vote + "/downsample_480p/")
            num_instances = 0
            vid_file_list = []
            for f in files_in_folder:
                if "downsample_480p" in f and f.startswith(user):
                    vid_file_list.append(f)
                    num_instances += 1
                    # print("Found instance:", f)

            for i in range(num_instances):
                print("Processing:", user, vote, str(i+1)+"/"+str(num_instances))
                vid_filename = vid_file_list[i]
                data_filename = vid_filename.replace("downsample_480p", "landmarks_480p").replace(".mp4", ".csv")

                # Open a video file and use Mediapipe to detect landmarks
                video_name = vid_directory + vote + "/downsample_480p/" + vid_filename
                cap = cv2.VideoCapture(video_name)

                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                frame_rate = int(cap.get(5))
                num_frames = int(cap.get(7))
                print("Actual resolution:", frame_width, frame_height)
                print("Actual FPS:", frame_rate)
                # print("Total frames:", num_frames)

                # Create a header for csv file
                # colomn names are timestamp, 0_x, 0_y, 0_z, 1_x, 1_y, 1_z, ..., 477_x, 477_y, 477_z
                csvheader = ['timestamp']
                for l in range(478):
                    csvheader.append(str(l)+'_x')
                    csvheader.append(str(l)+'_y')
                    csvheader.append(str(l)+'_z')
                total_array = pd.DataFrame(columns=csvheader)

                # # Plot 1 frame
                # success, image = cap.read()
                # # cv2.imshow("Frame", image)
                # # cv2.waitKey(0)
                # # Crop to the face region
                # image = image[160:560, 340:940]
                # # cv2.imshow("Frame", image)
                # # cv2.waitKey(0)

                frame_count = 0
                # Plot 1 frame with landmarks
                # with mp_face_mesh.FaceMesh(
                #     max_num_faces=1,
                #     refine_landmarks=True,
                #     min_detection_confidence=0.5,
                #     min_tracking_confidence=0.5) as face_mesh:

                #! If raw data is not available
                # csvheader = ['timestamp']
                # for l in range(478):
                #     csvheader.append(str(l)+'_x')
                #     csvheader.append(str(l)+'_y')
                #     csvheader.append(str(l)+'_z')
                # total_array = pd.DataFrame(columns=csvheader)
                
                with mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                        
                    while cap.isOpened():
                        success, image = cap.read()

                        if not success:
                            print("Video frames finished.")
                            # If loading a video, use 'break' instead of 'continue'.
                            break

                        # # Show a frame and its shape
                        # import numpy as np
                        # print("Frame shape:", np.shape(image))
                        # cv2.imshow("Frame", image)
                        # cv2.waitKey(0)

                        #! Get from raw data
                        frame_timestamp_ms = time.time()
                        frame_count += 1
                        # if frame_count % 2 == 0:
                        #     continue

                        # print("image shape:", image.shape)
                        # if frame_count % 2 == 0:
                        #     continue
                        print("Frames:", str(frame_count)+"/" + str(num_frames), end="\r")
                            
                        if frame_count == num_frames:
                            print()
                            # print("Finished")

                        # To improve performance, optionally mark the image as not writeable to
                        # pass by reference.
                        # image = image[160:560, 440:840]
                        image.flags.writeable = False
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(image)

                        # # Draw the face mesh annotations on the image.
                        # image.flags.writeable = True
                        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        if results.multi_face_landmarks:
                            # print(len(results.multi_face_landmarks), "face(s) detected at time:", time.time())
                            for face_landmarks in results.multi_face_landmarks:
                                # save landmarks
                                new_row = [frame_timestamp_ms]
                                for idx, landmark in enumerate(face_landmarks.landmark):
                                    new_row.append(landmark.x)
                                    new_row.append(landmark.y)
                                    new_row.append(landmark.z)
                                total_array.loc[len(total_array)] = new_row
                                # counter += 1
                                # counter += 1
                                # print(counter)
                                # if counter % 100 == 0:
                        else:
                            print()
                            print("No face detected.")

                        # if cv2.waitKey(5) & 0xFF == 27:
                        #     # time_step = total_array.time_step
                        #     # time_diff = time_step.diff()
                        #     # FPS = 1/time_diff
                        #     # print('FPS:',FPS.mean())
                        #     break
                                
                cap.release()
                cv2.destroyAllWindows()
                                
                save_path_csv = save_user_directory + data_filename
                total_array.to_csv(save_path_csv, index=False)
                print("Saved:", save_path_csv)
                print()