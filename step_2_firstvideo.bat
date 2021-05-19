cd inference
python infer_video_d2.py ^
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml ^
    --output-dir "..\video_output" ^
    --image-ext mp4 ^
    --im_or_folder "..\video_input"
cd ..\