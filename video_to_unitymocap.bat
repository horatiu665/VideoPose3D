REM ONLY ONE VIDEO should be in video_input/ folder
REM but you have to give it the name of the video here at set myvideo="NAME"
REM and the output file will be %output%.json
REM
REM you can also add the --rendervideo "yes"  argument, to get that nice video with a 2d-3d side by side, it will be exported to %output%.mp4
REM
REM you can also choose to skip the first (most time-consuming) part of 2d inference, by specifying --steps "45" instead of "245". See the video_to_unitymocap.py file for details why that works.

set myvideo="tommywebcam.mp4"
set output="output_tomtom"

python video_to_unitymocap.py ^
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml ^
    --output-dir "video_output" ^
    --image-ext mp4 ^
    --im_or_folder "video_input" ^
    ^
    -i "video_output" ^
    -o "%myvideo%" ^
    --dataoutputdir "data" ^
    ^
    -d custom ^
    --keypoints "%myvideo%" ^
    -arc 3,3,3,3,3 ^
    -c checkpoint ^
    --evaluate pretrained_h36m_detectron_coco.bin ^
    --render ^
    --viz-subject "%myvideo%" ^
    --viz-action custom ^
    --viz-camera 0 ^
    --viz-video "video_input/%myvideo%" ^
    --viz-output "%output%".mp4 ^
    --viz-size 6 ^
    --output_json "%output%".json ^
    --steps "245"
REM    --rendervideo "yes" ^


if 1==0 (
    echo this here is just to help me copy paste stuff in pycharm
    python video_to_unitymocap.py 
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml 
    --output-dir "video_output" 
    --image-ext mp4 
    --im_or_folder "video_input" 
    
    -i "video_output" 
    -o "maiamakhateli.mp4" 
    --dataoutputdir "data" 
    
    -d custom 
    --keypoints "maiamakhateli.mp4" 
    -arc 3,3,3,3,3 
    -c checkpoint 
    --evaluate pretrained_h36m_detectron_coco.bin 
    --render 
    --viz-subject "maiamakhateli.mp4" 
    --viz-action custom 
    --viz-camera 0 
    --viz-video "video_input/maiamakhateli.mp4" 
    --viz-output "output_maia.mp4"
    --viz-size 6 
    --output_json "output_maia.json"
    --rendervideo "yes" 
    --steps "245"

)
