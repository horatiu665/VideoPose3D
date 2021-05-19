:: ONLY ONE VIDEO should be in video_input/ folder

set myvideo="maiamakhateli.mp4"
set output="output_maia"

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
    --rendervideo "yes" ^
    --steps "245"


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
