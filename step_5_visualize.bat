python run.py ^
    -d custom ^
    -k video_output ^
    -arc 3,3,3,3,3 ^
    -c checkpoint ^
    --evaluate pretrained_h36m_detectron_coco.bin ^
    --render ^
    --viz-subject tommywebcam.mp4 ^
    --viz-action custom --viz-camera 0 --viz-video video_input/tommywebcam.mp4 ^
    --viz-output output.mp4 ^
    --viz-size 6
