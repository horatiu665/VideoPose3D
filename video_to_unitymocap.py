# This file aims to merge 3 steps into one
# we will need to add all the args together for this one.
# first step
#
# cd inference/
# python infer_video_d2
#    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml ^
#    --output-dir "C:\PythonProjects\VideoPose3D-master\video_output" ^
#    --image-ext mp4 "C:\PythonProjects\VideoPose3D-master\video_input"
# cd ..
#
#
# second step
#
# cd data
# python "prepare_data_2d_custom.py" -i "..\video_output" -o video_output
# cd ..\
#
#
# third step
#
# python run.py ^
#     -d custom ^
#     -k video_output ^
#     -arc 3,3,3,3,3 ^
#     -c checkpoint ^
#     --evaluate pretrained_h36m_detectron_coco.bin ^
#     --render ^
#     --viz-subject tommywebcam.mp4 ^
#     --viz-action custom --viz-camera 0 --viz-video video_input/tommywebcam.mp4 ^
#     --viz-output output.mp4 ^
#     --viz-size 6
#
#
# merging all the arguments should look like:
# set myvideo="maiamakhateli.mp4"
# set output="output"
# python video_to_unitymocap.py ^
#     --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml ^
#     --output-dir "video_output" ^
#     --image-ext mp4 "video_input" ^
#     ^
#     -i "video_output" ^
#     -o video_output ^
#     ^
#     -d custom ^
#     -k video_output ^
#     -arc 3,3,3,3,3 ^
#     -c checkpoint ^
#     --evaluate pretrained_h36m_detectron_coco.bin ^
#     --render ^
#     --viz-subject "%myvideo%" ^
#     --viz-action custom ^
#     --viz-camera 0 ^
#     --viz-video "video_input/%myvideo%" ^
#     --viz-output "%output%".mp4 ^
#     --viz-size 6 ^
#     --output_json "%output%".json
#
#

import inference.infer_video_d2 as step2
import data.prepare_data_2d_custom as step4
import run as step5

if __name__ == "__main__":

    step2.setup_logger()

    # we parse the args only once so we can catch all of them.
    args = step5.parse_args()

    steps: str = args.steps

    if '2' in steps:
        step2.main(args)

    if '4' in steps:
        step4.the_main_thing(args)

    if '5' in steps:
        step5.the_main_kaboose(args)
