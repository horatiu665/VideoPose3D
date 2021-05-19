import cv2
import numpy as np
import torch
import json

import socket_warionette

import inference.infer_video_d2 as infer

import data.data_utils as data_utils
from common.camera import normalize_screen_coordinates, camera_to_world, image_coordinates
from common.custom_dataset import FakeCustomDataset
from common.generators import UnchunkedGenerator
from common.h36m_dataset import h36m_skeleton
from common.loss import mpjpe, n_mpjpe, p_mpjpe, mean_velocity_error
from common.model import TemporalModelOptimized1f, TemporalModel

# visualization
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# end visualization

def init_predictor(frame):

    h = {}

    # this is the part from step2: infer_video_d2
    infer.setup_logger()
    h['predictor'] = infer.prepare_predictor(args_cfg="COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")

    h['frame_metadata'] = {
        'w': frame.shape[1],
        'h': frame.shape[0],
    }

    h['metadata'] = data_utils.suggest_metadata('coco')
    h['metadata']['video_metadata'] = {}
    # 'frame' was a video name, it could be multiple videos saved here. but we just gonna use 'frame'
    # video_metadata is the metadata from step 2, which is just the frame size.
    h['metadata']['video_metadata']['frame'] = h['frame_metadata']

    # dataset setup. we have to pretend like we are loading a dataset, because the fb boys are adding data upon load
    # we cannot load from a file... so we have to dissect this function.
    # dataset = CustomDataset('data/data_2d_' + "custom" + '_' + args.keypoints + '.npz')
    # we are gonna use the keypoints and metadata arrays, and not bother with the datasets.
    # and hopefully we figure out which parts to skip and which are necessary to operate on our frame data this way.

    # fake import of skeleton and some other stuff that is initialized in dataset and we skip it
    dataset = FakeCustomDataset(h['metadata'])
    h['dataset'] = dataset
    h['keypoints_metadata'] = h['metadata']
    keypoints_symmetry = h['keypoints_metadata']['keypoints_symmetry']
    h['keypoints_symmetry'] = keypoints_symmetry
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

    h['kps_left'] = kps_left
    h['kps_right'] = kps_right
    h['joints_left'] = joints_left
    h['joints_right'] = joints_right

    return h

def analyze_frame(h, frame):

    boxes, keypoints = infer.inference_on_frame(h['predictor'], frame)

    # step 4: prepare data.
    # take 2d keypoints, that's it
    # first element is empty array, second is our actual frame data, a 3d numpy array with first dimension 1, second and third being the 17 joints of 3 doubles each.
    kp = keypoints[1][0][:2, :].T  # extract (x, y) just like in prepare_data_2d_custom code

    # what to do if kp is NaN or missing data or something?
    # I guess just ignore it

    # they do this  at the end of step4. but we keep it simple, and take the data from step2 directly into a variable.
    #     output[canonical_name]['custom'] = [data[0]['keypoints'].astype('float32')]
    #output_custom_canonical_bullshit = kp.astype('float32')

    # this is what happens at  the end of step4. which is a file that is loaded in the beginning of step 5.
    #     np.savez_compressed(os.path.join(args.dataoutputdir, output_prefix_2d + args.output), positions_2d=output, metadata=metadata)

    # this is the bullshit they do in the original script.
    # confusingly, keypoints is actually just data, until it is set to keypoints[positions_2d]
    # keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True)

    # step 5: ..... all the other shit
    # starting to copy stuff over from run.py

    # extract dataset from the init dictionary
    dataset = h['dataset']
    keypoints_metadata = h['keypoints_metadata']
    keypoints_symmetry = h['keypoints_symmetry']

    kps_left = h['kps_left']
    kps_right = h['kps_right']
    joints_left = h['joints_left']
    joints_right = h['joints_right']

    # normalize
    for i in range(len(kp)):
        koord = kp[i]
        kp[i] = normalize_screen_coordinates(koord, h['frame_metadata']['w'], h['frame_metadata']['h'])
    #for kps in enumerate(keypoints):
    #    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], frame_metadata['w'], frame_metadata['h'])

    # this is taken from the args.architecture and run.py and just hardcoded, skipping a lot of nonsense
    filter_widths = [int(x) for x in "3,3,3,3,3".split(',')]
    skeleton_num_joints = dataset.skeleton().num_joints()
    #skeleton_num_joints = 17

    causal = True
    dropout = 0.25
    channels = 1024
    dense = False

    model_pos_train = TemporalModelOptimized1f(kp.shape[-2], kp.shape[-1], skeleton_num_joints,
                                               filter_widths=filter_widths, causal=causal, dropout=dropout,
                                               channels=channels)
    model_pos = TemporalModel(kp.shape[-2], kp.shape[-1], skeleton_num_joints,
                                         filter_widths=filter_widths, causal=causal, dropout=dropout,
                                         channels=channels, dense=dense)

    receptive_field = model_pos.receptive_field()
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pad = (receptive_field - 1) // 2  # Padding on each side
    #if args.causal:
    #    print('INFO: Using causal convolutions')
    #    causal_shift = pad
    #else:
    #    causal_shift = 0
    causal_shift = pad

    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
        model_pos_train = model_pos_train.cuda()

    #if args.resume or args.evaluate:
    if True:
        chk_filename = "checkpoint/pretrained_h36m_detectron_coco.bin"
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))
        model_pos_train.load_state_dict(checkpoint['model_pos'])
        model_pos.load_state_dict(checkpoint['model_pos'])

        # false in our particular case... we might benefit from getting rid of model_traj,
        # unless it's super fast then we should just keep it in case we ever upgrade
        if 'model_traj' in checkpoint:
            # Load trajectory model if it contained in the checkpoint (e.g. for inference in the wild)
            model_traj = TemporalModel(kp.shape[-2], kp.shape[-1], 1,
                                filter_widths=filter_widths, causal=causal, dropout=dropout, channels=channels,
                                dense=dense)
            if torch.cuda.is_available():
                model_traj = model_traj.cuda()
            model_traj.load_state_dict(checkpoint['model_traj'])
        else:
            model_traj = None

    test_generator = UnchunkedGenerator(None, None, kp,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=kps_left, kps_right=kps_right,
                                        joints_left=joints_left, joints_right=joints_right)
    print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

    # Evaluate
    def evaluate(eval_generator, action=None, return_predictions=False, use_trajectory_model=False):
        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_pos_scale = 0
        epoch_loss_3d_vel = 0
        with torch.no_grad():
            if not use_trajectory_model:
                model_pos.eval()
            else:
                model_traj.eval()
            N = 0
            for _, batch, batch_2d in eval_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()

                # Positional model
                if not use_trajectory_model:
                    predicted_3d_pos = model_pos(inputs_2d)
                else:
                    predicted_3d_pos = model_traj(inputs_2d)

                # Test-time augmentation (if enabled)
                if eval_generator.augment_enabled():
                    # Undo flipping and take average with non-flipped version
                    predicted_3d_pos[1, :, :, 0] *= -1
                    if not use_trajectory_model:
                        predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                    predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

                if return_predictions:
                    return predicted_3d_pos.squeeze(0).cpu().numpy()

                inputs_3d = torch.from_numpy(batch.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                inputs_3d[:, :, 0] = 0
                if eval_generator.augment_enabled():
                    inputs_3d = inputs_3d[:1]

                error = mpjpe(predicted_3d_pos, inputs_3d)
                epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

                epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]

                inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

                epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

                # Compute velocity error
                epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

        if action is None:
            print('----------')
        else:
            print('----'+action+'----')
        e1 = (epoch_loss_3d_pos / N)*1000
        e2 = (epoch_loss_3d_pos_procrustes / N)*1000
        e3 = (epoch_loss_3d_pos_scale / N)*1000
        ev = (epoch_loss_3d_vel / N)*1000
        print('Test time augmentation:', eval_generator.augment_enabled())
        print('Protocol #1 Error (MPJPE):', e1, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
        print('Velocity Error (MPJVE):', ev, 'mm')
        print('----------')

        return e1, e2, e3, ev

    image_keypoints2d = kp
    gen = UnchunkedGenerator(None, None, [[image_keypoints2d]],
                             pad=pad, causal_shift=causal_shift, augment=False,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, return_predictions=True)

    # here is the data format
    # public enum VideoPose3dJointOrder
    # {
    #     HIP = 0,
    #     R_HIP = 1,
    #     R_KNEE = 2,
    #     R_FOOT = 3,
    #     L_HIP = 4,
    #     L_KNEE = 5,
    #     L_FOOT = 6,
    #     SPINE = 7,
    #     THORAX = 8,
    #     NOSE = 9,
    #     HEAD = 10,
    #     L_SHOULDER = 11,
    #     L_ELBOW = 12,
    #     L_WRIST = 13,
    #     R_SHOULDER = 14,
    #     R_ELBOW = 15,
    #     R_WRIST = 16
    # }

    # this bugs out. dunno what the hell they were trying to do.
    # anyway we can fix it by just getting width/height some other way.

    # Invert camera transformation
    cam = dataset.cameras()

    width = cam['frame'][0]['res_w']
    height = cam['frame'][0]['res_h']

    image_keypoints2d = image_coordinates(image_keypoints2d[..., :2], w=width, h=height)

    viz_camera = 0

    # If the ground truth is not available, take the camera extrinsic params from a random subject.
    # They are almost the same, and anyway, we only need this for visualization purposes.
    for subject in dataset.cameras():
        if 'orientation' in dataset.cameras()[subject][viz_camera]:
            rot = dataset.cameras()[subject][viz_camera]['orientation']
            break
    prediction = camera_to_world(prediction, R=rot, t=0)
    # We don't have the trajectory, but at least we can rebase the height
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    # because algo was meant for a list of frames, we take the first frame (our only frame)
    prediction3d = prediction[0]

    return prediction3d, image_keypoints2d

    # do we want to visualize? this code used to write to json and create a video for visualization
    #if args.viz_output is not None:
    if True:

        anim_output = {'Reconstruction': prediction}

        # format the data in the same format as mediapipe, so we can load it in unity with the same script
        # we need a list (frames) of lists of 3d landmarks.
        unity_landmarks = prediction.tolist()

        # how to send data? or display it?
        # maybe draw it on the webcam feed....?!?!?!


        #with open(args.output_json, "w") as json_file:
        #    json.dump(unity_landmarks, json_file)

        #if args.rendervideo == "yes":
        #    from common.visualization import render_animation
        #    render_animation(input_keypoints, keypoints_metadata, anim_output,
        #                     dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
        #                     limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
        #                     input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
        #                     input_video_skip=args.viz_skip)

    we_re_done_here = 1


def main_loop():
    initialized = False
    h = {}  # this is where we keep all the good stuff

    # connect to socket where we will send the positional data
    socket = socket_warionette.WarionetteSocket()

    # start webcam
    video = cv2.VideoCapture(0)
    window_name = "Warionette"

    while True:
        # get frame from the webcam/videocapture
        check, frame = video.read()

        # only run this once. initialize the detectron2 predictor
        if not initialized:
            initialized = True
            h = init_predictor(frame)

        # process frame?
        # monochrome? change resolution?????!??!?!?!?!?! what else could we do?

        # run the frame through detectron2 and through VideoPose3D to get 2d and 3d body tracking points.
        pos3d, pos2d = analyze_frame(h, frame)

        # draw 2d on the image
        for p in pos2d:
            pos = (int(p[0]), int(p[1]))
            cv2.circle(frame, pos, 1, (0, 255, 0), -1)

        # draw 3d on the image. doesn't seem trivial so I skip it.
        if False:
            linethicc = 1
            skeleton = h['dataset'].skeleton()
            parents = skeleton.parents()
            children = skeleton.children()
            lines3d = []  # format: ((x, y, z), (x, y, z)) all floats
            for i, p in enumerate(pos3d):
                for child in children[i]:
                    pos2 = pos3d[child]
                    lines3d.append((p, pos2))

        # send points as packets on the socket.
        pos3d_bytes = pos3d.tobytes()
        socket.send_bytes(pos3d_bytes)

        # show the frame
        cv2.imshow(window_name, frame)


        # wait 1 millisec for a key
        key = cv2.waitKey(1)
        if key == 27:  # esc
            break
        # window was closed with x button because we cannot detect some property of it.
        elif cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 1:
            break

    # stop webcam and visualization.
    video.release()
    cv2.destroyAllWindows()

    # close network socket
    socket.close()


if __name__ == '__main__':
    main_loop()
