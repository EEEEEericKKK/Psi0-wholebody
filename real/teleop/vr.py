import os
import sys
import time
from multiprocessing import Event, Lock, Manager, Process, Queue, shared_memory
from pathlib import Path

import numpy as np
import yaml
from teleop.robot_control.dex_retargeting.retargeting_config import RetargetingConfig

from teleop.constants_vuer import tip_indices
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from TeleVision import OpenTeleVision

import threading
import zmq

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


########################
# INPUT YOUR HEIGHT HERE!
########################
# TELEOPERATOR_HEIGHT = 1.73

from teleop.constants_vuer import (
    T_robot_openxr,
    T_to_unitree_hand,
    grd_yup2grd_zup,
    hand2inspire,
    hand2inspire_l_arm,
    hand2inspire_l_finger,
    hand2inspire_r_arm,
    hand2inspire_r_finger,
)
from motion_utils import fast_mat_inv, mat_update

# height_offset = 1.88 - TELEOPERATOR_HEIGHT


class VuerPreprocessor:
    def __init__(self, manus_receiver=None):
        self.manus_receiver = manus_receiver

        self.vuer_head_mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, -0.2], [0, 0, 0, 1]]
        )
        self.vuer_right_wrist_mat = np.array(
            [[1, 0, 0, 0.5], [0, 1, 0, 1], [0, 0, 1, -0.5], [0, 0, 0, 1]]
        )

        self.vuer_left_wrist_mat = np.array(
            [[1, 0, 0, -0.5], [0, 1, 0, 1], [0, 0, 1, -0.5], [0, 0, 0, 1]]
        )

        self.y_offset = None
        self.target_height = 1.82
        self.calibration_enabled = False

    def trigger_calibration(self):
        self.y_offset = None
        self.calibration_enabled = True

    def process(self, tv):
        head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())

        if self.calibration_enabled and head_mat is not None and self.y_offset is None:
            current_y = head_mat[1, 3]
            if not np.allclose(current_y, 0):
                self.y_offset = self.target_height - current_y
                print(f"Height Calibrated! Head Y: {current_y:.3f}, Offset: {self.y_offset:.3f}")

        height_offset = self.y_offset if self.y_offset is not None else 0.0

        if head_mat is not None:
            head_mat[1, 3] += height_offset
        

        self.vuer_head_mat = head_mat

        self.vuer_right_wrist_mat = mat_update(
            self.vuer_right_wrist_mat, tv.right_hand.copy()
        )
        self.vuer_right_wrist_mat[1, 3] += height_offset

        self.vuer_left_wrist_mat = mat_update(
            self.vuer_left_wrist_mat, tv.left_hand.copy()
        )
        self.vuer_left_wrist_mat[1, 3] += height_offset

        left_landmarks = tv.left_landmarks.copy()
        right_landmarks = tv.right_landmarks.copy()
        left_landmarks[:, 1] += height_offset
        right_landmarks[:, 1] += height_offset
        

        # change of basis
        head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
        right_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )
        left_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )

        rel_left_wrist_mat = (
            fast_mat_inv(head_mat) @ left_wrist_mat @ hand2inspire_l_arm
        )

        rel_right_wrist_mat = (
            fast_mat_inv(head_mat) @ right_wrist_mat @ hand2inspire_r_arm
        )  # wTr = wTh @ hTr

        # homogeneous
        left_hand_vuer_mat = np.concatenate(
            [left_landmarks.copy().T, np.ones((1, left_landmarks.shape[0]))]
        )
        right_hand_vuer_mat = np.concatenate(
            [right_landmarks.copy().T, np.ones((1, right_landmarks.shape[0]))]
        )

        # change of basis
        left_hand_mat = T_robot_openxr @ left_hand_vuer_mat
        right_hand_mat = T_robot_openxr @ right_hand_vuer_mat

        left_hand_mat_wb = fast_mat_inv(left_wrist_mat) @ left_hand_mat
        right_hand_mat_wb = fast_mat_inv(right_wrist_mat) @ right_hand_mat

        unitree_left_hand = (T_to_unitree_hand @ left_hand_mat_wb)[0:3, :].T
        unitree_right_hand = (T_to_unitree_hand @ right_hand_mat_wb)[0:3, :].T

        unitree_tip_indices = [4, 9, 14]  # [thumb, index, middle] in OpenXR


        # Check if hand data is initialized
        left_q_target, right_q_target = None, None
        hand_retargeting = HandRetargeting(
            HandType.UNITREE_DEX3
        )  # TODO: add if to distinguish hand
        # hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)
        if not np.all(left_hand_mat == 0.0):
            # Extract the relevant tip indices (assumed defined elsewhere)
            ref_left_value = unitree_left_hand[unitree_tip_indices].copy()
            ref_right_value = unitree_right_hand[unitree_tip_indices].copy()

            # Apply scaling factors to calibrate the values
            ref_left_value[0] *= 1.15
            ref_left_value[1] *= 1.05
            ref_left_value[2] *= 0.95

            ref_right_value[0] *= 1.15
            ref_right_value[1] *= 1.05
            ref_right_value[2] *= 0.95

            # Use the retargeting methods to convert reference values to qpos.
            left_q_target = hand_retargeting.left_retargeting.retarget(ref_left_value)[
                hand_retargeting.right_dex_retargeting_to_hardware
            ]
            right_q_target = hand_retargeting.right_retargeting.retarget(
                ref_right_value
            )[hand_retargeting.right_dex_retargeting_to_hardware]

        return (
            head_mat,
            rel_left_wrist_mat,
            rel_right_wrist_mat,
            left_q_target,
            right_q_target,
        )

    def get_hand_gesture(self, tv):
        self.vuer_right_wrist_mat = mat_update(
            self.vuer_right_wrist_mat, tv.right_hand.copy()
        )
        self.vuer_left_wrist_mat = mat_update(
            self.vuer_left_wrist_mat, tv.left_hand.copy()
        )

        # change of basis
        right_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )
        left_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )

        left_fingers = np.concatenate(
            [tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))]
        )
        right_fingers = np.concatenate(
            [tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))]
        )

        # change of basis
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        rel_left_fingers = (hand2inspire_l_finger.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (hand2inspire_r_finger.T @ rel_right_fingers)[0:3, :].T
        all_fingers = np.concatenate([rel_left_fingers, rel_right_fingers], axis=0)

        return all_fingers


class VuerTeleop:
    def __init__(self, img_shm_name):
        # self.resolution = (720,1280) #(480,640)
        self.resolution = (720, 640)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (
            self.resolution[0] - self.crop_size_h,
            self.resolution[1] - 2 * self.crop_size_w,
        )

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]
        
        if img_shm_name is None:
            self.shm = shared_memory.SharedMemory(
                create=True, size=np.prod(self.img_shape) * np.uint8().itemsize
            )
        else:
            self.shm = shared_memory.SharedMemory(name=img_shm_name)
            
        self.img_array = np.ndarray(
            (self.img_shape[0], self.img_shape[1], 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(
            self.resolution_cropped, self.shm.name, image_queue, toggle_streaming
        )
        self.processor = VuerPreprocessor()


    def step(self, full_head=False):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = (
            self.processor.process(self.tv)
        )
        if full_head:
            head_rmat = head_mat
        else:
            head_rmat = head_mat[:3, :3]

        left_wrist_mat[2, 3] += 0.55
        right_wrist_mat[2, 3] += 0.55
        left_wrist_mat[0, 3] += 0.05
        right_wrist_mat[0, 3] += 0.05

        return head_rmat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat

    def shutdown(self):
        # self.shm.close()
        # self.shm.unlink()
        self.tv.shutdown()
