import unittest
import cv2
from AlphaPose.PoseEstimator import PoseEstimator
import json
import torch

class TestSum(unittest.TestCase):
    def test_image0(self):
        image_number = 0
        pose_estimator = PoseEstimator()
        image_dir = "tests/data/images"
        batch_image_files = ["{}.png".format(image_number)]
        batch_images = [cv2.imread(image_dir+'/'+f) for f in batch_image_files]
        batch_result = pose_estimator(batch_images)
        expected_result = json.load(open(
            "tests/data/expected_results/{}.json".format(image_number)))
        for key in batch_result[0].keys():
            assert torch.allclose(
                        batch_result[0][key],
                        torch.tensor(expected_result[0][key]), atol=1e-1)

    def test_image1(self):
        image_number = 1
        pose_estimator = PoseEstimator()
        image_dir = "tests/data/images"
        batch_image_files = ["{}.png".format(image_number)]
        batch_images = [cv2.imread(image_dir+'/'+f) for f in batch_image_files]
        batch_result = pose_estimator(batch_images)
        expected_result = json.load(open(
            "tests/data/expected_results/{}.json".format(image_number)))
        for key in batch_result[0].keys():
            assert torch.allclose(
                        batch_result[0][key],
                        torch.tensor(expected_result[0][key]), atol=1e-1)

    def test_image2(self):
        image_number = 2
        pose_estimator = PoseEstimator()
        image_dir = "tests/data/images"
        batch_image_files = ["{}.png".format(image_number)]
        batch_images = [cv2.imread(image_dir+'/'+f) for f in batch_image_files]
        batch_result = pose_estimator(batch_images)
        expected_result = json.load(open(
            "tests/data/expected_results/{}.json".format(image_number)))
        for key in batch_result[0].keys():
            assert torch.allclose(
                        batch_result[0][key],
                        torch.tensor(expected_result[0][key]), atol=1e-1)
