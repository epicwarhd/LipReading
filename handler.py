import torch
import logging
import os
from zipfile import ZipFile
from configparser import ConfigParser

from ts.torch_handler.base_handler import BaseHandler


logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):

    def initialize(self, context):
        """Initialize function loads the model and the tokenizer

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """

        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        self.extra_dir = os.path.join(model_dir, 'extra_dir.zip')
        self.config_file = os.path.join(model_dir, 'config_file.ini')

        with ZipFile(self.extra_dir) as zObject:
            zObject.extractall(path=model_dir)

        from pipelines.data.data_module import AVSRDataLoader
        from pipelines.model import AVSR
        from pipelines.detectors.mediapipe.detector import LandmarksDetector

        assert os.path.isfile(
            self.config_file), f"config_filename: {self.config_file} does not exist."

        config = ConfigParser()
        config.read(self.config_file)
        self.input_v_fps = config.getfloat("input", "v_fps")
        self.model_v_fps = config.getfloat("model", "v_fps")
        self.modality = config.get("input", "modality")

        self.dataloader = AVSRDataLoader(
            modality=self.modality, speed_rate=self.input_v_fps/self.model_v_fps, detector='mediapipe')

        self.model = AVSR(config=config, device=self.device).to(
            device=self.device)

        self.landmarks_detector = LandmarksDetector()

        logger.info(f'Successfully loaded model from ahihi')
        logger.info(f'Properties: {properties}')
        logger.info(f'Manifest: {self.manifest}')

        self.initialized = True

    def preprocess(self, data):
        input_data = data[0]['data'].decode()
        logger.info(f'Input: {input_data} and Type:{type(input_data)}')

        assert os.path.isfile(
            input_data), f"data_filename: {input_data} does not exist."

        landmarks = self.landmarks_detector(input_data)

        data_prepare = self.dataloader.load_data(input_data, landmarks)

        return data_prepare

    def inference(self, data):
        output = self.model.infer(data)

        return output

    def postprocess(self, data):
        logger.info(f'Output: {data}')
        return [data]
