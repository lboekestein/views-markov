from views_pipeline_core.managers.model import ModelPathManager, ModelManager
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.files.utils import read_dataframe, generate_model_file_name
import logging
import pickle
import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict

logger = logging.getLogger(__name__)


class MarkovModelManager(ModelManager):
    def __init__(
        self,
        model_path: ModelPathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
    ) -> None:
        super().__init__(model_path, wandb_notifications, use_prediction_store)