from views_pipeline_core.managers.model import ModelPathManager, ModelManager
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.files.utils import read_dataframe, generate_model_file_name
from views_markov.model.markov_model import MarkovModel
import logging
import pickle
import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class MarkovModelManager(ModelManager):
    def __init__(
        self,
        model_path: ModelPathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
    ) -> None:
        super().__init__(model_path, wandb_notifications, use_prediction_store)

        # TODO is this the right way of retrieving configs?
        self._markov_method = self.configs["markov_method"]
        self._regression_method = self.configs["regression_method"]


    def _split_markov_parameters(self):
        """
        Split the parameters dictionary into two separate dictionaries, one for the
        markov classification model and one for the markov regression model.

        Returns:
            A dictionary containing original config, the split classification and regression parameters.
        """
        clf_dict = {}
        reg_dict = {}
        config = self.configs

        for key, value in config.items():
            if key.startswith("clf_"):
                clf_key = key.replace("clf_", "")
                clf_dict[clf_key] = value
            elif key.startswith("reg_"):
                reg_key = key.replace("reg_", "")
                reg_dict[reg_key] = value

        config["clf"] = clf_dict
        config["reg"] = reg_dict

        return config
    
    def _get_model(self, partitioner_dict: dict) -> MarkovModel:
        """
        Get the markov model based on the specified method in the configuration.

        Args:
            partitioner_dict: A dictionary containing partitioner information.

        Returns:
            An instance of the MarkovModel class.
        """
        config = self._split_markov_parameters()

        model = MarkovModel(
            partitioner_dict=partitioner_dict,
            markov_method=self._markov_method,
            regression_method=self._regression_method,
            markov_threshold=config.get("markov_threshold", 0),
            random_state=config.get("random_state", None),
            n_jobs = config.get("n_jobs", -1),
            rf_class_params=config.get("clf", {}),
            rf_reg_params=config.get("reg", {}),
        )

        return model

    def _train_model_artifact(self):
        ...

    def _evaluate_model_artifact(self):
        ...

    def _forecast_model_artifact(self):
        ...

    def _evaluate_sweep(self, eval_type: str, model: Any):
        ...