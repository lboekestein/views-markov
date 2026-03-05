from views_pipeline_core.managers.model import ModelPathManager, ForecastingModelManager
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.files.utils import read_dataframe, generate_model_file_name
from views_markov.model.markov_model import MarkovModel
import logging
import pickle
import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class MarkovModelManager(ForecastingModelManager):
    """
    TODO: Add class docstring
    """
    def __init__(
        self,
        model_path: ModelPathManager,
        wandb_notifications: bool = False,
        use_prediction_store: bool = False,
    ) -> None:
        super().__init__(model_path, wandb_notifications, use_prediction_store)

        # TODO should not be hardcoded
        self._default_random_state = 42

        # raise error if level is not cm, since pgm is not implemented for MarkovModel
        if self._config_meta["level"] == "pgm":
            raise NotImplementedError("PGM level is not yet implemented for MarkovModel.")
        elif self._config_meta["level"] != "cm":
            raise ValueError(f"Invalid level: {self._config_meta['level']}. Expected 'cm' for MarkovModel.")
    

    def _get_model(self, partitioner_dict: dict) -> MarkovModel:
        """
        Get the Markov model based on the specified method in the configuration.

        Args:
            partitioner_dict: A dictionary containing partitioner information.

        Returns:
            An instance of the MarkovModel class.
        """

        model = MarkovModel(
            partitioner_dict=partitioner_dict,
            steps = self.config.get("steps", [*range(1, 36 + 1, 1)]),
            target = self.config.get("targets", ["ln_ged_sb_dep"])[0],
            markov_target=self.config.get("markov_target", "ln_ged_sb_dep"),
            markov_method=self.config.get("markov_method", "direct"),
            regression_method=self.config.get("regression_method", "single"),
            markov_threshold=self.config.get("markov_threshold", 0),
            random_state=self.config.get("random_state", self._default_random_state),
            n_jobs = self.config.get("n_jobs", -1),
            rf_class_params=self.config.get("rf_class_params", {}),
            rf_reg_params=self.config.get("rf_reg_params", {}),
        )

        return model

    def _train_model_artifact(self) -> MarkovModel:
        """
        Train the model and save it as an artifact if not a sweep.

        Returns:
            The trained model object
        """
        path_raw = self._model_path.data_raw
        path_artifacts = self._model_path.artifacts

        run_type = self.config["run_type"]
        df_viewser = read_dataframe(
            path_raw / f"{run_type}_viewser_df{PipelineConfig.dataframe_format}"
        )

        partitioner_dict = self._data_loader.partition_dict
        markov_model = self._get_model(partitioner_dict)
        markov_model.fit(df_viewser)

        if not self.config["sweep"]:
            model_filename = generate_model_file_name(
                run_type, file_extension=".pkl"
            )
            markov_model.save(path_artifacts / model_filename)

        return markov_model

    def _evaluate_model_artifact(
            self, eval_type: str, artifact_name: str
        ) -> List[pd.DataFrame]:
        ...

    def _forecast_model_artifact(self, artifact_name: str) -> pd.DataFrame:
        ...

    def _evaluate_sweep(self, eval_type: str, model: Any) -> List[pd.DataFrame]:
        ...