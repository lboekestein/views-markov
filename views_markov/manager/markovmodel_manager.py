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
        

    @staticmethod
    def _get_standardized_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the DataFrame based on the run type

        Args:
            df: The DataFrame to standardize

        Returns:
            The standardized DataFrame
        """

        def standardize_value(value):
            # 1) Replace inf, -inf, nan with 0; 
            # 2) Replace negative values with 0
            if isinstance(value, list):
                return [0 if (v == np.inf or v == -np.inf or v < 0 or np.isnan(v)) else v for v in value]
            else:
                return 0 if (value == np.inf or value == -np.inf or value < 0 or np.isnan(value)) else value

        df = df.applymap(standardize_value)

        return df
    

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
        """
        Evaluate the model artifact based on the evaluation type and the artifact name.

        Args:
            eval_type: The evaluation type
            artifact_name: The name of the artifact to evaluate

        Returns:
            A list of DataFrames containing the evaluation results
        """
        path_artifacts = self._model_path.artifacts
        run_type = self.configs["run_type"]

        # if an artifact name is provided through the CLI, use it.
        # Otherwise, get the latest model artifact based on the run type
        if artifact_name:
            logger.info(f"Using (non-default) artifact: {artifact_name}")

            if not artifact_name.endswith(".pkl"):
                artifact_name += ".pkl"
            path_artifact = path_artifacts / artifact_name
        else:
            # use the latest model artifact based on the run type
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)
            logger.info(
                f"Using latest (default) run type ({run_type}) specific artifact {path_artifact.name}"
            )

        self.configs = {"timestamp": path_artifact.stem[-15:]}

        try:
            with open(path_artifact, "rb") as f:
                markov_model = pickle.load(f)
        except FileNotFoundError:
            logger.exception(f"Model artifact not found at {path_artifact}")
            raise
        
        df_predictions = markov_model.predict(run_type, eval_type)
        df_predictions = [
            MarkovModelManager._get_standardized_df(df) for df in df_predictions
        ]
        return df_predictions


    def _forecast_model_artifact(self, artifact_name: str) -> pd.DataFrame:
        """
        Forecast using the model artifact.

        Args:
            artifact_name: The name of the artifact to use for forecasting

        Returns:
            The forecasted DataFrame
        """
        path_artifacts = self._model_path.artifacts
        run_type = self.configs["run_type"]

        # if an artifact name is provided through the CLI, use it.
        # Otherwise, get the latest model artifact based on the run type
        if artifact_name:
            logger.info(f"Using (non-default) artifact: {artifact_name}")

            if not artifact_name.endswith(".pkl"):
                artifact_name += ".pkl"
            path_artifact = path_artifacts / artifact_name
        else:
            # use the latest model artifact based on the run type
            path_artifact = self._model_path.get_latest_model_artifact_path(run_type)
            logger.info(
                f"Using latest (default) run type ({run_type}) specific artifact {path_artifact.name}"
            )
            
        self.configs = {"timestamp": path_artifact.stem[-15:]}

        try:
            with open(path_artifact, "rb") as f:
                markov_model = pickle.load(f)
        except FileNotFoundError:
            logger.exception(f"Model artifact not found at {path_artifact}")
            raise

        df_prediction = markov_model.predict(run_type)
        df_prediction = MarkovModelManager._get_standardized_df(df_prediction)

        return df_prediction


    def _evaluate_sweep(self, eval_type: str, model: any) -> List[pd.DataFrame]:
        run_type = self.configs["run_type"]

        df_predictions = model.predict(run_type, eval_type)
        df_predictions = [
            MarkovModelManager._get_standardized_df(df) for df in df_predictions
        ]

        return df_predictions