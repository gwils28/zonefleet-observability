"""Registry MLflow local : versions du forecaster, alias champion / challenger.

Le registry est un fichier SQLite local (`results/mlflow/mlflow.db`), suffisant pour
démontrer le cycle de vie sans serveur. En production, seule l'URI change.

Règles encodées ici :
- un modèle n'est enregistré qu'avec sa période d'entraînement, ses métriques de
  référence et l'empreinte du panel (traçabilité, Sculley « reproducibility debt ») ;
- la promotion d'un challenger exige un gain minimal sur une fenêtre de validation
  **postérieure** à son entraînement (`promote_if_better`) ;
- l'alias `champion` est la seule chose que le service de prévision lit.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path

from zonefleet.config import DATA_PROCESSED, RESULTS
from zonefleet.model import QuantileForecaster

MODEL_NAME = "zonefleet-demand"
TRACKING_DIR = RESULTS / "mlflow"


def tracking_uri(directory: Path | None = None) -> str:
    d = directory or TRACKING_DIR
    d.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(d / 'mlflow.db').as_posix()}"


def _mlflow(directory: Path | None = None):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri(directory))
    mlflow.set_registry_uri(tracking_uri(directory))
    return mlflow


def panel_fingerprint(path: Path | None = None) -> str:
    """Empreinte SHA-256 (16 premiers caractères) du panel ayant servi à l'entraînement."""
    p = path or DATA_PROCESSED / "panel.parquet"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def register(
    model: QuantileForecaster,
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    directory: Path | None = None,
) -> str:
    """Enregistre une nouvelle version ; renvoie son numéro."""
    from mlflow import MlflowClient

    if model.trained_on is None:
        raise ValueError("modèle non entraîné : rien à enregistrer")
    mlflow = _mlflow(directory)
    client = MlflowClient()
    d = directory or TRACKING_DIR
    exp = client.get_experiment_by_name("zonefleet")
    exp_id = (
        exp.experiment_id
        if exp
        else client.create_experiment("zonefleet", artifact_location=(d / "artifacts").as_uri())
    )
    with mlflow.start_run(experiment_id=exp_id) as run:
        mlflow.log_params({"trained_from": model.trained_on[0], "trained_to": model.trained_on[1]})
        mlflow.log_params({f"lgbm_{k}": v for k, v in model.params.items()})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"panel_sha256_16": panel_fingerprint(), **(tags or {})})
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "model.pkl"
            f.write_bytes(pickle.dumps(model))
            mlflow.log_artifact(str(f), artifact_path="model")
        source = f"{run.info.artifact_uri}/model"
    if not client.search_registered_models(filter_string=f"name='{MODEL_NAME}'"):
        client.create_registered_model(MODEL_NAME)
    mv = client.create_model_version(MODEL_NAME, source=source, run_id=run.info.run_id)
    return str(mv.version)


def set_alias(alias: str, version: str, directory: Path | None = None) -> None:
    from mlflow import MlflowClient

    _mlflow(directory)
    MlflowClient().set_registered_model_alias(MODEL_NAME, alias, version)


def load(alias: str = "champion", directory: Path | None = None) -> QuantileForecaster:
    """Charge le modèle désigné par un alias."""
    from mlflow import MlflowClient

    mlflow = _mlflow(directory)
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, alias)
    local = mlflow.artifacts.download_artifacts(f"runs:/{mv.run_id}/model/model.pkl")
    return pickle.loads(Path(local).read_bytes())


def promote_if_better(
    champion_error: float,
    challenger_error: float,
    challenger_version: str,
    margin: float = 0.05,
    directory: Path | None = None,
) -> bool:
    """Promeut le challenger si son erreur est inférieure d'au moins `margin` (relatif)."""
    if challenger_error < (1 - margin) * champion_error:
        set_alias("champion", challenger_version, directory)
        return True
    set_alias("challenger", challenger_version, directory)
    return False
