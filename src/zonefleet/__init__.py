"""ZoneFleet : observabilité d un forecaster de demande zone x heure."""

import os

# MLflow 3 écrit un message d'aide horodaté, avec des chemins locaux, à chaque import :
# il finirait dans les sorties publiées des notebooks.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
