""" Make script to run the evaluation on the test set.

The idea is to run the model on the test set and then compare the results to the actual PV generation.
The NWP (ICON) and PV data are both pulled Open Climate Fix's Hugging Face page.

Please note it can take hours to pull the NWP data.
The data will be cached locally so next time you run it, itll be much quicker
"""

import os

from quartz_solar_forecast.evaluation import run_eval

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), "..")

    testset_path = os.path.join(base_dir, "quartz_solar_forecast", "dataset", "testset.csv")
    models_dir = os.path.join(base_dir, "quartz_solar_forecast", "models")

    models = {
        "v1 (model-0.3.0)": os.path.join(models_dir, "model-0.3.0.pkl"),
        "v1-tilt (model-0.4.0)": os.path.join(models_dir, "model-0.4.0.pkl"),
    }

    for name, path in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}\n")
        run_eval(testset_path=testset_path, model_path=path)
