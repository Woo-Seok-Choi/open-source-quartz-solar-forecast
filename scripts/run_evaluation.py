""" Make script to run the evaluation on the test set.

The idea is to run the model on the test set and then compare the results to the actual PV generation.
The NWP (ICON) and PV data are both pulled Open Climate Fix's Hugging Face page.

Please note it can take hours to pull the NWP data.
The data will be cached locally so next time you run it, itll be much quicker
"""

import os

from quartz_solar_forecast.evaluation import run_eval, run_eval_nwp_transformer

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), "..")

    testset_path = os.path.join(base_dir, "quartz_solar_forecast", "dataset", "testset.csv")
    models_dir = os.path.join(base_dir, "quartz_solar_forecast", "models")

    models = {
        "v1 (model-0.3.0)": os.path.join(models_dir, "model-0.3.0.pkl"),
        "v1-tilt (model-0.4.0)": os.path.join(models_dir, "model-0.4.0.pkl"),
    }
    # Optional: evaluate the nwp_transformer checkpoint when available.
    # The .pt extension routes to run_eval_nwp_transformer, which uses
    # Open-Meteo Archive (matches training distribution) instead of ICON.
    nwp_transformer_ckpt = os.path.join(
        base_dir, "runs", "main_5000site", "best.pt",
    )
    if os.path.exists(nwp_transformer_ckpt):
        models["nwp_transformer (best.pt)"] = nwp_transformer_ckpt

    results_dir = os.path.join(base_dir, "results")

    for name, path in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}\n")
        model_name = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join(results_dir, f"{model_name}.csv")
        if path.endswith(".pt"):
            run_eval_nwp_transformer(
                testset_path=testset_path, model_path=path, output_path=output_path,
            )
        else:
            run_eval(
                testset_path=testset_path, model_path=path, output_path=output_path,
            )
