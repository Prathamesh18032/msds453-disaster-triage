"""Explicit runtime inventories; the Kaggle profile is user-reported, not GPU-validated."""
import importlib.metadata as metadata

MACOS_312 = {
    "numpy": "2.5.5", "scipy": "1.18.1", "scikit-learn": "1.9.1",
    "torch": "2.15.0", "transformers": "5.23.0", "huggingface-hub": "2.4.0",
    "safetensors": "0.8.0", "datasketch": "2.0.0",
}
KAGGLE_313 = {
    "numpy": "2.1.3", "scipy": "1.16.3", "scikit-learn": "1.6.1",
    "torch": "2.11.0+cu130", "transformers": "5.20.2", "huggingface-hub": "1.33.0",
    "safetensors": "0.8.0", "datasketch": "2.0.0",
}
PROFILES = {
    "macos-py312": {"python": "3.12.16", "packages": MACOS_312,
                    "evidence": "local synthetic validation"},
    "kaggle-py313": {"python": "3.13.16", "packages": KAGGLE_313,
                     "evidence": "user-reported inventory; execution validation pending"},
}

def installed_versions():
    result = {}
    for name in MACOS_312:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result

def profile_errors(name, python_version, installed, *, allow_missing_datasketch=False):
    profile = PROFILES[name]
    errors = []
    if python_version != profile["python"]:
        errors.append(f"Python {python_version}; expected {profile['python']}")
    for package, expected in profile["packages"].items():
        actual = installed.get(package)
        if package == "datasketch" and actual is None and allow_missing_datasketch:
            continue
        if actual != expected:
            errors.append(f"{package} {actual or 'missing'}; expected {expected}")
    return errors
