"""Public scientific constants from execution-contract.md."""
LABELS = (
    "caution_and_advice", "displaced_people_and_evacuations",
    "infrastructure_and_utility_damage", "injured_or_dead_people",
    "missing_or_found_people", "not_humanitarian", "other_relevant_information",
    "requests_or_urgent_needs", "rescue_volunteering_or_donation_effort",
    "sympathy_and_support",
)
EVENTS = {
    "train": (
        "canada_wildfires_2016", "ecuador_earthquake_2016", "hurricane_irma_2017",
        "hurricane_maria_2017", "italy_earthquake_aug_2016",
        "kaikoura_earthquake_2016", "maryland_floods_2018", "srilanka_floods_2017",
    ),
    "development": ("greece_wildfires_2018", "hurricane_harvey_2017"),
    "final_test": (
        "cyclone_idai_2019", "puebla_mexico_earthquake_2017", "hurricane_matthew_2016",
    ),
}
ARCHIVE_SHA256 = "5407c09c20286a11a594b5907af89b8319c5b4f365cff0063c4070bed1fd1b44"
ROLES = ("model_selection", "calibration", "policy", "final_test")
MODEL_ID = "distilbert/distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
CUTOFF = "2026-11-23T23:59:00+05:30"
CLASSICAL_SETTINGS = {
    "ngram_range": [1, 2], "min_df": 2, "max_df": .95, "sublinear_tf": True,
    "max_features": 100000, "penalty": "l2", "solver": "lbfgs", "max_iter": 2000,
    "C": [.1, 1., 10.], "class_weight": [None, "balanced"],
}
