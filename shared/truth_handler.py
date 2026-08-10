# truth_handler.py

from shared.evaluator import load_truth_data
from shared.ui_theme import banner


def truth_status_banner(*, sample_name: str, truth_exists: bool) -> str:
    """
    Build the banner reporting whether ground truth was found for a document

    Lives here, and is called from processor.py and sample_handler.py, because this
    exact banner was previously written out four separate times - each copy with its
    own hardcoded colour, so a change to any one of them silently diverged.

    Args:
        sample_name (str): Name of the document, as shown to the user.
        truth_exists (bool): Whether a ground truth file was found for it.

    Returns:
        str: An HTML div.
    """
    if truth_exists:
        return banner(tone="ok", text=f"Ground truth available for <b>{sample_name}</b>")

    return banner(
        tone="warn",
        text=f"No ground truth for <b>{sample_name}</b> — accuracy cannot be scored"
    )


def on_sample_selected_truth(sample_filename):
    """
    Handle sample selection and load truth data

    Args:
        sample_filename: Name of the selected sample file

    Returns:
        Tuple of (truth_data, truth_status_html)
    """
    truth_data, truth_exists = load_truth_data(sample_filename)

    return truth_data, truth_status_banner(
        sample_name=sample_filename, truth_exists=truth_exists)

# add_accuracy_column_to_results() was removed here. It built rows with the old
# "Samples Processed" / "Avg. Processing Time (s)" columns, had no caller anywhere in
# the app, and would have raised ZeroDivisionError for any engine that processed
# nothing. shared/results_table.build_run_rows() is now the only place a comparison
# row is built, for both the single-document and the batch path.