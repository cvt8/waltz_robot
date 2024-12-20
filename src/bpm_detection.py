import essentia.standard as ess


def get_bpm(file_path: str) -> float:
    """
    Analyzes the BPM of a given music file.

    Parameters:
    - file_path (str): Path to the music file.

    Returns:
    - float: The estimated BPM of the music file."""
    
    audio = ess.MonoLoader(filename=file_path)()
    bpm = ess.RhythmExtractor2013(method="multifeature")(audio)[0]
    return bpm