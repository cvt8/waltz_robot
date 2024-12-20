import essentia.standard as ess

def music_length(file_path: str) -> float:
    """
    Analyzes the length of a given music file.

    Parameters:
    - file_path (str): Path to the music file.

    Returns:
    - float: The estimated length of the music file in seconds.
    """
    audio = ess.MonoLoader(filename=file_path)()
    length = len(audio) / audio.sampleRate
    return length