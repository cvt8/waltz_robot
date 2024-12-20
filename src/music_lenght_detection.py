import essentia.standard as ess

def music_lenght(file_path: str) -> float:
    """
    Analyzes the length of a given music file.

    Parameters:
    - file_path (str): Path to the music file.

    Returns:
    - float: The estimated length of the music file in seconds.
    """
    loader = ess.MonoLoader(filename=file_path)
    audio = loader()  # This returns a numpy array with the audio samples
    sample_rate = 44100
    
    length = len(audio) / sample_rate

    return length