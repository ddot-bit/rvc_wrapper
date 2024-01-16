# Built-in imports
import os

# External imports
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import soundfile as sf

# Local imports
from src.transform.paths import SpleeterOutPath


class SpleeterProcess:
    SAMPLE_RATE = 44100
    PROCESS_NAME = "spleeter"

    def __init__(self, audio_out_dir: str, orig_song_path: str, filename: str) -> None:
        self.output_dir = audio_out_dir
        self.filename = filename
        # Model type
        self.separator = Separator("spleeter:2stems", multiprocess=True)

    def extract_audio(self) -> SpleeterOutPath:
        waveform = self.load_audio()
        audio_prediction = self.split_audio(waveform)
        del waveform
        # Keys defined by spleeter lib
        main_vocals_path, accompaniment_path = self.save_audio(
            audio_prediction["vocals"], audio_prediction["accompaniment"]
        )
        del self.separator, audio_prediction
        return SpleeterOutPath(
            orig_song_path=self.output_dir,
            main_vocals_path=main_vocals_path,
            accompiant_path=accompaniment_path,
        )

    def load_audio(self):
        audio_loader = AudioAdapter.default()
        path = "/Users/diegopuducay/repos/rvc_wrapper/song_output/qxJ_f67YW8Y/21 Savage - A Lot (Lyrics) (rick Ver).mp3"
        waveform, _ = audio_loader.load(path, sample_rate=self.SAMPLE_RATE)
        return waveform

    def split_audio(self, waveform: dict) -> dict:
        # Perform the separation:
        prediction = self.separator.separate(waveform)
        return prediction

    def save_audio(self, vocals, accompaniment) -> tuple[str, str]:
        main_filepath = lambda file_suffix: os.path.join(
            self.output_dir,
            f"{os.path.basename(os.path.splitext(self.filename)[0])}_{self.PROCESS_NAME}_{file_suffix}.wav",
        )
        main_vocals_path = main_filepath("vocals")
        accompaniment_path = main_filepath("accompaniment")
        sf.write(main_vocals_path, vocals, self.SAMPLE_RATE)
        sf.write(accompaniment_path, accompaniment, self.SAMPLE_RATE)
        return main_vocals_path, accompaniment_path
