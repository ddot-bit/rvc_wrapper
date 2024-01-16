from typing import Optional


class MDXOutPath:
    def __init__(
        self,
        orig_song_path,
        instrumentals_path,
        backup_vocals_path,
        main_vocals_dereverb_path,
        vocals_path: Optional[str] = None,
        main_vocals_path: Optional[str] = None,
    ) -> None:
        self.orig_song_path: str = orig_song_path
        self.vocals_path: str = vocals_path
        self.instrumentals_path: str = instrumentals_path
        self.main_vocals_path: str = main_vocals_path
        self.backup_vocals_path: str = backup_vocals_path
        self.main_vocals_dereverb_path: str = main_vocals_dereverb_path

    @property
    def required_paths(self) -> list:
        # any files that are required to be present
        req_set = {
            "orig_song_path",
            "instrumentals_path",
            "backup_vocals_path",
            "main_vocals_dereverb_path",
        }
        return [req_file for req_file, path in vars(self).items()]


class SpleeterOutPath:
    def __init__(self, orig_song_path, main_vocals_path, accompiant_path) -> None:
        self.orig_song_path = orig_song_path
        self.main_vocals_path = main_vocals_path
        self.accompiant_path = accompiant_path
