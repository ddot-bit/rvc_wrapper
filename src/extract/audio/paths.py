from typing import Union

from extract.audio.enums import ProcessType


class MDXOutPath:
    def __init__(
        self,
        orig_song_path,
        vocals_path,
        instrumentals_path,
        main_vocals_path,
        backup_vocals_path,
        main_vocals_dereverb_path,
    ) -> None:
        self.orig_song_path: str = orig_song_path
        self.vocals_path: str = vocals_path
        self.instrumentals_path: str = instrumentals_path
        self.main_vocals_path: str = main_vocals_path
        self.backup_vocals_path: str = backup_vocals_path
        self.main_vocals_dereverb_path: str = main_vocals_dereverb_path


class OtherPath:
    def __init__(self, orig_song_path, main_vocals_path) -> None:
        self.orig_song_path = orig_song_path
        self.main_vocals_path = main_vocals_path


def get_audio_path(
    process_type: ProcessType, **audio_paths
) -> Union[MDXOutPath, OtherPath]:
    out_path_factory = {
        ProcessType.MDX_PROCESS.value: MDXOutPath,
        ProcessType.OTHER_PROCESS.value: OtherPath,
    }
    return out_path_factory[process_type](**audio_paths)
