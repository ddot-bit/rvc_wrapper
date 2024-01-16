from typing import Union
import os

from mdx import run_mdx
from transform.paths import MDXOutPath


class MdxProcess:
    def __init__(
        self,
        song_output_dir: str,
        orig_song_path: str,
        mdxnet_models_dir: str,
        mdx_model_params,
    ) -> None:
        self.orig_song_path = orig_song_path
        self.song_output_dir = song_output_dir
        self.mdx_model_params = mdx_model_params
        self.mdx_net_model_path = os.path.join(
            mdxnet_models_dir, "UVR-MDX-NET-Voc_FT.onnx"
        )
        self.mdx_kara_model_path = os.path.join(
            mdxnet_models_dir, "UVR_MDXNET_KARA_2.onnx"
        )
        self.reverb_hq_model_path = os.path.join(
            mdxnet_models_dir, "Reverb_HQ_By_FoxJoy.onnx"
        )

    def extract_audio(self) -> MDXOutPath:
        vocals_path, instrumentals_path = self.seperate_vocals_from_instrumental()
        backup_vocals_path, main_vocals_path = self.seperate_main_vocals_from_backup(
            vocals_path
        )
        _, main_vocals_dereverb_path = self.apply_dereverb_to_main_vocals(
            main_vocals_path
        )
        return MDXOutPath(
            orig_song_path=self.orig_song_path,
            vocals_path=vocals_path,
            instrumentals_path=instrumentals_path,
            main_vocals_path=main_vocals_path,
            backup_vocals_path=backup_vocals_path,
            main_vocals_dereverb_path=main_vocals_dereverb_path,
        )

    def seperate_vocals_from_instrumental(self) -> tuple[str, str]:
        vocals_path, instrumentals_path = run_mdx(
            self.mdx_model_params,
            self.song_output_dir,
            self.mdx_net_model_path,
            self.orig_song_path,
            denoise=True,
            # original mp3 file from source
            keep_orig=False,
        )
        return vocals_path, instrumentals_path

    def seperate_main_vocals_from_backup(self, vocals_path) -> tuple[str, str]:
        backup_vocals_path, main_vocals_path = run_mdx(
            self.mdx_model_params,
            self.song_output_dir,
            self.mdx_kara_model_path,
            vocals_path,
            suffix="Backup",
            invert_suffix="Main",
            denoise=True,
        )
        return backup_vocals_path, main_vocals_path

    def apply_dereverb_to_main_vocals(self, main_vocals_path) -> tuple[str, str]:
        _, main_vocals_dereverb_path = run_mdx(
            self.mdx_model_params,
            self.song_output_dir,
            self.reverb_hq_model_path,
            main_vocals_path,
            invert_suffix="DeReverb",
            exclude_main=True,
            denoise=True,
        )
        return _, main_vocals_dereverb_path
