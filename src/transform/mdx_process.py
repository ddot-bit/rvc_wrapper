from typing import Union
import os
import json

from mdx import run_mdx, MDX, MDXModel
from transform.paths import MDXOutPath
import ray
import torch

SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ray.is_initialized():
    pass
else:
    ray.init(runtime_env={"working_dir": SOURCE_DIR})


class MdxProcess:
    def __init__(
        self,
        song_output_dir: str,
        orig_song_path: str,
        mdxnet_models_dir: str,
    ) -> None:
        self.orig_song_path = orig_song_path
        self.song_output_dir = song_output_dir

        with open(os.path.join(mdxnet_models_dir, "model_data.json")) as infile:
            self.mdx_model_params = json.load(infile)

        self.mdx_net_model_path = os.path.join(
            mdxnet_models_dir, "UVR-MDX-NET-Voc_FT.onnx"
        )
        self.mdx_kara_model_path = os.path.join(
            mdxnet_models_dir, "UVR_MDXNET_KARA_2.onnx"
        )
        self.reverb_hq_model_path = os.path.join(
            mdxnet_models_dir, "Reverb_HQ_By_FoxJoy.onnx"
        )

    def load_model(self, model_path):
        # Load of each model
        model_hash = MDX.get_hash(model_path)
        mp = self.mdx_model_params.get(model_hash)
        model_var = MDXModel(
            device=torch.device("cpu"),
            dim_f=mp["mdx_dim_f_set"],
            dim_t=2 ** mp["mdx_dim_t_set"],
            n_fft=mp["mdx_n_fft_scale_set"],
            stem_name=mp["primary_stem"],
            compensation=mp["compensate"],
        )
        model_sess = MDX(model_path, model_var)
        return model_var, model_sess

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
        mdx_net_model, mdx_net_sess = self.load_model(self.mdx_net_model_path)
        mdx_net_model_ref = ray.put(mdx_net_model)
        mdx_net_sess_ref = ray.put(mdx_net_sess)
        vocals_path, instrumentals_path = run_mdx(
            self.mdx_model_params,
            self.song_output_dir,
            self.mdx_net_model_path,
            self.orig_song_path,
            denoise=True,
            # original mp3 file from source
            keep_orig=False,
            mdx_net_model_ref=mdx_net_model_ref,
            mdx_sess_ref=mdx_net_sess_ref,
        )
        del mdx_net_model_ref, mdx_net_sess_ref
        return vocals_path, instrumentals_path

    def seperate_main_vocals_from_backup(self, vocals_path) -> tuple[str, str]:
        mdx_kara_model, mdx_kara_sess = self.load_model(self.mdx_kara_model_path)

        mdx_kara_model_ref = ray.put(mdx_kara_model)
        mdx_kara_sess_ref = ray.put(mdx_kara_sess)
        backup_vocals_path, main_vocals_path = run_mdx(
            self.mdx_model_params,
            self.song_output_dir,
            self.mdx_kara_model_path,
            vocals_path,
            suffix="Backup",
            invert_suffix="Main",
            denoise=True,
            mdx_model_ref=mdx_kara_model_ref,
            mdx_sess_ref=mdx_kara_sess_ref,
        )
        del mdx_kara_model_ref, mdx_kara_sess_ref
        return backup_vocals_path, main_vocals_path

    def apply_dereverb_to_main_vocals(self, main_vocals_path) -> tuple[str, str]:
        reverb_hq_model, reverb_hq_sess = self.load_model(self.reverb_hq_model_path)
        reverb_hq_model_ref = ray.put(reverb_hq_model)
        reverb_hq_sess_ref = ray.put(reverb_hq_sess)
        _, main_vocals_dereverb_path = run_mdx(
            self.mdx_model_params,
            self.song_output_dir,
            self.reverb_hq_model_path,
            main_vocals_path,
            invert_suffix="DeReverb",
            exclude_main=True,
            denoise=True,
            mdx_model_ref=reverb_hq_model_ref,
            mdx_sess_ref=reverb_hq_sess_ref,
        )
        del reverb_hq_model_ref, reverb_hq_sess_ref
        return _, main_vocals_dereverb_path
