from typing import Union

from src.transform import spleeter_process
from src.transform.enums import ProcessType
from src.transform import mdx_process


def set_audio_process(
    process_type: ProcessType, **process_params
) -> Union[mdx_process.MdxProcess, spleeter_process.SpleeterProcess]:
    out_path_factory = {
        ProcessType.MDX_PROCESS.value: mdx_process.MdxProcess,
        ProcessType.OTHER_PROCESS.value: spleeter_process.SpleeterProcess,
    }
    return out_path_factory[process_type](**process_params)
