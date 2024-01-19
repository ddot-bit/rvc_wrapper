import gc
import hashlib
import os
import warnings

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torch.multiprocessing as mp

import ray
from torch.multiprocessing import Queue
from tqdm import tqdm

warnings.filterwarnings("ignore")
stem_naming = {
    "Vocals": "Instrumental",
    "Other": "Instruments",
    "Instrumental": "Vocals",
    "Drums": "Drumless",
    "Bass": "Bassless",
}
TOTAL_CPUS = mp.cpu_count()
TOTAL_GPUS = torch.cuda.device_count()
NUM_PROCESSES = max(TOTAL_GPUS // 2, 1)
if ray.is_initialized():
    pass
else:
    ray.init(num_cpus=NUM_PROCESSES, num_gpus=TOTAL_GPUS / NUM_PROCESSES)


class WrapInferenceSession:
    def __init__(self, model_path, providers):
        """Allow InferenceSession to be serializable
        Refrence:
            - https://github.com/microsoft/onnxruntime/pull/800#issuecomment-844326099
            - https://github.com/microsoft/onnxruntime/issues/7846

        Args:
            onnx_bytes (_type_): _description_
        """
        self.onnx_bytes = model_path
        self.sess = ort.InferenceSession(self.onnx_bytes, providers=providers)

    def run(self, *args):
        return self.sess.run(*args)

    def __getstate__(self):
        # Object being pickeled
        return {"onnx_bytes": self.onnx_bytes}

    def __setstate__(self, values):
        # Object unpicked with some values
        self.onnx_bytes = values["onnx_bytes"]
        self.sess = ort.InferenceSession(self.onnx_bytes)


class MDXModel:
    def __init__(
        self, device, dim_f, dim_t, n_fft, hop=1024, stem_name=None, compensation=1.000
    ):
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation

        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(
            device
        )

        out_c = self.dim_c

        self.freq_pad = torch.zeros(
            [1, out_c, self.n_bins - self.dim_f, self.dim_t]
        ).to(device)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 4, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        # c = 4*2 if self.target_name=='*' else 2
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        )
        return x.reshape([-1, 2, self.chunk_size])


class MDX:
    DEFAULT_SR = 44100
    # Unit: seconds
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    DEFAULT_PROCESSOR = 0

    def __init__(self, model_path: str, params: MDXModel, processor=DEFAULT_PROCESSOR):
        # Set the device and the provider (CPU or CUDA)
        # TODO: Use generic class to populate this
        self.device = (
            torch.device(f"cuda:{processor}")
            if torch.cuda.is_available() and processor >= 0
            else torch.device("cpu")
        )
        self.provider = (
            ["CUDAExecutionProvider"] if processor >= 0 else ["CPUExecutionProvider"]
        )
        print(
            f"MDX model backend device: {self.device.type}. MDX EXC provider: {self.provider}"
        )

        self.model = params

        # Load the ONNX model using ONNX Runtime
        self.ort = WrapInferenceSession(model_path=model_path, providers=self.provider)
        # Preload the model for faster performance
        self.ort.run(
            None, {"input": torch.rand(1, 4, params.dim_f, params.dim_t).numpy()}
        )

        self.prog = None

    def process(self, spec):
        return self.ort.run(None, {"input": spec.cpu().numpy()})[0]

    @staticmethod
    def get_hash(model_path):
        try:
            with open(model_path, "rb") as f:
                f.seek(-10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except:
            model_hash = hashlib.md5(open(model_path, "rb").read()).hexdigest()

        return model_hash

    @staticmethod
    def segment(
        wave,
        combine=True,
        chunk_size=DEFAULT_CHUNK_SIZE,
        margin_size=DEFAULT_MARGIN_SIZE,
    ):
        """
        Segment or join segmented wave array

        Args:
            wave: (np.array) Wave array to be segmented or joined
            combine: (bool) If True, combines segmented wave array. If False, segments wave array.
            chunk_size: (int) Size of each segment (in samples)
            margin_size: (int) Size of margin between segments (in samples)

        Returns:
            numpy array: Segmented or joined wave array
        """

        if combine:
            processed_wave = None  # Initializing as None instead of [] for later numpy array concatenation
            for segment_count, segment in enumerate(wave):
                start = 0 if segment_count == 0 else margin_size
                end = None if segment_count == len(wave) - 1 else -margin_size
                if margin_size == 0:
                    end = None
                if processed_wave is None:  # Create array for first segment
                    processed_wave = segment[:, start:end]
                else:  # Concatenate to existing array for subsequent segments
                    processed_wave = np.concatenate(
                        (processed_wave, segment[:, start:end]), axis=-1
                    )

        else:
            processed_wave = []
            sample_count = wave.shape[-1]

            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count

            if margin_size > chunk_size:
                margin_size = chunk_size

            for segment_count, skip in enumerate(range(0, sample_count, chunk_size)):
                margin = 0 if segment_count == 0 else margin_size
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - margin

                cut = wave[:, start:end].copy()
                processed_wave.append(cut)

                if end == sample_count:
                    break

        return processed_wave

    def pad_wave(self, wave):
        """
        Pad the wave array to match the required chunk size

        Args:
            wave: (np.array) Wave array to be padded

        Returns:
            tuple: (padded_wave, pad, trim)
                - padded_wave: Padded wave array
                - pad: Number of samples that were padded
                - trim: Number of samples that were trimmed
        """
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size

        # Padded wave
        wave_p = np.concatenate(
            (np.zeros((2, trim)), wave, np.zeros((2, pad)), np.zeros((2, trim))), 1
        )

        mix_waves = []
        for i in range(0, n_sample + pad, gen_size):
            waves = np.array(wave_p[:, i : i + self.model.chunk_size])
            mix_waves.append(waves)

        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)

        return mix_waves, pad, trim

    @ray.remote(num_cpus=NUM_PROCESSES, num_gpus=TOTAL_GPUS / NUM_PROCESSES)
    def _process_wave(self, model, mix_waves, trim, pad, _id: int, q: Queue = None):
        """
        Process each wave segment in a multi-threaded environment

        Args:
            mix_waves: (torch.Tensor) Wave segments to be processed
            trim: (int) Number of samples trimmed during padding
            pad: (int) Number of samples padded during padding
            q: (Queue) Queue to hold the processed wave segments
            _id: (int) Identifier of the processed wave segment

        Returns:
            numpy array: Processed wave segment
        """
        mix_waves = mix_waves.split(1)
        with torch.no_grad():
            pw = []
            for mix_wave in tqdm(mix_waves, desc=f"[~] Multi-Process:{_id} MDX Waves"):
                spec = model.stft(mix_wave)
                processed_spec = torch.tensor(self.process(spec))
                processed_wav = model.istft(processed_spec.to(self.device))
                processed_wav = (
                    processed_wav[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .cpu()
                    .numpy()
                )
                pw.append(processed_wav)
        processed_signal = np.concatenate(pw, axis=-1)[:, :-pad]
        # q.put({_id: processed_signal})
        return {_id: processed_signal}

    def process_wave(self, wave: np.array, mt_threads=1):
        """
        Process the wave array in a multi-threaded environment

        Args:
            wave: (np.array) Wave array to be processed
            mt_threads: (int) Number of threads to be used for processing

        Returns:
            numpy array: Processed wave array
        """
        # Will use only half of cpu cores
        num_processes = NUM_PROCESSES
        # Reduce thread switching
        # https://pytorch.org/docs/stable/notes/multiprocessing.html
        # torch.set_num_threads(mp.cpu_count() // num_processes)
        chunk = wave.shape[-1] // num_processes
        waves = self.segment(wave, False, chunk)
        wave_batches = []
        for c, batch in enumerate(waves):
            mix_waves, pad, trim = self.pad_wave(batch)
            wave_batches.append(
                self._process_wave.remote(self, self.model, mix_waves, trim, pad, c)
            )

        output = ray.get(wave_batches)
        # Manually sort results by ProcessID
        processed_batches = [
            list(wave.values())[0]
            for wave in sorted(output, key=lambda d: list(d.keys())[0])
        ]

        assert len(processed_batches) == len(
            waves
        ), "Incomplete processed batches, please reduce batch size!"
        return self.segment(processed_batches, True, chunk)


def run_mdx(
    model_params,
    output_dir,
    model_path,
    filename,
    mdx_model: MDXModel,
    mdx_sess: MDX,
    exclude_main=False,
    exclude_inversion=False,
    suffix=None,
    invert_suffix=None,
    denoise=False,
    keep_orig=True,
    m_threads=2,
):
    # TODO: use generic solution to determine this

    if torch.cuda.is_available():
        print("CUDA IS AVAILABLE, SETTING BACKEND TO CUDA")
        device = torch.device("cuda:0")

        device_properties = torch.cuda.get_device_properties(device)
        vram_gb = device_properties.total_memory / 1024**3
    else:
        device = torch.device("cpu")
        vram_gb = 1

    m_threads = 1 if vram_gb < 8 else 2

    # Read models from object store
    model = ray.get(mdx_model)
    mdx_sess = ray.get(mdx_sess)

    wave, sr = librosa.load(filename, mono=False, sr=44100)
    # normalizing input wave gives better output
    peak = max(np.max(wave), abs(np.min(wave)))
    wave /= peak
    if denoise:
        wave_processed = -(mdx_sess.process_wave(-wave, m_threads)) + (
            mdx_sess.process_wave(wave, m_threads)
        )
        wave_processed *= 0.5
    else:
        wave_processed = mdx_sess.process_wave(wave, m_threads)
    # return to previous peak
    wave_processed *= peak
    stem_name = model.stem_name if suffix is None else suffix

    main_filepath = None
    if not exclude_main:
        main_filepath = os.path.join(
            output_dir,
            f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav",
        )
        sf.write(main_filepath, wave_processed.T, sr)

    invert_filepath = None
    if not exclude_inversion:
        diff_stem_name = (
            stem_naming.get(stem_name) if invert_suffix is None else invert_suffix
        )
        stem_name = f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
        invert_filepath = os.path.join(
            output_dir,
            f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav",
        )
        sf.write(invert_filepath, (-wave_processed.T * model.compensation) + wave.T, sr)

    if not keep_orig:
        os.remove(filename)

    del mdx_sess, wave_processed, wave
    gc.collect()
    return main_filepath, invert_filepath
