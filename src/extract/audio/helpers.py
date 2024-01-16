def get_youtube_video_id(url, ignore_playlist=True):
    """
    Examples:
    http://youtu.be/SA2iWivDJiE
    http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    http://www.youtube.com/embed/SA2iWivDJiE
    http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    query = urlparse(url)
    if query.hostname == "youtu.be":
        if query.path[1:] == "watch":
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {"www.youtube.com", "youtube.com", "music.youtube.com"}:
        if not ignore_playlist:
            # use case: get playlist id not current video in playlist
            with suppress(KeyError):
                return parse_qs(query.query)["list"][0]
        if query.path == "/watch":
            return parse_qs(query.query)["v"][0]
        if query.path[:7] == "/watch/":
            return query.path.split("/")[1]
        if query.path[:7] == "/embed/":
            return query.path.split("/")[2]
        if query.path[:3] == "/v/":
            return query.path.split("/")[2]

    # returns None for invalid YouTube url
    return None


def yt_download(link):
    ydl_opts = {
        "format": "bestaudio",
        "outtmpl": "%(title)s",
        "nocheckcertificate": True,
        "ignoreerrors": True,
        "no_warnings": True,
        "quiet": True,
        "extractaudio": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result, outtmpl="%(title)s.mp3")

    return download_path


def raise_exception(error_msg, is_webui):
    if is_webui:
        raise gr.Error(error_msg)
    else:
        raise Exception(error_msg)


def get_rvc_model(voice_model, is_webui):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == ".pth":
            rvc_model_filename = file
        if ext == ".index":
            rvc_index_filename = file

    if rvc_model_filename is None:
        error_msg = f"No model file exists in {model_dir}."
        raise_exception(error_msg, is_webui)

    return (
        os.path.join(model_dir, rvc_model_filename),
        os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else "",
    )


def get_audio_paths(song_dir):
    orig_song_path = None
    instrumentals_path = None
    main_vocals_dereverb_path = None
    backup_vocals_path = None

    for file in os.listdir(song_dir):
        if file.endswith("_Instrumental.wav"):
            instrumentals_path = os.path.join(song_dir, file)
            orig_song_path = instrumentals_path.replace("_Instrumental", "")

        elif file.endswith("_Vocals_Main_DeReverb.wav"):
            main_vocals_dereverb_path = os.path.join(song_dir, file)

        elif file.endswith("_Vocals_Backup.wav"):
            backup_vocals_path = os.path.join(song_dir, file)

    return (
        orig_song_path,
        instrumentals_path,
        main_vocals_dereverb_path,
        backup_vocals_path,
    )


def convert_to_stereo(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)

    # check if mono
    if type(wave[0]) != np.ndarray:
        stereo_path = f"{os.path.splitext(audio_path)[0]}_stereo.wav"
        command = shlex.split(
            f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"'
        )
        subprocess.run(command)
        return stereo_path
    else:
        return audio_path


def pitch_shift(audio_path, pitch_change):
    output_path = f"{os.path.splitext(audio_path)[0]}_p{pitch_change}.wav"
    if not os.path.exists(output_path):
        y, sr = sf.read(audio_path)
        tfm = sox.Transformer()
        tfm.pitch(pitch_change)
        y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
        sf.write(output_path, y_shifted, sr)

    return output_path


def get_hash(filepath):
    with open(filepath, "rb") as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:11]
