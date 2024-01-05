from torrentp import TorrentDownloader
import os, sys
import argparse, requests
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument('--output_path', required=True,
                    help='path to ImageNet-P folder that contains tar files')

FLAGS, FIRE_FLAGS = parser.parse_known_args()

if not os.path.exists(FLAGS.output_path):
    os.mkdir(FLAGS.output_path)

f = open(f"{FLAGS.output_path}/test.txt", "w")

for kw in ["blur", "digital", "noise", "weather"]:

    url = f"https://zenodo.org/records/3565846/files/{kw}.tar?download=1"
    filepath = f"{FLAGS.output_path}/{kw}.tar"

    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(filepath, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")