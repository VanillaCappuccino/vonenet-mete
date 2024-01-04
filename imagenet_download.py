from torrentp import TorrentDownloader
import os, sys
import argparse, requests

# installs full imagenet. script probably won't be necessary but just to keep it here.

parser = argparse.ArgumentParser()

parser.add_argument('--output_path', required=True,
                    help='path to ImageNet folder that contains tar files')
parser.add_argument("-t", "--torrent_path", required=True,
                    help = "path for imagenet download torrents")

FLAGS, FIRE_FLAGS = parser.parse_known_args()

train_link = "https://academictorrents.com/download/a306397ccf9c2ead27155983c254227c0fd938e2.torrent"
val_link = "https://academictorrents.com/download/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5.torrent"

if not os.path.exists(FLAGS.torrent_path):
    os.mkdir(FLAGS.torrent_path)

if not os.path.exists(f"{FLAGS.torrent_path}/train.torrent"):
    trainee = requests.get(train_link)
    open(f"{FLAGS.torrent_path}/train.torrent", "wb").write(trainee.content)

if not os.path.exists(f"{FLAGS.torrent_path}/val.torrent"):
    valee = requests.get(val_link)
    open(f"{FLAGS.torrent_path}/val.torrent", "wb").write(valee.content)

if not os.path.exists(FLAGS.output_path):
    os.mkdir(FLAGS.output_path)

if not os.path.exists(FLAGS.output_path + "/train"):
    # print("Creating train path")
    os.mkdir(FLAGS.output_path + "/train")

if not os.path.exists(FLAGS.output_path + "/test"):
    # print("Creating test path")
    os.mkdir(FLAGS.output_path + "/test")

if not os.path.exists(f"{FLAGS.output_path}/ILSVRC2012_img_train.tar"):

    torrent_file = TorrentDownloader("torrents/train.torrent", f'{FLAGS.output_path}')
    torrent_file.start_download()

if not os.path.exists(f"{FLAGS.output_path}/ILSVRC2012_img_val.tar"):

    torrent_file = TorrentDownloader("torrents/val.torrent", f'{FLAGS.output_path}')
    torrent_file.start_download()