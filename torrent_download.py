from torrentp import TorrentDownloader

torrent_file = TorrentDownloader("torrents/train.torrent", '.')
torrent_file.start_download()

torrent_file = TorrentDownloader("torrents/val.torrent", '.')
torrent_file.start_download()