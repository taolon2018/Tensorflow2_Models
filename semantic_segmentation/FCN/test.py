import argparse


def test(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--voc_path",
        type=str,
        default="/home/taolong/dataset/VOC/test/VOCdevkit/VOC2007",
    )
    args = parser.parse_args()
    test(args)
