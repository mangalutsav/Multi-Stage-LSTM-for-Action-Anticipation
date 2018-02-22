import os
import cv2
import subprocess

def cv2_dump_frames(fn, output_path, fmt="jpg", quality=90):

    cap = cv2.VideoCapture(fn)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    index = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        index += 1

        if fmt == "webp":
            cv2_ext = "png"
        else:
            cv2_ext = fmt

        fn = os.path.join(output_path, '%08d.%s' % (index, cv2_ext))
        cv2.imwrite(fn, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if fmt == 'webp':
            with open("/dev/null", "w") as null:
                subprocess.check_call(['cwebp', fn, '-lossless', '-noalpha', '-mt', '-o', os.path.splitext(fn)[0] + ".webp"], stdout=null, stderr=null)
                os.remove(fn)

    return index


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='convert torch lmdb to portable lmdb')

    parser.add_argument(
        "--video",
        metavar="<path>",
        required=True, type=str,
        help="input video")

    parser.add_argument(
        "--output-dir",
        metavar="<path>",
        required=True,
        type=str,
        help="output base directory")

    parser.add_argument(
        "--format",
        metavar="<path>",
        default='jpg',
        choices=['jpg', 'png', 'webp'],
        type=str,
        help="output image format")

    args = parser.parse_args()
    cv2_dump_frames(args.video, args.output_dir, args.format, 94)


