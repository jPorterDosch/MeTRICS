import io
import os
import struct

import numpy as np
import zlib
import imageio
import cv2
import png

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = f.read(strlen).decode("utf-8")
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack("i", f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack("i", f.read(4))[0]
            ]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, " depth frames to", output_path
        )
        for f in range(0, len(self.frames), frame_skip):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            # frombuffer replaces the deprecated fromstring(binary): identical
            # uint16 values from the same bytes. The only object-level diff is
            # frombuffer's array is a read-only view; harmless here (reshape +
            # tolist below only read it, and image_size is None for every caller
            # so the cv2.resize branch is never taken).
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            if image_size is not None:
                depth = cv2.resize(
                    depth,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            # imageio.imwrite(os.path.join(output_path, str(f) + '.png'), depth)
            with open(
                os.path.join(output_path, str(f) + ".png"), "wb"
            ) as f:  # write 16-bit
                writer = png.Writer(
                    width=depth.shape[1], height=depth.shape[0], bitdepth=16
                )
                depth = depth.reshape(-1, depth.shape[1]).tolist()
                writer.write(f, depth)

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, "color frames to", output_path
        )
        for f in range(0, len(self.frames), frame_skip):
            color = self.frames[f].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(
                    color,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            imageio.imwrite(os.path.join(output_path, str(f) + ".jpg"), color)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, "camera poses to", output_path
        )
        for f in range(0, len(self.frames), frame_skip):
            self.save_mat_to_file(
                self.frames[f].camera_to_world,
                os.path.join(output_path, str(f) + ".txt"),
            )

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("exporting camera intrinsics to", output_path)
        self.save_mat_to_file(
            self.intrinsic_color, os.path.join(output_path, "intrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_color, os.path.join(output_path, "extrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.intrinsic_depth, os.path.join(output_path, "intrinsic_depth.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_depth, os.path.join(output_path, "extrinsic_depth.txt")
        )

    # ------------------------------------------------------------------
    # Zip (inode-safe) variants of the four exporters above.
    #
    # Each *_bytes helper encodes exactly what its export_* counterpart
    # writes to disk -- byte-for-byte, verified against the loose-file
    # output -- so the two layouts are interchangeable and a tree extracted
    # either way feeds preprocess_scannet.py identically. The member names
    # mirror the on-disk relative paths (color/0.jpg, depth/0.png,
    # pose/0.txt, intrinsic/intrinsic_depth.txt, ...).
    # ------------------------------------------------------------------

    def _depth_png_bytes(self, frame_index, image_size=None):
        depth_data = self.frames[frame_index].decompress_depth(
            self.depth_compression_type
        )
        depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
            self.depth_height, self.depth_width
        )
        if image_size is not None:
            depth = cv2.resize(
                depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST
            )
        buf = io.BytesIO()
        writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16)
        writer.write(buf, depth.reshape(-1, depth.shape[1]).tolist())
        return buf.getvalue()

    def _color_jpg_bytes(self, frame_index, image_size=None):
        color = self.frames[frame_index].decompress_color(self.color_compression_type)
        if image_size is not None:
            color = cv2.resize(
                color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST
            )
        buf = io.BytesIO()
        # same PIL JPEG encoder and default parameters imageio.imwrite() picks
        # from a '.jpg' extension, so the bytes match the loose-file export
        imageio.imwrite(buf, color, format="JPEG")
        return buf.getvalue()

    @staticmethod
    def _mat_bytes(matrix):
        """Same text np.savetxt(fmt='%f') row-by-row produces on disk."""
        buf = io.StringIO()
        for line in matrix:
            np.savetxt(buf, line[np.newaxis], fmt="%f")
        return buf.getvalue().encode("utf-8")

    def export_all_to_zip(self, writer, image_size=None, frame_skip=1):
        """Write color/depth/pose/intrinsic for every frame into one open
        SceneZipWriter, replacing the four export_* directory writes."""
        for f in range(0, len(self.frames), frame_skip):
            writer.writestr(f"color/{f}.jpg", self._color_jpg_bytes(f, image_size))
            writer.writestr(f"depth/{f}.png", self._depth_png_bytes(f, image_size))
            writer.writestr(
                f"pose/{f}.txt", self._mat_bytes(self.frames[f].camera_to_world)
            )
        for name, mat in (
            ("intrinsic_color", self.intrinsic_color),
            ("extrinsic_color", self.extrinsic_color),
            ("intrinsic_depth", self.intrinsic_depth),
            ("extrinsic_depth", self.extrinsic_depth),
        ):
            writer.writestr(f"intrinsic/{name}.txt", self._mat_bytes(mat))
