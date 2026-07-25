import argparse
import sys
import torch
import os
import onnx
import numpy as np
from onnxconverter_common import float16 as onnx_float16

from promptda.promptda import PromptDA
import torch.nn as nn

def validate_onnx_model(model_path: str, bs: int, rgb_h: int, rgb_w: int, depth_h: int, depth_w: int) -> bool:
    """
    Load an ONNX model and run a tiny dummy inference to validate it.
    Returns True on success, False otherwise.
    """
    try:
        # Structural check
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"ONNX structural check failed for {model_path}: {e}")
        return False

    # Runtime validation
    try:
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        sess = ort.InferenceSession(model_path, providers=providers)

        rgb_np = np.random.randn(bs, 3, rgb_h, rgb_w).astype(np.float32)
        depth_np = np.random.randn(bs, 1, depth_h, depth_w).astype(np.float32)

        feed = {"rgb": rgb_np, "depth": depth_np}
        session_inputs = [i.name for i in sess.get_inputs()]
        if not all(name in session_inputs for name in feed.keys()):
            # Fallback mapping by order
            feed = {session_inputs[0]: rgb_np, session_inputs[1]: depth_np}

        outputs = sess.run(None, feed)
        if len(outputs) == 0:
            print(f"Warning: Validation produced no outputs for {model_path}.")
            return False
        out0 = outputs[0]
        print(
            f"Validation OK for {model_path}. Output[0] shape={getattr(out0, 'shape', None)}, dtype={getattr(out0, 'dtype', None)}"
        )
        return True
    except ImportError:
        print("onnxruntime not installed; skipping runtime validation. Install with: pip install onnxruntime")
        return True  # Structural check passed
    except Exception as e:
        print(f"ONNX runtime validation failed for {model_path}: {e}")
        return False

def convert_onnx_to_fp16(fp32_path: str, fp16_path: str, keep_io_types: bool = True, op_block_list=None) -> str:
    """
    Convert an ONNX model from FP32 to FP16, optionally keeping IO tensors as FP32.
    Requires: onnxconverter-common
    """
    if onnx_float16 is None:
        raise RuntimeError(
            "onnxconverter-common is required for FP16 conversion. "
            "Install it with: pip install onnxconverter-common"
        )
    model = onnx.load(fp32_path)
    if op_block_list is None:
        op_block_list = []
    model_fp16 = onnx_float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        # Skip converting selected ops (e.g., Cast) to prevent type mismatches.
        op_block_list=op_block_list,
    )
    # Validate the converted graph
    onnx.checker.check_model(model_fp16)
    onnx.save(model_fp16, fp16_path)
    return fp16_path

def export_promptda(
    model,
    output_path,
    batch_size=1,
    height=640,
    width=480,
    depth_height=640,
    depth_width=480,
    rotate=False,
    dynamic=False,
    fp16=False,
    opset=21,
    device=torch.device("cpu")
):
    model.eval()

    # Determine dummy input shapes and dtype
    bs = batch_size if batch_size and batch_size > 0 else 1
    rgb_h = height if height and height > 0 else 518
    rgb_w = width if width and width > 0 else 518
    depth_h = depth_height if depth_height and depth_height > 0 else rgb_h
    depth_w = depth_width if depth_width and depth_width > 0 else rgb_w
    
    # Check aspect ratio consistency
    rgb_aspect = rgb_w / rgb_h
    depth_aspect = depth_w / depth_h
    if abs(rgb_aspect - depth_aspect) > 0.01:  # Allow small floating point differences
        print(f"Warning: RGB aspect ratio ({rgb_aspect:.3f}) differs from depth aspect ratio ({depth_aspect:.3f})")
    
    dtype = torch.float32
    dummy_rgb = torch.randn(bs, 3, rgb_h, rgb_w, dtype=dtype).to(device)
    dummy_depth = torch.randn(bs, 1, depth_h, depth_w, dtype=dtype).to(device)

    # Prepare model to export; if fp16 requested and CUDA available, wrap model to run internal ops in FP16
    model_to_export = model
    used_fp16_wrapper = False
    if fp16 and (str(device) == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda')):
        class FP16IOWrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner.half()  # convert weights/buffers to FP16

            def forward(self, rgb, depth, rotate: bool = False):
                rgb_half = rgb.to(dtype=torch.float16)
                depth_half = depth.to(dtype=torch.float16)
                out = self.inner(rgb_half, depth_half, rotate)
                # keep output as FP32 to preserve I/O types
                return out.to(dtype=torch.float32)

        model_to_export = FP16IOWrapper(model).to(device)
        used_fp16_wrapper = True
    
    # Configure dynamic axes if requested
    if dynamic or batch_size == 0 or height == 0 or width == 0:
        dynamic_axes = {
            "rgb":   {0: "batch_size", 2: "rgb_height", 3: "rgb_width"},
            "depth": {0: "batch_size", 2: "depth_height", 3: "depth_width"},
            "output": {0: "batch_size", 2: "output_height", 3: "output_width"}
        }
    else:
        dynamic_axes = None

    print(f"Exporting with batch_size={bs}, RGB: {rgb_h}x{rgb_w}, depth: {depth_h}x{depth_w}, opset={opset}, dynamic={dynamic}")
    onnx_program = torch.onnx.export(
        model_to_export,
        (dummy_rgb, dummy_depth, rotate),
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["rgb", "depth"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=True
    )
    print(f"Model exported to {output_path} as FP32")

    # Try to optimize and save via exporter API if available
    optimized_output_path = None
    try:
        if hasattr(onnx_program, "optimize") and hasattr(onnx_program, "save"):
            onnx_program.optimize()
            optimized_output_path = output_path.replace(".onnx", "_optimized.onnx")
            onnx_program.save(optimized_output_path)
            print(f"Optimized ONNX model saved to {optimized_output_path}")
    except Exception as e:
        print(f"Skipping exporter-driven optimization: {e}")

    # Validate the exported ONNX by running a dummy inference
    validate_onnx_model(output_path, bs, rgb_h, rgb_w, depth_h, depth_w)
    if optimized_output_path and os.path.isfile(optimized_output_path):
        validate_onnx_model(optimized_output_path, bs, rgb_h, rgb_w, depth_h, depth_w)

    # If fp16 is requested and we didn't use the fast CUDA wrapper, fall back to post-export conversion without op_block_list
    breakpoint()
    if fp16 and not used_fp16_wrapper:
        try:
            print("Converting model to mixed precision (FP16) via post-export conversion (no op_block_list)...")
            output_fp16_path = output_path.replace(".onnx", "_fp16.onnx")
            # Avoid op_block_list to prevent hangs
            convert_onnx_to_fp16(output_path, output_fp16_path, keep_io_types=True)
            print(f"Model converted to mixed precision (FP16 weights, FP32 I/O) and saved to {output_fp16_path}.")
            validate_onnx_model(output_fp16_path, bs, rgb_h, rgb_w, depth_h, depth_w)
        except Exception as e:
            print(f"Failed to convert model to FP16 via post-export path: {e}")
            print("Proceeding with FP32 model:", output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PromptDA to ONNX")
    parser.add_argument("--load-from", type=str, required=True,
                        help="Path to the .pth checkpoint for PromptDA.")
    parser.add_argument("--encoder", type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help="Type of ViT encoder in the PromptDA model.")
    parser.add_argument("--max-depth", type=float, default=20.0,
                        help="Max depth value for PromptDA.")

    # ONNX arguments
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy input (use 0 for dynamic batch)")
    parser.add_argument("--height", type=int, default=518, help="Input RGB image height (use 0 for dynamic height)")
    parser.add_argument("--width", type=int, default=518, help="Input RGB image width (use 0 for dynamic width)")
    parser.add_argument("--depth-width", type=int, default=518, help="Input depth width (use 0 to match RGB width)") 
    parser.add_argument("--depth-height", type=int, default=518, help="Input depth height (use 0 to match RGB height)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes for batch, height, and width")
    parser.add_argument("--fp16", action="store_true", help="Export model in FP16 half-precision")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version to use")
    parser.add_argument("--output", type=str, default="PromptDA.onnx", help="Output ONNX file path")
    parser.add_argument("--rotate", action="store_true", help="Rotate the input image and depth by 90 degrees clockwise")
    args = parser.parse_args()
    # set device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # load checkpoint
    if os.path.isfile(args.load_from):
        model = PromptDA.from_pretrained(args.load_from)
    else:
        print(f"model_path={args.load_from} not found!")
        sys.exit(1)

    model.to(DEVICE)

    # Export
    export_promptda(
        model=model,
        output_path=args.output,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        depth_height=args.depth_height,
        depth_width=args.depth_width,
        rotate=args.rotate,
        dynamic=args.dynamic,
        fp16=args.fp16,
        opset=args.opset,
        device=DEVICE
    )