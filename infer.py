import argparse
from pathlib import Path
from typing import List

import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OminiControl/FLUX inference with one LoRA control checkpoint."
    )
    parser.add_argument("--base-model", required=True, help="Path or Hugging Face id of the FLUX base model.")
    parser.add_argument("--lora", required=True, help="Path to the LoRA checkpoint, e.g. default.safetensors.")
    parser.add_argument("--condition-image", help="Single condition image.")
    parser.add_argument("--input-dir", help="Directory containing condition images for batch inference.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated images.")
    parser.add_argument("--prompt", default="A photo of a human face", help="Text prompt used for all inputs.")
    parser.add_argument("--negative-prompt", default=None, help="Reserved for compatibility; FLUX does not use it here.")
    parser.add_argument("--adapter-name", default="deblurring", help="Adapter name used when loading the LoRA.")
    parser.add_argument(
        "--preprocess",
        choices=["none", "deblurring", "canny", "coloring", "depth"],
        default="none",
        help="Optional preprocessing applied to each input before conditioning.",
    )
    parser.add_argument("--width", type=int, default=512, help="Output width.")
    parser.add_argument("--height", type=int, default=512, help="Output height.")
    parser.add_argument("--steps", type=int, default=28, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=3.0, help="Text guidance scale.")
    parser.add_argument("--image-guidance-scale", type=float, default=1.0, help="Condition guidance scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype.",
    )
    parser.add_argument("--save-condition", action="store_true", help="Save the resized/preprocessed condition image.")
    parser.add_argument("--concat", action="store_true", help="Save side-by-side condition/result images.")
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def iter_images(condition_image: str | None, input_dir: str | None) -> List[Path]:
    if bool(condition_image) == bool(input_dir):
        raise ValueError("Please pass exactly one of --condition-image or --input-dir.")

    if condition_image:
        image_path = Path(condition_image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Condition image not found: {image_path}")
        return [image_path]

    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_path}")
    images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_path}")
    return images


def prepare_condition(image_path: Path, size: tuple[int, int], preprocess: str):
    from PIL import Image
    from omini.pipeline.flux_omini import convert_to_condition

    image = Image.open(image_path).convert("RGB").resize(size)
    if preprocess == "none":
        return image
    return convert_to_condition(preprocess, image).resize(size)


def save_concat(condition, result, output_path: Path) -> None:
    from PIL import Image

    canvas = Image.new("RGB", (condition.width + result.width, max(condition.height, result.height)))
    canvas.paste(condition, (0, 0))
    canvas.paste(result, (condition.width, 0))
    canvas.save(output_path)


def main() -> None:
    args = parse_args()

    from diffusers.pipelines import FluxPipeline
    from tqdm.auto import tqdm
    from omini.pipeline.flux_omini import Condition, generate, seed_everything

    image_paths = iter_images(args.condition_image, args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = FluxPipeline.from_pretrained(args.base_model, torch_dtype=get_dtype(args.dtype))
    pipe = pipe.to(args.device)
    pipe.load_lora_weights(args.lora, adapter_name=args.adapter_name)
    pipe.set_adapters(args.adapter_name)

    size = (args.width, args.height)
    for image_path in tqdm(image_paths, desc="Generating"):
        condition_image = prepare_condition(image_path, size, args.preprocess)
        condition = Condition(condition_image, args.adapter_name)

        seed_everything(args.seed)
        result = generate(
            pipe,
            prompt=args.prompt,
            conditions=[condition],
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
        ).images[0]

        stem = image_path.stem
        result_path = output_dir / f"{stem}_seed{args.seed}_scale{args.guidance_scale:g}.png"
        result.save(result_path)

        if args.save_condition:
            condition_image.save(output_dir / f"{stem}_condition.png")
        if args.concat:
            save_concat(condition_image, result, output_dir / f"{stem}_concat.png")

        print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
