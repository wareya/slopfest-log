import argparse
import os
import sys
from PIL import Image
import ddslop

def run_conversion():
    parser = argparse.ArgumentParser(description="Convert standard images to DDS (DXT1/DXT5) using ddslop.")
    
    # Arguments
    parser.add_argument("input", help="Path to the source image (PNG, JPG, etc.)")
    parser.add_argument("-o", "--output", help="Path for the output DDS file (defaults to input name + .dds)")
    parser.add_argument("-f", "--format", choices=["DXT1", "DXT5", "BC7", "BC7lite", "BC7nano", "BC7zero"], help="DDS compression format (auto-detected if omitted)")
    parser.add_argument("--no-mips", action="store_false", dest="mipmaps", help="Disable mipmap generation")
    parser.add_argument("--mips-linear", action="store_true", dest="mips_linear", help="Generate mipmaps in linear RGB")
    
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    # Determine output path if not provided
    if not args.output:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}.dds"

    try:
        with Image.open(args.input) as img:
            print(f"Processing: {args.input}")
            print(f"Dimensions: {img.size[0]}x{img.size[1]}")
            
            final_format = args.format if args.format else ("DXT5" if img.mode in ("RGBA", "LA") else "DXT1")
            
            ddslop.save_dds(
                image=img,
                dest=args.output,
                pixel_format=final_format,
                mipmaps=args.mipmaps,
                mipmaps_linear=args.mips_linear,
                pca=3,
            )
            
            print(f"Success: Saved {args.output} [{final_format}]")

    except Exception as e:
        print(f"Failed to convert image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_conversion()
